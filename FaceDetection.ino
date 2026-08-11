/*
 * FaceDetection.ino  —  MTCNN face detection on ESP32-P4
 *
 * Captures 96×96 RGB frames and runs a 2-stage cascade
 * (P-Net → R-Net) to detect face bounding boxes.
 * O-Net removed — it only adds facial landmarks (not needed here)
 * and costs ~70% of total inference time.
 *
 * Camera selection: see image_provider.h — change CAMERA_TYPE to
 * switch between IMX219 and OV5647.
 */

#include "image_provider.h"
#include "model_settings.h"
#include "mtcnn_utils.h"

// Include MTCNN model data arrays (4 models: 3 P-Net scales + R-Net).
// O-Net omitted — we only need face location, not facial landmarks.
#include "models/pnet_1.c"
#include "models/pnet_2.c"
#include "models/pnet_3.c"
#include "models/rnet.c"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ESP-NN acceleration (ESP32-P4 RISC-V SIMD)
#if __has_include("esp_nn.h") && !defined(ESP_NN)
#define ESP_NN 1
#endif

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"

/* ── Constants ─────────────────────────────────────────────────────── */

static constexpr int kDebugBaud = 921600;

// Tensor arena: shared by 4 interpreters (3 P-Net + R-Net, sequential use).
// O-Net was the biggest arena consumer; without it 120 KB is ample.
static constexpr int kTensorArenaSize = 120 * 1024;

// Inference task stack (less work without O-Net)
static constexpr uint32_t kInferenceTaskStackBytes = 32 * 1024;

/* ── Globals ───────────────────────────────────────────────────────── */

namespace {
uint8_t *tensor_arena = nullptr;

tflite::MicroInterpreter *pnet_1_interpreter = nullptr;
tflite::MicroInterpreter *pnet_2_interpreter = nullptr;
tflite::MicroInterpreter *pnet_3_interpreter = nullptr;
tflite::MicroInterpreter *rnet_interpreter  = nullptr;
}  // namespace

/* ── TFLite initialisation ─────────────────────────────────────────── */

static bool tflm_init() {
    const tflite::Model *pnet_1_model = tflite::GetModel(pnet_1_model_data);
    const tflite::Model *pnet_2_model = tflite::GetModel(pnet_2_model_data);
    const tflite::Model *pnet_3_model = tflite::GetModel(pnet_3_model_data);
    const tflite::Model *rnet_model    = tflite::GetModel(rnet_model_data);

    if (pnet_1_model->version() != TFLITE_SCHEMA_VERSION ||
        pnet_2_model->version() != TFLITE_SCHEMA_VERSION ||
        pnet_3_model->version() != TFLITE_SCHEMA_VERSION ||
        rnet_model->version()    != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: model schema version mismatch");
        return false;
    }

    tensor_arena = (uint8_t *)heap_caps_aligned_alloc(
        16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        tensor_arena = (uint8_t *)heap_caps_aligned_alloc(
            16, kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == nullptr) {
        Serial.println("ERROR: failed to allocate tensor arena");
        return false;
    }
    Serial.printf("Tensor arena: %d bytes allocated\n", kTensorArenaSize);

    // Op resolver: 10 ops (PReLU + Dequantize needed by all MTCNN models)
    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddPrelu();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddTranspose();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddDequantize();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();

    // Build 4 interpreters sharing the same arena (sequential use is safe)
    static tflite::MicroInterpreter static_pnet_1(
        pnet_1_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_pnet_2(
        pnet_2_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_pnet_3(
        pnet_3_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_rnet(
        rnet_model, micro_op_resolver, tensor_arena, kTensorArenaSize);

    pnet_1_interpreter = &static_pnet_1;
    pnet_2_interpreter = &static_pnet_2;
    pnet_3_interpreter = &static_pnet_3;
    rnet_interpreter   = &static_rnet;

    if (pnet_1_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_1 AllocateTensors failed"); return false;
    }
    if (pnet_2_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_2 AllocateTensors failed"); return false;
    }
    if (pnet_3_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_3 AllocateTensors failed"); return false;
    }
    if (rnet_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: rnet AllocateTensors failed"); return false;
    }

    return true;
}

/* ── Inference task (runs on Core 1) ───────────────────────────────── */

static void inference_task(void *arg) {
    (void)arg;
    uint16_t frame_id = 0;

    uint8_t *rgb_buf = (uint8_t *)malloc(kMaxImageSize);
    if (!rgb_buf) {
        Serial.println("ERROR: failed to allocate RGB buffer");
        vTaskDelete(nullptr);
        return;
    }

    for (;;) {
        uint64_t t_loop_start = esp_timer_get_time();

        // ── 1. Capture frame (200 ms timeout) ──
        bool updated = false;
        for (int tries = 0; tries < 100; tries++) {
            if (CameraUpdate()) { updated = true; break; }
            delayMicroseconds(2000);
        }
        if (!updated) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }

        const uint8_t *cam_rgb = CameraGetRgb();
        if (!cam_rgb) { vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        memcpy(rgb_buf, cam_rgb, kMaxImageSize);
        frame_id++;

        // ── 2. P-Net: fast full-image scan at 3 scales ──
        //
        // P-Net is a fully-convolutional 12×12 classifier.  It slides over
        // the image and scores every 12×12 window.  Three image scales
        // detect faces of different sizes:
        //   scale 0.333 → detects faces ~36 px and larger
        //   scale 0.250 → detects faces ~48 px and larger
        //   scale 0.125 → detects faces filling most of the 96×96 frame
        //
        // Threshold PNET_THRESHOLD (0.5): windows scored below 0.5 are
        // dropped immediately.  Lower → more candidates (slower but fewer
        // missed faces).  Higher → fewer candidates (faster but may miss
        // faces in poor lighting).
        uint64_t t_pnet_start = esp_timer_get_time();

        candidate_windows_t pnet_candidates;
        pnet_candidates.candidate_window = nullptr;
        pnet_candidates.len = 0;

        run_pnet(&pnet_candidates, pnet_1_interpreter, rgb_buf, IMG_W, IMG_H, PNET_1_SCALE);
        run_pnet(&pnet_candidates, pnet_2_interpreter, rgb_buf, IMG_W, IMG_H, PNET_2_SCALE);
        run_pnet(&pnet_candidates, pnet_3_interpreter, rgb_buf, IMG_W, IMG_H, PNET_3_SCALE);

        int pnet_raw_count = pnet_candidates.len;

        // NMS_THRESHOLD (0.35): overlapping boxes with IoU > 35% are merged.
        // Lower → more aggressive merging (fewer boxes per face).
        // Higher → looser merging (may keep duplicate boxes on the same face).
        nms(&pnet_candidates, NMS_THRESHOLD, IOU_MODE);

        bboxes_t *pnet_bboxes = nullptr;
        get_calibrated_boxes(&pnet_bboxes, &pnet_candidates);
        if (pnet_candidates.candidate_window) free(pnet_candidates.candidate_window);

        uint32_t pnet_time_us = (uint32_t)(esp_timer_get_time() - t_pnet_start);

        if (pnet_bboxes) {
            square_boxes(pnet_bboxes);
            correct_boxes(pnet_bboxes, IMG_W, IMG_H);
        }

        // ── 3. R-Net: refine each candidate ──
        //
        // R-Net is a 24×24 CNN with a fully-connected layer (not FCN).
        // It takes each P-Net candidate, crops that region from the
        // original image, resizes to 24×24, and classifies it again.
        // This is much more accurate than P-Net because:
        //   - Higher input resolution (24×24 vs 12×12)
        //   - Dense layers can see the whole face at once
        //
        // Threshold RNET_THRESHOLD (0.7): windows scored below 0.7 are
        // dropped.  This is stricter than P-Net because R-Net sees fewer
        // windows so it can afford to be more discriminating.
        uint64_t t_rnet_start = esp_timer_get_time();

        candidate_windows_t rnet_candidates;
        rnet_candidates.candidate_window = nullptr;
        rnet_candidates.len = 0;

        if (pnet_bboxes && pnet_bboxes->len > 0) {
            run_rnet(&rnet_candidates, rnet_interpreter, rgb_buf, IMG_W, IMG_H, pnet_bboxes);
        }
        nms(&rnet_candidates, NMS_THRESHOLD, IOU_MODE);

        bboxes_t *final_bboxes = nullptr;
        get_calibrated_boxes(&final_bboxes, &rnet_candidates);
        if (rnet_candidates.candidate_window) free(rnet_candidates.candidate_window);
        if (pnet_bboxes) { free(pnet_bboxes->bbox); free(pnet_bboxes); }

        uint32_t rnet_time_us = (uint32_t)(esp_timer_get_time() - t_rnet_start);

        if (final_bboxes) {
            square_boxes(final_bboxes);
            correct_boxes(final_bboxes, IMG_W, IMG_H);
        }

        uint32_t total_ms = (uint32_t)((esp_timer_get_time() - t_loop_start) / 1000);

        // ── 4. Print results ──
        int num_faces = (final_bboxes) ? final_bboxes->len : 0;
        Serial.printf("[frame %u] %d faces | P=%lums R=%lums total=%lums | P-Net raw=%d\n",
                      frame_id, num_faces,
                      (unsigned long)pnet_time_us / 1000,
                      (unsigned long)rnet_time_us / 1000,
                      (unsigned long)total_ms,
                      pnet_raw_count);

        if (final_bboxes) {
            for (int f = 0; f < num_faces; f++) {
                Serial.printf("  face[%d]: (%.0f, %.0f) - (%.0f, %.0f)\n",
                              f,
                              final_bboxes->bbox[f].x1, final_bboxes->bbox[f].y1,
                              final_bboxes->bbox[f].x2, final_bboxes->bbox[f].y2);
            }
            free(final_bboxes->bbox);
            free(final_bboxes);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/* ── Arduino entry points ──────────────────────────────────────────── */

void setup() {
    Serial.begin(kDebugBaud);
    delay(100);
    Serial.println("\n=== FaceDetection (MTCNN P+R, no O-Net) ===");

    Serial.printf("Camera: %s ... ", CameraGetName());
    if (!CameraBegin()) {
        Serial.println("FAILED");
        return;
    }
    Serial.println("OK");

    Serial.print("TFLite: ");
    if (!tflm_init()) {
        Serial.println("FAILED");
        return;
    }
    Serial.println("OK");

    // Diagnostics: show P-Net tensor shapes
    {
        TfLiteTensor *inp = pnet_1_interpreter->input(0);
        Serial.printf("P-Net input:  %dx%dx%d type=%d\n",
                      inp->dims->data[1], inp->dims->data[2], inp->dims->data[3], inp->type);
    }

#if defined(ESP_NN)
    Serial.println("ESP-NN: ENABLED");
#endif

    xTaskCreatePinnedToCore(
        inference_task, "mtcnn",
        kInferenceTaskStackBytes, nullptr,
        3, nullptr, 1);

    Serial.println("Running...");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
