/*
 * FaceDetection.ino  —  MTCNN face detection on ESP32-P4
 *
 * Captures 96×96 RGB frames and runs the 3-stage MTCNN cascade
 * (P-Net → R-Net → O-Net) to detect face bounding boxes.
 * Outputs box coordinates and timing to the serial console.
 *
 * Camera selection: see image_provider.h — change CAMERA_TYPE to
 * switch between IMX219 and OV5647.
 */

#include "image_provider.h"
#include "model_settings.h"
#include "mtcnn_utils.h"

// Include the 5 MTCNN model data arrays.
#include "models/pnet_1.c"
#include "models/pnet_2.c"
#include "models/pnet_3.c"
#include "models/rnet.c"
#include "models/onet.c"

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

// Tensor arena: shared by all 5 interpreters (sequential use is safe).
// 200 KB should be ample; the reference uses ~149 KB on S3.
static constexpr int kTensorArenaSize = 200 * 1024;

// Inference task stack/key
static constexpr uint32_t kInferenceTaskStackBytes = 48 * 1024;

/* ── Globals ───────────────────────────────────────────────────────── */

namespace {
uint8_t *tensor_arena = nullptr;

tflite::MicroInterpreter *pnet_1_interpreter = nullptr;
tflite::MicroInterpreter *pnet_2_interpreter = nullptr;
tflite::MicroInterpreter *pnet_3_interpreter = nullptr;
tflite::MicroInterpreter *rnet_interpreter  = nullptr;
tflite::MicroInterpreter *onet_interpreter  = nullptr;
}  // namespace

/* ── TFLite initialisation ─────────────────────────────────────────── */

static bool tflm_init() {
    // Map models
    const tflite::Model *pnet_1_model = tflite::GetModel(pnet_1_model_data);
    const tflite::Model *pnet_2_model = tflite::GetModel(pnet_2_model_data);
    const tflite::Model *pnet_3_model = tflite::GetModel(pnet_3_model_data);
    const tflite::Model *rnet_model    = tflite::GetModel(rnet_model_data);
    const tflite::Model *onet_model    = tflite::GetModel(onet_model_data);

    if (pnet_1_model->version() != TFLITE_SCHEMA_VERSION ||
        pnet_2_model->version() != TFLITE_SCHEMA_VERSION ||
        pnet_3_model->version() != TFLITE_SCHEMA_VERSION ||
        rnet_model->version()    != TFLITE_SCHEMA_VERSION ||
        onet_model->version()    != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: model schema version mismatch");
        return false;
    }

    // Allocate tensor arena from PSRAM (16-byte aligned)
    tensor_arena = (uint8_t *)heap_caps_aligned_alloc(
        16, kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        // Fallback to internal SRAM
        tensor_arena = (uint8_t *)heap_caps_aligned_alloc(
            16, kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    }
    if (tensor_arena == nullptr) {
        Serial.println("ERROR: failed to allocate tensor arena");
        return false;
    }
    Serial.printf("Tensor arena: %d bytes allocated\n", kTensorArenaSize);

    // Op resolver: all 10 ops needed by MTCNN
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

    // Build 5 interpreters sharing the same arena (sequential use)
    static tflite::MicroInterpreter static_pnet_1(
        pnet_1_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_pnet_2(
        pnet_2_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_pnet_3(
        pnet_3_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_rnet(
        rnet_model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    static tflite::MicroInterpreter static_onet(
        onet_model, micro_op_resolver, tensor_arena, kTensorArenaSize);

    pnet_1_interpreter = &static_pnet_1;
    pnet_2_interpreter = &static_pnet_2;
    pnet_3_interpreter = &static_pnet_3;
    rnet_interpreter   = &static_rnet;
    onet_interpreter   = &static_onet;

    // Allocate tensors for each model
    if (pnet_1_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_1 AllocateTensors failed");
        return false;
    }
    if (pnet_2_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_2 AllocateTensors failed");
        return false;
    }
    if (pnet_3_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: pnet_3 AllocateTensors failed");
        return false;
    }
    if (rnet_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: rnet AllocateTensors failed");
        return false;
    }
    if (onet_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: onet AllocateTensors failed");
        return false;
    }

    return true;
}

/* ── Inference task (runs on Core 1) ───────────────────────────────── */

static void inference_task(void *arg) {
    (void)arg;
    uint16_t frame_id = 0;

    // Buffer for the latest RGB frame (96×96×3)
    uint8_t *rgb_buf = (uint8_t *)malloc(kMaxImageSize);
    if (!rgb_buf) {
        Serial.println("ERROR: failed to allocate RGB buffer");
        vTaskDelete(nullptr);
        return;
    }

    for (;;) {
        uint64_t t_loop_start = esp_timer_get_time();

        // ── 1. Capture frame ──
        bool updated = false;
        for (int tries = 0; tries < 100; tries++) {
            if (CameraUpdate()) {
                updated = true;
                break;
            }
            delayMicroseconds(2000);
        }
        if (!updated) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        const uint8_t *cam_rgb = CameraGetRgb();
        if (!cam_rgb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        // Copy to local buffer (camera buffer may be overwritten on next update)
        memcpy(rgb_buf, cam_rgb, kMaxImageSize);

        frame_id++;

        // ── 2. P-Net stage (3 scales) ──
        uint64_t t_pnet_start = esp_timer_get_time();

        candidate_windows_t pnet_candidates;
        pnet_candidates.candidate_window = nullptr;
        pnet_candidates.len = 0;

        run_pnet(&pnet_candidates, pnet_1_interpreter, rgb_buf, IMG_W, IMG_H, PNET_1_SCALE);
        run_pnet(&pnet_candidates, pnet_2_interpreter, rgb_buf, IMG_W, IMG_H, PNET_2_SCALE);
        run_pnet(&pnet_candidates, pnet_3_interpreter, rgb_buf, IMG_W, IMG_H, PNET_3_SCALE);

        int pnet_raw_count = pnet_candidates.len;
        nms(&pnet_candidates, NMS_THRESHOLD, IOU_MODE);

        bboxes_t *pnet_bboxes = nullptr;
        get_calibrated_boxes(&pnet_bboxes, &pnet_candidates);
        if (pnet_candidates.candidate_window) free(pnet_candidates.candidate_window);

        uint32_t pnet_time_us = (uint32_t)(esp_timer_get_time() - t_pnet_start);

        if (pnet_bboxes) {
            square_boxes(pnet_bboxes);
            correct_boxes(pnet_bboxes, IMG_W, IMG_H);
        }

        // ── 3. R-Net stage ──
        uint64_t t_rnet_start = esp_timer_get_time();

        candidate_windows_t rnet_candidates;
        rnet_candidates.candidate_window = nullptr;
        rnet_candidates.len = 0;

        if (pnet_bboxes && pnet_bboxes->len > 0) {
            run_rnet(&rnet_candidates, rnet_interpreter, rgb_buf, IMG_W, IMG_H, pnet_bboxes);
        }
        nms(&rnet_candidates, NMS_THRESHOLD, IOU_MODE);

        bboxes_t *rnet_bboxes = nullptr;
        get_calibrated_boxes(&rnet_bboxes, &rnet_candidates);
        if (rnet_candidates.candidate_window) free(rnet_candidates.candidate_window);
        if (pnet_bboxes) { free(pnet_bboxes->bbox); free(pnet_bboxes); }

        uint32_t rnet_time_us = (uint32_t)(esp_timer_get_time() - t_rnet_start);

        if (rnet_bboxes) {
            square_boxes(rnet_bboxes);
            correct_boxes(rnet_bboxes, IMG_W, IMG_H);
        }

        // ── 4. O-Net stage ──
        uint64_t t_onet_start = esp_timer_get_time();

        candidate_windows_t onet_candidates;
        onet_candidates.candidate_window = nullptr;
        onet_candidates.len = 0;

        if (rnet_bboxes && rnet_bboxes->len > 0) {
            run_onet(&onet_candidates, onet_interpreter, rgb_buf, IMG_W, IMG_H, rnet_bboxes);
        }
        nms(&onet_candidates, NMS_THRESHOLD, IOU_MODE);

        bboxes_t *onet_bboxes = nullptr;
        get_calibrated_boxes(&onet_bboxes, &onet_candidates);
        if (onet_candidates.candidate_window) free(onet_candidates.candidate_window);
        if (rnet_bboxes) { free(rnet_bboxes->bbox); free(rnet_bboxes); }

        uint32_t onet_time_us = (uint32_t)(esp_timer_get_time() - t_onet_start);

        if (onet_bboxes) {
            square_boxes(onet_bboxes);
            correct_boxes(onet_bboxes, IMG_W, IMG_H);
        }

        uint32_t total_ms = (uint32_t)((esp_timer_get_time() - t_loop_start) / 1000);

        // ── 5. Print results ──
        int num_faces = (onet_bboxes) ? onet_bboxes->len : 0;
        Serial.printf("[frame %u] MTCNN: %d faces | P=%lums R=%lums O=%lums total=%lums | raw_cands=%d\n",
                      frame_id, num_faces,
                      (unsigned long)pnet_time_us / 1000,
                      (unsigned long)rnet_time_us / 1000,
                      (unsigned long)onet_time_us / 1000,
                      (unsigned long)total_ms,
                      pnet_raw_count);

        if (onet_bboxes) {
            for (int f = 0; f < num_faces; f++) {
                Serial.printf("  face[%d]: (%.0f, %.0f) - (%.0f, %.0f)\n",
                              f,
                              onet_bboxes->bbox[f].x1, onet_bboxes->bbox[f].y1,
                              onet_bboxes->bbox[f].x2, onet_bboxes->bbox[f].y2);
            }
            free(onet_bboxes->bbox);
            free(onet_bboxes);
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

/* ── Arduino entry points ──────────────────────────────────────────── */

void setup() {
    Serial.begin(kDebugBaud);
    delay(100);
    Serial.println("\n=== FaceDetection (MTCNN on ESP32-P4) ===");

    // Camera
    Serial.printf("Camera: %s ... ", CameraGetName());
    if (!CameraBegin()) {
        Serial.println("FAILED");
        return;
    }
    Serial.println("OK");

    // TFLite
    Serial.print("TFLite: ");
    if (!tflm_init()) {
        Serial.println("FAILED");
        return;
    }
    Serial.println("OK");

    // Model diagnostics
    {
        TfLiteTensor *inp = pnet_1_interpreter->input(0);
        Serial.printf("P-Net input:  %dx%dx%d type=%d\n",
                      inp->dims->data[1], inp->dims->data[2], inp->dims->data[3], inp->type);
        TfLiteTensor *out0 = pnet_1_interpreter->output(0);
        TfLiteTensor *out1 = pnet_1_interpreter->output(1);
        Serial.printf("P-Net output: [0]=%dx%dx%d [1]=%dx%dx%d\n",
                      out0->dims->data[1], out0->dims->data[2], out0->dims->data[3],
                      out1->dims->data[1], out1->dims->data[2], out1->dims->data[3]);
    }

#if defined(ESP_NN)
    Serial.println("ESP-NN: ENABLED");
#endif

    // Launch inference task on Core 1
    xTaskCreatePinnedToCore(
        inference_task, "mtcnn",
        kInferenceTaskStackBytes, nullptr,
        3, nullptr, 1);

    Serial.println("Running...");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
