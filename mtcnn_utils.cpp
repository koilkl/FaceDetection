/* ─────────────────────────────────────────────────────────────────────
 * mtcnn_utils.cpp  —  MTCNN face detection pipeline (Arduino / TFLM)
 *
 * Ported from Mauricio Barroso's mtcnn_esp32s3 (MIT License).
 * Changes from the original:
 *   - Fixed run_rnet / run_onet offsets indexing bug (i*2 → i*4)
 *   - Replaced stack VLAs with heap allocations
 *   - Removed ESP-IDF dependencies (esp_timer, etc.)
 *   - Simplified logging to Serial
 * ───────────────────────────────────────────────────────────────────*/

#include "mtcnn_utils.h"
#include <Arduino.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Private helpers ───────────────────────────────────────────────── */

static uint16_t get_width(coordinates_t *coordinates) {
    return (uint16_t)(coordinates->x2 - coordinates->x1 + 1.0f);
}

static uint16_t get_height(coordinates_t *coordinates) {
    return (uint16_t)(coordinates->y2 - coordinates->y1 + 1.0f);
}

static int compare_scores(const void *n1vp, const void *n2vp) {
    const candidate_window_t *n1ptr = (const candidate_window_t *)n1vp;
    const candidate_window_t *n2ptr = (const candidate_window_t *)n2vp;
    if (n1ptr->score <= n2ptr->score) return 1;
    if (n1ptr->score > n2ptr->score)  return -1;
    return 0;
}

/* ── Debug printers ────────────────────────────────────────────────── */

void print_candidate_windows(candidate_windows_t *candidate_windows) {
    for (uint8_t i = 0; i < candidate_windows->len; i++) {
        Serial.printf("%d: [score:%.3f] [x1:%.0f y1:%.0f x2:%.0f y2:%.0f]\n",
            i,
            candidate_windows->candidate_window[i].score,
            candidate_windows->candidate_window[i].window.x1,
            candidate_windows->candidate_window[i].window.y1,
            candidate_windows->candidate_window[i].window.x2,
            candidate_windows->candidate_window[i].window.y2);
    }
}

void print_bboxes(bboxes_t *bboxes) {
    for (uint8_t i = 0; i < bboxes->len; i++) {
        Serial.printf("%d: [x1:%.0f y1:%.0f x2:%.0f y2:%.0f]\n",
            i,
            bboxes->bbox[i].x1, bboxes->bbox[i].y1,
            bboxes->bbox[i].x2, bboxes->bbox[i].y2);
    }
}

/* ── P-Net box generator ───────────────────────────────────────────── */

void get_pnet_boxes(bboxes_t **bboxes, uint16_t width, uint16_t height, float scale) {
    *bboxes = (bboxes_t *)malloc(sizeof(bboxes_t));
    if (*bboxes == NULL) return;

    (*bboxes)->len = 0;

    // Calculate the grid of 12×12 windows that P-Net's outputs cover.
    // P-Net strides by 2, starting at pixel (1,1) for the 12-pixel window.
    uint8_t cols = (uint8_t)((((float)width * scale - 12.0f) / 2.0f) + 1.0f);
    uint8_t rows = (uint8_t)((((float)height * scale - 12.0f) / 2.0f) + 1.0f);

    (*bboxes)->bbox = (coordinates_t *)malloc((size_t)cols * (size_t)rows * sizeof(coordinates_t));
    if ((*bboxes)->bbox == NULL) {
        free(*bboxes);
        *bboxes = NULL;
        return;
    }

    (*bboxes)->len = (uint8_t)(cols * rows);

    for (uint16_t i = 0; i < (*bboxes)->len; i++) {
        // Each output position maps back to a 12×12 window on the original image.
        (*bboxes)->bbox[i].x1 = ceilf((float)(2 * (i % cols) + 1) / scale);
        (*bboxes)->bbox[i].y1 = ceilf((float)(2 * (i / cols) + 1) / scale);
        (*bboxes)->bbox[i].x2 = ceilf((float)(2 * (i % cols) + 1 + 12) / scale);
        (*bboxes)->bbox[i].y2 = ceilf((float)(2 * (i / cols) + 1 + 12) / scale);
    }
}

/* ── Candidate window accumulation ─────────────────────────────────── */

void add_candidate_windows(candidate_windows_t *candidate_windows, float *scores,
                           float *offsets, bboxes_t *bboxes, float threshold) {
    for (uint8_t i = 0; i < bboxes->len; i++) {
        // scores layout: [non_face, face_confidence] per position
        if (scores[(i * 2) + 1] < threshold) continue;

        // Grow the candidate list
        candidate_windows->candidate_window = (candidate_window_t *)realloc(
            candidate_windows->candidate_window,
            (size_t)(candidate_windows->len + 1) * sizeof(candidate_window_t));
        if (candidate_windows->candidate_window == NULL) continue;

        uint8_t idx = candidate_windows->len;
        candidate_windows->len++;

        candidate_windows->candidate_window[idx].score = scores[(i * 2) + 1];
        candidate_windows->candidate_window[idx].window.x1 = bboxes->bbox[i].x1;
        candidate_windows->candidate_window[idx].window.y1 = bboxes->bbox[i].y1;
        candidate_windows->candidate_window[idx].window.x2 = bboxes->bbox[i].x2;
        candidate_windows->candidate_window[idx].window.y2 = bboxes->bbox[i].y2;
        // offsets layout: [dx1, dy1, dx2, dy2] per position
        candidate_windows->candidate_window[idx].offsets.x1 = offsets[(i * 4) + 0];
        candidate_windows->candidate_window[idx].offsets.y1 = offsets[(i * 4) + 1];
        candidate_windows->candidate_window[idx].offsets.x2 = offsets[(i * 4) + 2];
        candidate_windows->candidate_window[idx].offsets.y2 = offsets[(i * 4) + 3];
    }
}

/* ── Non-Maximum Suppression (IoU mode) ────────────────────────────── */

void nms(candidate_windows_t *candidate_windows, float threshold, nms_mode_t mode) {
    if (candidate_windows->len == 0) return;

    // Sort descending by score
    qsort(candidate_windows->candidate_window, candidate_windows->len,
          sizeof(candidate_window_t), compare_scores);

    // Heap-allocate area array (was stack VLA in original)
    uint16_t *areas = (uint16_t *)malloc((size_t)candidate_windows->len * sizeof(uint16_t));
    if (areas == NULL) return;

    for (uint8_t i = 0; i < candidate_windows->len; i++) {
        areas[i] = get_width(&candidate_windows->candidate_window[i].window)
                 * get_height(&candidate_windows->candidate_window[i].window);
    }

    for (uint8_t i = 0; i < candidate_windows->len; i++) {
        uint8_t counter = i + 1;
        for (uint8_t j = i + 1; j < candidate_windows->len; j++) {
            uint8_t ix1 = IMG_MAX(candidate_windows->candidate_window[i].window.x1,
                                  candidate_windows->candidate_window[j].window.x1);
            uint8_t iy1 = IMG_MAX(candidate_windows->candidate_window[i].window.y1,
                                  candidate_windows->candidate_window[j].window.y1);
            uint8_t ix2 = IMG_MIN(candidate_windows->candidate_window[i].window.x2,
                                  candidate_windows->candidate_window[j].window.x2);
            uint8_t iy2 = IMG_MIN(candidate_windows->candidate_window[i].window.y2,
                                  candidate_windows->candidate_window[j].window.y2);

            uint16_t iarea = IMG_MAX(0.0f, (float)ix2 - (float)ix1 + 1.0f)
                           * IMG_MAX(0.0f, (float)iy2 - (float)iy1 + 1.0f);

            float overlap;
            if (mode == MIN_MODE) {
                overlap = (float)iarea / (float)IMG_MIN(areas[i], areas[j]);
            } else {
                overlap = (float)iarea / (float)(areas[i] + areas[j] - iarea);
            }

            // Keep this candidate if overlap is low enough
            if (overlap < threshold) {
                memcpy(&candidate_windows->candidate_window[counter],
                       &candidate_windows->candidate_window[j],
                       sizeof(candidate_window_t));
                counter++;
            }
        }
        candidate_windows->candidate_window = (candidate_window_t *)realloc(
            candidate_windows->candidate_window, (size_t)counter * sizeof(candidate_window_t));
        candidate_windows->len = counter;
    }

    free(areas);
}

/* ── Bounding-box calibration ──────────────────────────────────────── */

void get_calibrated_boxes(bboxes_t **bboxes, candidate_windows_t *candidate_windows) {
    *bboxes = (bboxes_t *)malloc(sizeof(bboxes_t));
    if (*bboxes == NULL) return;

    (*bboxes)->len = candidate_windows->len;
    (*bboxes)->bbox = (coordinates_t *)malloc((size_t)(*bboxes)->len * sizeof(coordinates_t));
    if ((*bboxes)->bbox == NULL) {
        free(*bboxes);
        *bboxes = NULL;
        return;
    }

    // Copy window coordinates
    for (uint8_t i = 0; i < (*bboxes)->len; i++) {
        (*bboxes)->bbox[i] = candidate_windows->candidate_window[i].window;
    }

    // Apply regression offsets
    for (uint8_t i = 0; i < (*bboxes)->len; i++) {
        uint16_t w = get_width(&(*bboxes)->bbox[i]);
        uint16_t h = get_height(&(*bboxes)->bbox[i]);

        // Clamp extreme offsets (original quirk: offsets > 1.0 → 0)
        if (candidate_windows->candidate_window[i].offsets.x1 > 1.0f)
            candidate_windows->candidate_window[i].offsets.x1 = 0.0f;
        if (candidate_windows->candidate_window[i].offsets.y1 > 1.0f)
            candidate_windows->candidate_window[i].offsets.y1 = 0.0f;
        if (candidate_windows->candidate_window[i].offsets.x2 > 1.0f)
            candidate_windows->candidate_window[i].offsets.x2 = 0.0f;
        if (candidate_windows->candidate_window[i].offsets.y2 > 1.0f)
            candidate_windows->candidate_window[i].offsets.y2 = 0.0f;

        (*bboxes)->bbox[i].x1 += (float)w * candidate_windows->candidate_window[i].offsets.x1;
        (*bboxes)->bbox[i].y1 += (float)h * candidate_windows->candidate_window[i].offsets.y1;
        (*bboxes)->bbox[i].x2 += (float)w * candidate_windows->candidate_window[i].offsets.x2;
        (*bboxes)->bbox[i].y2 += (float)h * candidate_windows->candidate_window[i].offsets.y2;
    }
}

/* ── Box squaring ──────────────────────────────────────────────────── */

void square_boxes(bboxes_t *bboxes) {
    for (uint8_t i = 0; i < bboxes->len; i++) {
        uint16_t w = get_width(&bboxes->bbox[i]);
        uint16_t h = get_height(&bboxes->bbox[i]);
        uint16_t max_side = IMG_MAX(h, w);

        bboxes->bbox[i].x1 += (float)w * 0.5f - (float)max_side * 0.5f;
        bboxes->bbox[i].y1 += (float)h * 0.5f - (float)max_side * 0.5f;
        bboxes->bbox[i].x2 = bboxes->bbox[i].x1 + (float)max_side - 1.0f;
        bboxes->bbox[i].y2 = bboxes->bbox[i].y1 + (float)max_side - 1.0f;
    }
}

/* ── Clamp boxes to image bounds ───────────────────────────────────── */

void correct_boxes(bboxes_t *bboxes, uint16_t w, uint16_t h) {
    for (uint8_t i = 0; i < bboxes->len; i++) {
        if (bboxes->bbox[i].x2 > (float)(w - 1)) bboxes->bbox[i].x2 = (float)(w - 1);
        if (bboxes->bbox[i].y2 > (float)(h - 1)) bboxes->bbox[i].y2 = (float)(h - 1);
        if (bboxes->bbox[i].x1 < 0.0f) bboxes->bbox[i].x1 = 0.0f;
        if (bboxes->bbox[i].y1 < 0.0f) bboxes->bbox[i].y1 = 0.0f;
    }
}

/* ── Crop RGB888 image ─────────────────────────────────────────────── */

void crop_rgb888_img(uint8_t *src, uint8_t *dst, uint16_t width, coordinates_t *coordinates) {
    uint16_t x1 = (uint16_t)coordinates->x1;
    uint16_t y1 = (uint16_t)coordinates->y1;
    uint16_t y2 = (uint16_t)coordinates->y2;
    uint16_t crop_w = get_width(coordinates);

    for (size_t i = y1; i < y2; i++) {
        memcpy(dst + 3 * ((i - y1) * crop_w),
               src + 3 * (i * width) + (x1 * 3),
               (size_t)crop_w * 3 * sizeof(uint8_t));
    }
}

/* ── Bilinear image resize ─────────────────────────────────────────── */

static void image_zoom_in_twice(uint8_t *dimage, int dw, int dh, int dc,
                                uint8_t *simage, int sw, int sc) {
    for (int dyi = 0; dyi < dh; dyi++) {
        int _di = dyi * dw;
        int _si0 = dyi * 2 * sw;
        int _si1 = _si0 + sw;

        for (int dxi = 0; dxi < dw; dxi++) {
            int di = (_di + dxi) * dc;
            int si0 = (_si0 + dxi * 2) * sc;
            int si1 = (_si1 + dxi * 2) * sc;

            if (1 == dc) {
                dimage[di] = (uint8_t)(((int)simage[si0] + (int)simage[si0 + 1]
                                      + (int)simage[si1] + (int)simage[si1 + 1]) >> 2);
            } else if (3 == dc) {
                dimage[di]     = (uint8_t)(((int)simage[si0] + (int)simage[si0 + 3]
                                          + (int)simage[si1] + (int)simage[si1 + 3]) >> 2);
                dimage[di + 1] = (uint8_t)(((int)simage[si0 + 1] + (int)simage[si0 + 4]
                                          + (int)simage[si1 + 1] + (int)simage[si1 + 4]) >> 2);
                dimage[di + 2] = (uint8_t)(((int)simage[si0 + 2] + (int)simage[si0 + 5]
                                          + (int)simage[si1 + 2] + (int)simage[si1 + 5]) >> 2);
            } else {
                for (int dci = 0; dci < dc; dci++) {
                    dimage[di + dci] = (uint8_t)(((int)simage[si0 + dci] + (int)simage[si0 + 3 + dci]
                                                + (int)simage[si1 + dci] + (int)simage[si1 + 3 + dci]
                                                + 2) >> 2);
                }
            }
        }
    }
}

void image_resize_linear(uint8_t *dst_image, uint8_t *src_image,
                         int dst_w, int dst_h, int dst_c,
                         int src_w, int src_h) {
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)dst_h;

    // Fast path: exact 2× downscale
    if (fabsf(scale_x - 2.0f) <= 1e-6f && fabsf(scale_y - 2.0f) <= 1e-6f) {
        image_zoom_in_twice(dst_image, dst_w, dst_h, dst_c, src_image, src_w, dst_c);
        return;
    }

    int src_stride = dst_c * src_w;
    int dst_stride = dst_c * dst_w;

    for (int y = 0; y < dst_h; y++) {
        float fy0 = (float)((y + 0.5) * scale_y - 0.5);
        int src_y = (int)fy0;
        fy0 -= (float)src_y;
        float fy1 = 1.0f - fy0;
        src_y = IMG_MAX(0, src_y);
        src_y = IMG_MIN(src_y, src_h - 2);

        for (int x = 0; x < dst_w; x++) {
            float fx0 = (float)((x + 0.5) * scale_x - 0.5);
            int src_x = (int)fx0;
            fx0 -= (float)src_x;

            if (src_x < 0)       { fx0 = 0.0f; src_x = 0; }
            if (src_x > src_w - 2) { fx0 = 0.0f; src_x = src_w - 2; }
            float fx1 = 1.0f - fx0;

            for (int c = 0; c < dst_c; c++) {
                float v = (float)src_image[src_y * src_stride + src_x * dst_c + c] * fx1 * fy1
                        + (float)src_image[src_y * src_stride + (src_x + 1) * dst_c + c] * fx0 * fy1
                        + (float)src_image[(src_y + 1) * src_stride + src_x * dst_c + c] * fx1 * fy0
                        + (float)src_image[(src_y + 1) * src_stride + (src_x + 1) * dst_c + c] * fx0 * fy0;
                dst_image[y * dst_stride + x * dst_c + c] = (uint8_t)roundf(v);
            }
        }
    }
}

/* ── MTCNN stage 1: P-Net ──────────────────────────────────────────── */

void run_pnet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter,
              uint8_t *img, uint16_t w, uint16_t h, float scale) {
    uint16_t ws = (uint16_t)((float)w * scale);
    uint16_t hs = (uint16_t)((float)h * scale);

    uint8_t *imgs = (uint8_t *)malloc((size_t)ws * (size_t)hs * 3 * sizeof(uint8_t));
    if (imgs == NULL) return;
    image_resize_linear(imgs, img, ws, hs, 3, w, h);

    // Feed model: uint8 → int8
    TfLiteTensor *input = interpreter->input(0);
    for (int i = 0; i < ws * hs * 3; i++) {
        input->data.int8[i] = (int8_t)(imgs[i] ^ 0x80);
    }

    free(imgs);

    if (kTfLiteOk != interpreter->Invoke()) {
        return;
    }

    TfLiteTensor *probs   = interpreter->output(0);
    TfLiteTensor *offsets = interpreter->output(1);

    bboxes_t *bboxes = NULL;
    get_pnet_boxes(&bboxes, w, h, scale);
    if (bboxes == NULL) return;

    add_candidate_windows(candidate_windows, probs->data.f, offsets->data.f, bboxes, PNET_THRESHOLD);

    free(bboxes->bbox);
    free(bboxes);
}

/* ── MTCNN stage 2: R-Net ──────────────────────────────────────────── */

void run_rnet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter,
              uint8_t *img, uint16_t w, uint16_t h, bboxes_t *bboxes) {
    if (bboxes->len == 0) return;

    // Heap-allocate (was stack VLA in original)
    float *probs_buf   = (float *)malloc((size_t)bboxes->len * 2 * sizeof(float));
    float *offsets_buf = (float *)malloc((size_t)bboxes->len * 4 * sizeof(float));
    if (probs_buf == NULL || offsets_buf == NULL) {
        free(probs_buf);
        free(offsets_buf);
        return;
    }

    for (uint8_t i = 0; i < bboxes->len; i++) {
        uint16_t wc = get_width(&bboxes->bbox[i]);
        uint16_t hc = get_height(&bboxes->bbox[i]);

        uint8_t *imgc = (uint8_t *)malloc((size_t)wc * (size_t)hc * 3 * sizeof(uint8_t));
        if (imgc == NULL) continue;
        crop_rgb888_img(img, imgc, w, &bboxes->bbox[i]);

        uint8_t *rnet_image = (uint8_t *)malloc((size_t)RNET_SIZE * RNET_SIZE * 3 * sizeof(uint8_t));
        if (rnet_image == NULL) {
            free(imgc);
            continue;
        }
        image_resize_linear(rnet_image, imgc, RNET_SIZE, RNET_SIZE, 3, wc, hc);
        free(imgc);

        // Feed model: uint8 → int8
        TfLiteTensor *input = interpreter->input(0);
        for (int j = 0; j < RNET_SIZE * RNET_SIZE * 3; j++) {
            input->data.int8[j] = (int8_t)(rnet_image[j] ^ 0x80);
        }
        free(rnet_image);

        if (kTfLiteOk != interpreter->Invoke()) continue;

        TfLiteTensor *probs   = interpreter->output(0);
        TfLiteTensor *offsets = interpreter->output(1);

        // FIXED: i*2 → i*4 for offsets
        for (uint8_t j = 0; j < 2; j++) {
            probs_buf[j + (i * 2)] = probs->data.f[j];
        }
        for (uint8_t j = 0; j < 4; j++) {
            offsets_buf[j + (i * 4)] = offsets->data.f[j];
        }
    }

    add_candidate_windows(candidate_windows, probs_buf, offsets_buf, bboxes, RNET_THRESHOLD);
    free(probs_buf);
    free(offsets_buf);
}

/* ── MTCNN stage 3: O-Net ──────────────────────────────────────────── */

void run_onet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter,
              uint8_t *img, uint16_t w, uint16_t h, bboxes_t *bboxes) {
    if (bboxes->len == 0) return;

    // Heap-allocate (was stack VLA in original)
    float *probs_buf   = (float *)malloc((size_t)bboxes->len * 2 * sizeof(float));
    float *offsets_buf = (float *)malloc((size_t)bboxes->len * 4 * sizeof(float));
    if (probs_buf == NULL || offsets_buf == NULL) {
        free(probs_buf);
        free(offsets_buf);
        return;
    }

    for (uint8_t i = 0; i < bboxes->len; i++) {
        uint16_t wc = get_width(&bboxes->bbox[i]);
        uint16_t hc = get_height(&bboxes->bbox[i]);

        uint8_t *imgc = (uint8_t *)malloc((size_t)wc * (size_t)hc * 3 * sizeof(uint8_t));
        if (imgc == NULL) continue;
        crop_rgb888_img(img, imgc, w, &bboxes->bbox[i]);

        uint8_t *onet_image = (uint8_t *)malloc((size_t)ONET_SIZE * ONET_SIZE * 3 * sizeof(uint8_t));
        if (onet_image == NULL) {
            free(imgc);
            continue;
        }
        image_resize_linear(onet_image, imgc, ONET_SIZE, ONET_SIZE, 3, wc, hc);
        free(imgc);

        // Feed model: uint8 → int8
        TfLiteTensor *input = interpreter->input(0);
        for (int j = 0; j < ONET_SIZE * ONET_SIZE * 3; j++) {
            input->data.int8[j] = (int8_t)(onet_image[j] ^ 0x80);
        }
        free(onet_image);

        if (kTfLiteOk != interpreter->Invoke()) continue;

        TfLiteTensor *probs   = interpreter->output(0);
        TfLiteTensor *offsets = interpreter->output(1);

        // FIXED: i*2 → i*4 for offsets
        for (uint8_t j = 0; j < 2; j++) {
            probs_buf[j + (i * 2)] = probs->data.f[j];
        }
        for (uint8_t j = 0; j < 4; j++) {
            offsets_buf[j + (i * 4)] = offsets->data.f[j];
        }
    }

    add_candidate_windows(candidate_windows, probs_buf, offsets_buf, bboxes, ONET_THRESHOLD);
    free(probs_buf);
    free(offsets_buf);
}
