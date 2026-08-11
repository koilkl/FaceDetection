#ifndef MTCNN_UTILS_H_
#define MTCNN_UTILS_H_

#include "tensorflow/lite/micro/micro_interpreter.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include "models/models_settings.h"

/* ── Constants ────────────────────────────────────────────────────── */
#define PNET_SIZE       12
#define RNET_SIZE       24
#define ONET_SIZE       48

#define PNET_THRESHOLD  0.5f
#define RNET_THRESHOLD  0.7f
#define ONET_THRESHOLD  0.8f

#define NMS_THRESHOLD   0.35f

#define IMG_MIN(A, B) ((A) < (B) ? (A) : (B))
#define IMG_MAX(A, B) ((A) < (B) ? (B) : (A))

/* ── Types ─────────────────────────────────────────────────────────── */
typedef enum {
    MIN_MODE = 0,
    IOU_MODE
} nms_mode_t;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
} coordinates_t;

typedef struct {
    float score;
    coordinates_t window;
    coordinates_t offsets;
} candidate_window_t;

typedef struct {
    uint8_t len;
    candidate_window_t *candidate_window;
} candidate_windows_t;

typedef struct {
    uint8_t len;
    coordinates_t *bbox;
} bboxes_t;

/* ── Core MTCNN pipeline functions ────────────────────────────────── */
void get_pnet_boxes(bboxes_t **bboxes, uint16_t width, uint16_t height, float scale);
void add_candidate_windows(candidate_windows_t *candidate_windows, float *scores, float *offsets, bboxes_t *bboxes, float threshold);
void nms(candidate_windows_t *candidate_windows, float threshold, nms_mode_t mode);
void get_calibrated_boxes(bboxes_t **bboxes, candidate_windows_t *candidate_windows);
void square_boxes(bboxes_t *bboxes);
void correct_boxes(bboxes_t *bboxes, uint16_t w, uint16_t h);

/* ── Image utilities ──────────────────────────────────────────────── */
void crop_rgb888_img(uint8_t *src, uint8_t *dst, uint16_t width, coordinates_t *coordinates);
void image_resize_linear(uint8_t *dst_image, uint8_t *src_image, int dst_w, int dst_h, int dst_c, int src_w, int src_h);

/* ── MTCNN stage runners ──────────────────────────────────────────── */
void run_pnet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter, uint8_t *img, uint16_t w, uint16_t h, float scale);
void run_rnet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter, uint8_t *img, uint16_t w, uint16_t h, bboxes_t *bboxes);
void run_onet(candidate_windows_t *candidate_windows, tflite::MicroInterpreter *interpreter, uint8_t *img, uint16_t w, uint16_t h, bboxes_t *bboxes);

/* ── Debug helpers ────────────────────────────────────────────────── */
void print_candidate_windows(candidate_windows_t *candidate_windows);
void print_bboxes(bboxes_t *bboxes);

#ifdef __cplusplus
}
#endif

#endif /* MTCNN_UTILS_H_ */
