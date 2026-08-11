#ifndef FACEDETECT_IMAGE_PROVIDER_H_
#define FACEDETECT_IMAGE_PROVIDER_H_

#include <stdint.h>
#include <tensorflow/lite/c/common.h>

// ── Camera type selection ─────────────────────────────────────────────
// Change CAMERA_TYPE to switch between supported cameras.
//   CAMERA_TYPE_IMX219  →  IMX219 (MIPI CSI, 1536×1232 sensor)
//   CAMERA_TYPE_OV5647  →  OV5647 (MIPI CSI, 1920×1080 sensor)
#define CAMERA_TYPE_IMX219  1
#define CAMERA_TYPE_OV5647  2

#ifndef CAMERA_TYPE
#define CAMERA_TYPE CAMERA_TYPE_IMX219
#endif

// ── Output image dimensions ───────────────────────────────────────────
// IMG_SIZE must be defined BEFORE the camera library headers so both
// IMX219 and OV5647 allocate internal buffers at 96×96 (not their
// respective defaults of 96 and 160).
#ifdef IMG_SIZE
#undef IMG_SIZE
#endif
#define IMG_SIZE 96

#define OUT_WIDTH   96
#define OUT_HEIGHT  96
#define OUT_CHANNELS 3

// Output buffer size in bytes (96×96×3 RGB)
#define OUT_RGB_SIZE  (OUT_WIDTH * OUT_HEIGHT * OUT_CHANNELS)

// ── Public API ────────────────────────────────────────────────────────

// Initialise the selected camera.  Returns true on success.
bool CameraBegin();

// Capture one frame.  Returns true when a new frame is available.
bool CameraUpdate();

// Get pointer to the latest 96×96 RGB buffer.
// Buffer is OUT_RGB_SIZE bytes, interleaved R,G,B.
const uint8_t* CameraGetRgb();

// Get the RGB buffer size in bytes.
size_t CameraGetRgbSize();

// Get camera name string for logging.
const char* CameraGetName();

// ── TFLite image provider interface ───────────────────────────────────
// Returns OUT_WIDTH×OUT_HEIGHT×3 int8 RGB data.
// image_data must be pre-allocated to OUT_RGB_SIZE.
TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      int image_width, int image_height, int channels,
                      int8_t* image_data);

#endif  // FACEDETECT_IMAGE_PROVIDER_H_
