#include "image_provider.h"

#include <Arduino.h>
#include <driver/gpio.h>

#if CAMERA_TYPE == CAMERA_TYPE_IMX219
#include <ESP32_P4_IMX219.h>
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
#include <ESP32_P4_OV5647.h>
#else
#error "CAMERA_TYPE must be CAMERA_TYPE_IMX219 or CAMERA_TYPE_OV5647"
#endif

// ── Camera wrapper: thin veneer over both library APIs ────────────────
// Both libraries have identical function signatures, only the prefix differs.

bool CameraBegin() {
#if CAMERA_TYPE == CAMERA_TYPE_IMX219
  return esp32_p4_imx219_begin();
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
  bool ok = esp32_p4_ov5647_begin();
  // Optional: flip 180° if the sensor is mounted upside-down.
  // esp32_p4_ov5647_set_flip_180(true);
  return ok;
#endif
}

bool CameraUpdate() {
#if CAMERA_TYPE == CAMERA_TYPE_IMX219
  return esp32_p4_imx219_update();
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
  return esp32_p4_ov5647_update();
#endif
}

const uint8_t* CameraGetRgb() {
#if CAMERA_TYPE == CAMERA_TYPE_IMX219
  return esp32_p4_imx219_rgb();
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
  return esp32_p4_ov5647_rgb();
#endif
}

size_t CameraGetRgbSize() {
#if CAMERA_TYPE == CAMERA_TYPE_IMX219
  return esp32_p4_imx219_rgb_size();
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
  return esp32_p4_ov5647_rgb_size();
#endif
}

const char* CameraGetName() {
#if CAMERA_TYPE == CAMERA_TYPE_IMX219
  return "IMX219";
#elif CAMERA_TYPE == CAMERA_TYPE_OV5647
  return "OV5647";
#endif
}

// ── TFLite image provider ─────────────────────────────────────────────
// Returns int8 RGB data for MTCNN.
// MTCNN expects int8 input: each pixel byte ^ 0x80 (i.e. uint8[0,255] → int8[-128,127]).

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter,
                      int image_width, int image_height, int channels,
                      int8_t* image_data) {
  (void)error_reporter;

  if (image_width != OUT_WIDTH || image_height != OUT_HEIGHT || channels != OUT_CHANNELS) {
    return kTfLiteError;
  }

  // Poll for a fresh frame (100 tries × 2 ms = 200 ms timeout).
  bool updated = false;
  for (int tries = 0; tries < 100; tries++) {
    if (CameraUpdate()) {
      updated = true;
      break;
    }
    delayMicroseconds(2000);
  }
  if (!updated) {
    return kTfLiteError;
  }

  const uint8_t* rgb = CameraGetRgb();
  if (!rgb) {
    return kTfLiteError;
  }

  // uint8 → int8 conversion for TFLite (same as MTCNN reference: ^ 0x80).
  for (int i = 0; i < OUT_RGB_SIZE; i++) {
    image_data[i] = (int8_t)(rgb[i] ^ 0x80);
  }

  return kTfLiteOk;
}
