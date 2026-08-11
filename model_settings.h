#ifndef FACEDETECT_MODEL_SETTINGS_H_
#define FACEDETECT_MODEL_SETTINGS_H_

#include "image_provider.h"

constexpr int kNumCols = OUT_WIDTH;      // 96
constexpr int kNumRows = OUT_HEIGHT;     // 96
constexpr int kNumChannels = OUT_CHANNELS;  // 3 (RGB)

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;  // 27648

#endif  // FACEDETECT_MODEL_SETTINGS_H_
