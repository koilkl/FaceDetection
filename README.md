# MTCNN Face Detection on ESP32-P4

Real-time face detection on ESP32-P4 using MTCNN (Multi-task Cascaded Convolutional Networks) with TensorFlow Lite Micro.

Outputs face bounding box coordinates and per-stage timing to the serial console.

## Hardware

- ESP32-P4 (with PSRAM)
- Camera: **IMX219** or **OV5647** (MIPI CSI)
- Arduino IDE with `esp32_mannual:esp32p4` core

## Camera Selection

Edit `image_provider.h` line 15:

```cpp
#define CAMERA_TYPE CAMERA_TYPE_OV5647   // OV5647
//#define CAMERA_TYPE CAMERA_TYPE_IMX219  // IMX219
```

## Build & Run

1. Open `FaceDetection.ino` in Arduino IDE
2. Select board: **ESP32P4 Dev Module**
3. Set camera type in `image_provider.h`
4. Upload

## Serial Output (921600 baud)

```
=== FaceDetection (MTCNN on ESP32-P4) ===
Camera: OV5647 ... OK
TFLite: OK
ESP-NN: ENABLED
Running...
[frame 1] MTCNN: 1 faces | P=65ms R=232ms O=789ms total=1086ms | raw_cands=47
  face[0]: (12, 8) - (54, 52)
```

## How It Works

MTCNN is a 3-stage cascade:

1. **P-Net** (3 image scales) — fast full-image scan, proposes candidate face windows
2. **R-Net** (24×24 input) — filters false positives, refines bounding boxes
3. **O-Net** (48×48 input) — final precise face localization

All 5 models are int8 quantized and embedded as C arrays. Inference runs sequentially on a shared 200KB PSRAM tensor arena with ESP-NN RISC-V SIMD acceleration.

## Credits

- MTCNN implementation ported from [mauriciobarroso/mtcnn_esp32s3](https://github.com/mauriciobarroso/mtcnn_esp32s3) (MIT License)
- Original MTCNN paper: Zhang et al., "Joint Face Detection and Alignment using Multitask Cascaded Convolutional Networks"
