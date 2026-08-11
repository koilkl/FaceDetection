#ifndef FACEDETECT_MODELS_SETTINGS_H_
#define FACEDETECT_MODELS_SETTINGS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>

/* Input image shape (must match camera output) */
#define IMG_W   96
#define IMG_H   96
#define IMG_CH  3

/* P-Net models scales */
#define PNET_1_SCALE  0.3333333f
#define PNET_2_SCALE  0.25f
#define PNET_3_SCALE  0.125f

/* Extern declarations for model data arrays */
extern const unsigned char pnet_1_model_data[];
extern const unsigned char pnet_2_model_data[];
extern const unsigned char pnet_3_model_data[];
extern const unsigned char rnet_model_data[];
extern const unsigned char onet_model_data[];

#ifdef __cplusplus
}
#endif

#endif /* FACEDETECT_MODELS_SETTINGS_H_ */
