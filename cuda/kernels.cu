#include "calibration.hpp"
#include "constants.hpp"

__global__ void jungfrau_image_corrections(GainData::GainModePointers gains,
                                           PedestalData::GainModePointers pedestals,
                                           uint16_t *halfmodule_data,
                                           double energy) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = y * HM_WIDTH + x;
    int gain_mode = halfmodule_data[index] >> 14;
    int value = halfmodule_data[index] & 0x3fff;

    halfmodule_data[index] =
        (value - pedestals[gain_mode][index]) * gains[gain_mode][index];
    // for (int j = 0; j < 256 * 1024; j++) {
    //     unsigned short corrected = 0x8000;
    //     short mode = image[j] >> 14;
    //     float value = (float)(image[j] & 0x3fff);
    //     if (mode == 0) {
    //         if (pedestal0[j]) corrected = rint((value - pedestal0[j]) * gain0[j]);
    //     } else if (mode == 1) {
    //         if (pedestal1[j]) corrected = rint((value - pedestal1[j]) * gain1[j]);
    //     } else if (mode == 3) {
    //         if (pedestal2[j]) corrected = rint((value - pedestal2[j]) * gain2[j]);
    //     }
    //     image[j] = corrected;
    // }
}

void call_jungfrau_image_corrections(cudaStream_t stream,
                                     GainData::GainModePointers gains,
                                     PedestalData::GainModePointers pedestals,
                                     uint16_t *halfmodule_data,
                                     double energy) {
    jungfrau_image_corrections<<<dim3(HM_WIDTH / 32, HM_HEIGHT / 32),
                                 dim3(32, 32),
                                 0,
                                 stream>>>(gains, pedestals, halfmodule_data, energy);
}