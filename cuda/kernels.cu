#include "calibration.hpp"
#include "constants.hpp"

__global__ void jungfrau_image_corrections(GainData::GainModePointers gains,
                                           PedestalData::GainModePointers pedestals,
                                           const uint16_t *halfmodule_data,
                                           uint16_t *out_corrected_data,
                                           float energy_kev) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = y * HM_WIDTH + x;
    int gain_mode = halfmodule_data[index] >> 14;
    int value = halfmodule_data[index] & 0x3fff;
    if (gain_mode == 3) {
        gain_mode = 2;
    }

    out_corrected_data[index] = rintf((value - pedestals[gain_mode][index])
                                      / (gains[gain_mode][index] * energy_kev));
}

void call_jungfrau_image_corrections(cudaStream_t stream,
                                     GainData::GainModePointers gains,
                                     PedestalData::GainModePointers pedestals,
                                     const uint16_t *halfmodule_data,
                                     uint16_t *out_corrected_data,
                                     float energy_kev) {
    jungfrau_image_corrections<<<dim3(HM_WIDTH / 32, HM_HEIGHT / 32),
                                 dim3(32, 32),
                                 0,
                                 stream>>>(
        gains, pedestals, halfmodule_data, out_corrected_data, energy_kev);
}