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
    if (gain_mode == 3) {
        gain_mode = 2;
    }
    halfmodule_data[index] =
        (value - pedestals[gain_mode][index]) * gains[gain_mode][index];
    // TODO:
    // - Where does rint come in
    // - Where does energy come in?
    //      - Possibly has pre-multiplied in PR https://github.com/graeme-winter/jungfrau/pull/39/ ?
    // - Output writing format
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