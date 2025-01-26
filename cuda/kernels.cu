#include "calibration.hpp"
#include "constants.hpp"

__global__ void jungfrau_image_corrections(GainData::GainModePointers gains,
                                           PedestalData::GainModePointers pedestals,
                                           uint16_t *halfmodule_data,
                                           double energy) {}

void do_jungfrau_image_corrections(cudaStream_t stream,
                                   GainData::GainModePointers gains,
                                   PedestalData::GainModePointers pedestals,
                                   uint16_t *halfmodule_data,
                                   double energy) {
    jungfrau_image_corrections<<<dim3(HM_WIDTH / 32, HM_HEIGHT / 32),
                                 dim3(32, 32),
                                 0,
                                 stream>>>(gains, pedestals, halfmodule_data, energy);
}