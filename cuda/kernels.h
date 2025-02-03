#pragma once

#include <cuda_runtime.h>

#include "calibration.hpp"

void call_jungfrau_image_corrections(cudaStream_t stream,
                                     GainData::GainModePointers gains,
                                     PedestalData::GainModePointers pedestals,
                                     const uint16_t *halfmodule_data,
                                     uint16_t *out_corrected_data,
                                     float energy_kev);
void call_jungfrau_pedestal_accumulate(cudaStream_t stream,
                                       const uint16_t *halfmodule_data,
                                       uint32_t *pedestals_n,
                                       uint32_t *pedestals_x,
                                       uint64_t *pedestals_x_sq,
                                       int expected_gain_mode);

void call_jungfrau_pedestal_finalize(cudaStream_t stream,
                                     const uint32_t *pedestals_n,
                                     const uint32_t *pedestals_x,
                                     float *pedestals,
                                     bool *pedestals_mask);