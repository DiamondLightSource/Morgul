#pragma once

#include <cuda_runtime.h>

#include "calibration.hpp"
#include "cuda_common.hpp"

void call_jungfrau_image_corrections(cudaStream_t stream,
                                     GainData::GainModePointers gains,
                                     PedestalData::GainModePointers pedestals,
                                     const uint16_t *halfmodule_data,
                                     shared_device_ptr<uint16_t[]> out_corrected_data,
                                     float energy_kev);

void call_jungfrau_pedestal_accumulate(cudaStream_t stream,
                                       const uint16_t *halfmodule_data,
                                       shared_device_ptr<uint32_t[]> pedestals_n,
                                       shared_device_ptr<uint32_t[]> pedestals_x,
                                       shared_device_ptr<uint64_t[]> pedestals_x_sq,
                                       int expected_gain_mode);

void call_jungfrau_pedestal_finalize(cudaStream_t stream,
                                     const shared_device_ptr<uint32_t[]> pedestals_n,
                                     const shared_device_ptr<uint32_t[]> pedestals_x,
                                     float *pedestals,
                                     bool *pedestals_mask);

void launch_bitshuffle(cudaStream_t stream,
                       raw_device_ptr<std::byte[]> d_in,
                       shared_device_ptr<std::byte[]> d_out);