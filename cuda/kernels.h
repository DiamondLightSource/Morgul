#pragma once

#include <cuda_runtime.h>

#include "calibration.hpp"

void call_jungfrau_image_corrections(cudaStream_t stream,
                                     GainData::GainModePointers gains,
                                     PedestalData::GainModePointers pedestals,
                                     uint16_t *halfmodule_data,
                                     float energy_kev);