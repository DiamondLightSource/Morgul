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

__global__ void jungfrau_pedestal_accumulate(const uint16_t *halfmodule_data,
                                             uint32_t *pedestals_n,
                                             uint32_t *pedestals_x,
                                             uint64_t *pedestals_x_sq,
                                             int expected_gain_mode) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = y * HM_WIDTH + x;
    int gain_mode = halfmodule_data[index] >> 14;
    int value = halfmodule_data[index] & 0x3fff;
    // if (expected_gain_mode && gain_mode = expected_gain_mode.value())

    if (gain_mode == 3) {
        gain_mode = 2;
    }
    // // Let's be clever, and use live data to determine what the expected
    // // gain mode is. We need this, because we don't want pixels that
    // // get stuck in a different gain mode
    // int votes[3];
    // votes[0] = __popc(__ballot_sync(0xFFFFFFFF, gain_mode == 0));
    // votes[1] = __popc(__ballot_sync(0xFFFFFFFF, gain_mode == 1));
    // votes[2] = __popc(__ballot_sync(0xFFFFFFFF, gain_mode == 3));
    // int winner_count = max(max(votes[0], votes[1]), votes[2]);
    // if (votes[gain_mode] == winner_count) {
    if (gain_mode == expected_gain_mode) {
        auto gain_offset = index + HM_HEIGHT * HM_WIDTH * gain_mode;
        pedestals_n[gain_offset] += 1;
        pedestals_x[gain_offset] += value;
        pedestals_x_sq[gain_offset] += value * value;
    }
}

__global__ void jungfrau_pedestal_finalize(const uint32_t *pedestals_n,
                                           const uint32_t *pedestals_x,
                                           float *pedestals,
                                           bool *pedestals_mask) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int index = y * HM_WIDTH + x;
    float sum_x = pedestals_x[index];

    pedestals[index] = sum_x / static_cast<float>(pedestals_n[index]);
    // Set the mask if any gain is zero. This is safe to clobber because
    // it is only ever setting "true".
    if (pedestals_n[index] == 0) {
        pedestals_mask[(y % HM_HEIGHT) * HM_WIDTH + x] = true;
    }
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

void call_jungfrau_pedestal_accumulate(cudaStream_t stream,
                                       const uint16_t *halfmodule_data,
                                       uint32_t *pedestals_n,
                                       uint32_t *pedestals_x,
                                       uint64_t *pedestals_x_sq,
                                       int expected_gain_mode) {
    jungfrau_pedestal_accumulate<<<dim3(HM_WIDTH / 32, HM_HEIGHT / 32),
                                   dim3(32, 32),
                                   0,
                                   stream>>>(
        halfmodule_data, pedestals_n, pedestals_x, pedestals_x_sq, expected_gain_mode);
}
void call_jungfrau_pedestal_finalize(cudaStream_t stream,
                                     const uint32_t *pedestals_n,
                                     const uint32_t *pedestals_x,
                                     float *pedestals,
                                     bool *pedestals_mask) {
    jungfrau_pedestal_finalize<<<dim3(HM_WIDTH / 32, 3 * HM_HEIGHT / 32),
                                 dim3(32, 32),
                                 0,
                                 stream>>>(
        pedestals_n, pedestals_x, pedestals, pedestals_mask);
}