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
    float value = halfmodule_data[index] & 0x3fff;
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

#define SIZE 4096
#define ELEM 2

__global__ void bitshuffle(const uint8_t *in, uint8_t *out) {
    const int b = blockIdx.x;

    in += b * SIZE * ELEM;
    out += b * SIZE * ELEM;

    __shared__ uint8_t scr0[SIZE * ELEM];
    __shared__ uint8_t scr1[SIZE * ELEM];

    const int tid = threadIdx.x;

    const int i0 = (tid & 0xffe) << 2;
    const int j0 = tid & 0x1;

    for (int k = 0; k < 8; k++) {
        scr0[j0 * SIZE + i0 + k] = in[(i0 + k) * ELEM + j0];
    }

    __syncthreads();

    uint64_t x, t;

    x = ((const uint64_t *)scr0)[tid];

    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;
    x = x ^ t ^ (t << 28);

    for (int k = 0; k < 8; k++) {
        scr1[k * ELEM * SIZE / 8 + tid] = x;
        x = x >> 8;
    }

    __syncthreads();

    const int i2 = (tid & 0xe) >> 1;
    const int j2 = tid & 0x1;
    const int k2 = tid >> 4;
    ((uint64_t *)out)[(j2 * 8 + i2) * SIZE / 64 + k2] =
        ((const uint64_t *)scr1)[(i2 * ELEM + j2) * SIZE / 64 + k2];
}

void launch_bitshuffle(cudaStream_t stream,
                       void *in,
                       void *out,
                       void *d_in,
                       void *d_out) {
    const dim3 block(1024);
    const dim3 grid(64);
    cudaMemcpy(d_in, in, 256 * 1024 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    bitshuffle<<<grid, block, 0, stream>>>((const uint8_t *)d_in, (uint8_t *)d_out);
    cudaMemcpy(out, d_out, 256 * 1024 * sizeof(uint16_t), cudaMemcpyDeviceToHost);
}