
#include <fmt/ranges.h>
#include <pthread.h>
#include <sls/Receiver.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <stop_token>
#include <thread>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "bitshuffle.h"
#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"
#include "hdf5_tools.hpp"
#include "kernels.h"
#include "lz4.h"

using namespace fmt;
using json = nlohmann::json;
using namespace std::chrono_literals;

/// Location of some pedestal data to load. Make this automatic/specifiable later
const auto PEDESTAL_DATA = std::filesystem::path{"/dev/shm/pedestals.h5"};

std::stop_source global_stop;

/// Count how many threads are waiting, so we know if everything is idle
std::atomic_int threads_waiting{0};
std::atomic_bool in_acquisition{false};
/// Used to identify the first validation each acquisition, to avoid spamming
std::atomic_bool is_first_validation_this_acquisition{false};
std::atomic_int acquisition_number{0};
std::atomic<float> acq_progress{0};

/// Get an environment variable if present, with optional default
auto getenv_or(std::string name, std::optional<std::string> _default = std::nullopt)
    -> std::optional<std::string> {
    auto data = std::getenv(name.c_str());
    if (data == nullptr) {
        return _default;
    }
    return {data};
}

/// Print a spinner animation, then return the number of characters printed
int spinner(const std::string_view &message) {
    static int index = 0;
    std::vector<std::string> ball = {
        "( ●    )",
        "(  ●   )",
        "(   ●  )",
        "(    ● )",
        "(     ●)",
        "(    ● )",
        "(   ●  )",
        "(  ●   )",
        "( ●    )",
        "(●     )",
    };
    index = (index + 1) % ball.size();
    std::string msg = fmt::format("  {} {}\r", message, ball[index]);
    std::cout << msg << std::flush;
    return msg.size();
}

class Timer {
  public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    auto get_elapsed_seconds() -> double {
        std::chrono::duration<double, std::milli> dt =
            std::chrono::high_resolution_clock::now() - start;
        return dt.count() / 1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point start;
};

#pragma region Header Parsing

auto read_boolish(std::string value) -> bool {
    if (value.size() == 0) {
        return false;
    }
    if (value != "true" || value != "false") {
        throw std::runtime_error(
            fmt::format("Got non-boolish json value for pedestal: '{}'", value));
    }
    return value == "true";
}

struct DLSHeaderAdditions {
    bool pedestal = false;
    /// @brief Photon Energy (KeV)
    std::optional<double> energy;
    bool raw = false;
    std::optional<size_t> pedestal_frames;
    std::optional<size_t> pedestal_loops;

    static auto from_map(const std::map<std::string, std::string> raw)
        -> DLSHeaderAdditions {
        DLSHeaderAdditions out;
        if (raw.find("pedestal") != raw.end()) {
            out.pedestal = read_boolish(raw.at("pedestal"));
        }
        if (raw.find("raw") != raw.end()) {
            out.raw = read_boolish(raw.at("raw"));
        }
        if (raw.find("wavelength") != raw.end()) {
            auto value = raw.at("wavelength");
            double wavelength_angstrom = std::strtod(value.c_str(), nullptr) * 10;
            double energy_kev = 12.39841984055037 / wavelength_angstrom;
            out.energy = energy_kev;
        }
        if (raw.find("pedestal_frames") != raw.end()) {
            out.pedestal_frames = {
                static_cast<size_t>(std::stoi(raw.at("pedestal_frames")))};
        }
        if (raw.find("pedestal_loops") != raw.end()) {
            out.pedestal_loops = {
                static_cast<size_t>(std::stoi(raw.at("pedestal_loops")))};
        }
        return out;
    }
};

class SLSHeader {
  public:
    uint32_t bitmode;
    std::array<uint32_t, 2> detshape;
    std::array<uint32_t, 2> shape;
    size_t acqIndex;
    /// Index of this frame in the current acquisition e.g. 0....N-1
    size_t frameIndex;
    double progress;
    /// The number of frames since the detector count was reset - NOT frame index
    size_t frameNumber;
    uint32_t expLength;
    uint32_t packetNumber;
    uint32_t row;
    uint32_t column;
    uint32_t detType;
    std::map<std::string, std::string> addJsonHeader;
    /// DLS-specific additional headers that may be present in addJsonHeader
    DLSHeaderAdditions dls;
    uint16_t udp_port;
};

#pragma endregion

#pragma region Pedestal Library

class PedestalsLibrary {
    auto load_pedestal_cache(std::filesystem::path path) -> bool {
        auto pd = PedestalData(path, _detector);
        auto [dx, dy] = DETECTOR_SIZE.at(_detector);
        uint64_t exposure_ns = llrint(pd.exposure_time() * 1e9);
        for (size_t m = 0; m < dx * dy * 2; ++m) {
            auto &ped_0 = pd.get_pedestal(m, 0);
            auto &ped_1 = pd.get_pedestal(m, 1);
            auto &ped_2 = pd.get_pedestal(m, 2);
            register_pedestals(
                exposure_ns, m, ped_0.data(), ped_1.data(), ped_2.data());
        }
        // print("Loaded prexisting {:.1} ms pedestals from {}\n",
        //       styled(pd.exposure_time() * 1000, style::number),
        //       styled(path, style::path));
        return true;
    }

  public:
    // typedef float pedestal_t;
    using pedestal_t = PedestalData::pedestal_t;

    // using GainModePointers = std::array<pedestal_t *, GAIN_MODES.size()>;

    PedestalsLibrary(Detector detector) : _detector(detector) {
        assert(detector == JF1M);
        // Try to load existing data from /dev/shm
        auto store = std::filesystem::path{PEDESTAL_DATA};
        if (std::filesystem::exists(store)) {
            load_pedestal_cache(store);
        }
    }

    bool has_pedestals(uint64_t exposure_ns, uint8_t halfmodule_index) const {
        return _gains.contains(exposure_ns)
               && _gains.at(exposure_ns).contains(halfmodule_index);
    }
    auto get_gpu_ptrs(uint64_t exposure_ns, uint8_t halfmodule_index) const
        -> PedestalData::GainModePointers {
        auto &lookup = _gains.at(exposure_ns).at(halfmodule_index);
        return {lookup.at(0), lookup.at(1), lookup.at(2)};
    }

    void save_pedestals() {
        if (_gains.size() == 0) return;
        assert(_gains.size() == 1);
        // Because HDF5, only allow us to do this one-at-a-time
        static std::mutex hdf5_write_lock;
        auto lock = std::scoped_lock(hdf5_write_lock);

        auto file = H5Cleanup<H5Fclose>(H5Fcreate(
            "/dev/shm/pedestals.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        std::vector<PedestalsLibrary::pedestal_t> pedestal_host(HM_PIXELS);
        for (auto [exp, hmod_map] : _gains) {
            write_scalar_hdf5_value<float>(
                file, "/exptime", static_cast<float>(exp) * 1e9f);
            write_scalar_hdf5_value<std::string>(file, "/module_mode", {"half"});
            for (auto [hmi, hm_gains] : hmod_map) {
                auto group_name = fmt::format("hmi_{:02}", hmi);
                auto hm_group = H5Cleanup<H5Gclose>(H5Gcreate(
                    file, group_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                hsize_t dims[2]{HM_HEIGHT, HM_WIDTH};
                auto space = H5Cleanup<H5Sclose>(H5Screate_simple(2, dims, nullptr));
                for (auto [gain, pedestal_data] : hm_gains) {
                    auto ds_name = fmt::format("pedestal_{}", gain);
                    cudaMemcpyAsync(pedestal_host.data(), pedestal_data, HM_PIXELS, 0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    auto dset = H5Cleanup<H5Dclose>(H5Dcreate(hm_group,
                                                              ds_name.c_str(),
                                                              H5T_NATIVE_FLOAT,
                                                              space,
                                                              H5P_DEFAULT,
                                                              H5P_DEFAULT,
                                                              H5P_DEFAULT));
                    H5Dwrite(dset,
                             H5T_NATIVE_FLOAT,
                             H5S_ALL,
                             H5S_ALL,
                             H5P_DEFAULT,
                             pedestal_host.data());
                }
            }
        }
    }
    /// @brief Register a new set of pedestal data
    ///
    /// Safe to call from multiple threads.
    void register_pedestals(uint64_t exposure_ns,
                            uint8_t halfmodule_index,
                            std::span<pedestal_t> pedestal_0,
                            std::span<pedestal_t> pedestal_1,
                            std::span<pedestal_t> pedestal_2) {
        auto dev_0 = make_cuda_malloc<float>(HM_PIXELS);
        auto dev_1 = make_cuda_malloc<float>(HM_PIXELS);
        auto dev_2 = make_cuda_malloc<float>(HM_PIXELS);

        {
            std::scoped_lock lock(_write_guard);
            // Safety: For now, only allow one pedestal to be registered
            if (!_gains.contains(exposure_ns)) {
                _gains.clear();
            }
            _gains[exposure_ns][halfmodule_index][0] = dev_0;
            _gains[exposure_ns][halfmodule_index][1] = dev_1;
            _gains[exposure_ns][halfmodule_index][2] = dev_2;
        }
        assert(pedestal_0.size() == HM_PIXELS);
        assert(pedestal_1.size() == HM_PIXELS);
        assert(pedestal_2.size() == HM_PIXELS);
        cudaMemcpyAsync(dev_0, pedestal_0.data(), pedestal_0.size(), 0);
        cudaMemcpyAsync(dev_1, pedestal_1.data(), pedestal_1.size(), 0);
        cudaMemcpyAsync(dev_2, pedestal_2.data(), pedestal_2.size(), 0);

        // CUDA_CHECK(cudaMemcpy(dev_0.get(),
        //                       pedestal_0.data(),
        //                       pedestal_0.size_bytes(),
        //                       cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(dev_1.get(),
        //                       pedestal_1.data(),
        //                       pedestal_1.size_bytes(),
        //                       cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(dev_2.get(),
        //                       pedestal_2.data(),
        //                       pedestal_2.size_bytes(),
        //                       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        print(
            "Registering pedestals for {} ns {} hmi\n", exposure_ns, halfmodule_index);
    }

  private:
    std::map<uint64_t,
             std::map<uint8_t, std::map<uint8_t, shared_device_ptr<pedestal_t[]>>>>
        _gains;
    const Detector _detector;
    std::mutex _write_guard;
};

#pragma region Handler Class
class DataStreamHandler {
  public:
    // Once we receive an HMI, we must always receive the same one
    std::optional<uint32_t> known_hmi;

    // Keep track of how many images we have seen/the highest index.
    // since the last end-packet.
    size_t num_images_seen = 0;
    size_t highest_image_seen = 0;
    // Keep track of the last frame number seen, so we know if a frame was skipped
    uint64_t hm_frameNumber = 0;
    uint64_t exposure_ns = 0;
    bool is_pedestal_mode = false;
    /// Should we attempt to send packets onward to a writer.
    ///
    /// Turned off for the acquisition on send error to the writer.
    bool send_onwards = true;

    DataStreamHandler(const Arguments &args,
                      uint16_t port,
                      const CudaStream &stream,
                      const GainData &gains,
                      PedestalsLibrary &pedestals,
                      zmq::socket_t &send_socket)
        : _args(args),
          _port(port),
          stream(stream),
          gains(gains),
          pedestals(pedestals),
          send(send_socket) {
        is_first_validation_this_acquisition.store(true);
        // Work out the maximum size the compressed data can be, add 12 for the HDF5 header
        size_t compress_size = LZ4_compressBound(sizeof(pixel_t) * HM_PIXELS) + 12;
        compression_buffer = std::vector<std::byte>(compress_size);
        // partial_compression_buffer = std::vector<std::byte>(LZ4_compressBound(sizeof(pixel_t) * HM_PIXELS) + 12;)
        assert(compression_buffer.size() == compress_size);

        pedestal_n = make_cuda_malloc<uint32_t>(GAIN_MODES.size() * HM_PIXELS);
        pedestal_x = make_cuda_malloc<uint32_t>(GAIN_MODES.size() * HM_PIXELS);
        pedestal_x_sq = make_cuda_malloc<uint64_t>(GAIN_MODES.size() * HM_PIXELS);
        reset_pedestal_buffers();
        dev_bitshuffle_buffer_out = make_cuda_malloc<std::byte>(HM_PIXELS * 2);
    }
    ~DataStreamHandler() {}

    auto validate_header(const SLSHeader &header) -> bool;
    auto process_frame(const SLSHeader &header, std::span<uint16_t> &frame) -> void;
    auto end_acquisition() -> void;

    double stats_lz4_time = 0;
    double stats_process_frame_time = 0;
    double stats_push = 0;
    double stats_correct = 0;
    double stats_bs = 0;

  private:
    void reset_pedestal_buffers() {
        cudaMemsetAsync(pedestal_n, 0, GAIN_MODES.size() * HM_PIXELS, 0);
        cudaMemsetAsync(pedestal_x, 0, GAIN_MODES.size() * HM_PIXELS, 0);
        cudaMemsetAsync(pedestal_x_sq, 0, GAIN_MODES.size() * HM_PIXELS, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const Arguments &_args;
    uint16_t _port;
    const CudaStream &stream;
    const GainData &gains;
    PedestalsLibrary &pedestals;
    zmq::socket_t &send;
    std::unique_ptr<std::byte[]> bitshuffled_buffer =
        std::make_unique<std::byte[]>(HM_PIXELS * sizeof(pixel_t));
    /// Where to store the output data, that will be fed into compression
    shared_device_ptr<uint16_t[]> dev_output_buffer =
        make_cuda_malloc<uint16_t>(HM_PIXELS);
    std::vector<std::byte> compression_buffer;
    std::vector<std::byte> partial_compression_buffer;
    // Accumulation buffers for calculating pedestals on-the-fly
    // Note: Because the value is max. 14-bit, we have worst-case 18-bits
    // of count before 32-bit saturation, so n is 32-bit to cover this.
    shared_device_ptr<uint32_t[]> pedestal_n;
    shared_device_ptr<uint32_t[]> pedestal_x;
    shared_device_ptr<uint64_t[]> pedestal_x_sq;
    shared_device_ptr<std::byte[]> dev_bitshuffle_buffer_out;
};

#pragma region Validate Header

auto DataStreamHandler::validate_header(const SLSHeader &header) -> bool {
    // Once per acquisition, the first thread through gets this flag
    bool _expected = true;
    bool first_acquisition =
        is_first_validation_this_acquisition.compare_exchange_strong(_expected, false);

    // Validate this matches our expectations
    if (header.shape != std::array{HM_WIDTH, HM_HEIGHT}) {
        if (first_acquisition) {
            print(style::error,
                  "{}: Error: Got wrong sized image ({}), expected (1024,256)",
                  _port,
                  header.shape);
        }
        return false;
    }
    uint32_t det_w = std::get<0>(DETECTOR_SIZE.at(_args.detector));
    uint32_t det_h = std::get<1>(DETECTOR_SIZE.at(_args.detector)) * 2;
    if (header.detshape != std::array{det_w, det_h}) {
        if (first_acquisition) {
            print(style::error,
                  "{}: Error: Got wrong sized detector {}; expected {},{}",
                  _port,
                  header.detshape,
                  det_w,
                  det_h);
        }
        return false;
    }
    // Handle knowing which module we handle
    auto hmi = header.column * det_h + header.row;
    if (hmi > 100) {
        print(style::error,
              "{}: Fatal error: Got unrealistic HMI from\n {} * {} + {} = {}\n",
              _port,
              header.column,
              det_h,
              header.row,
              hmi);
        std::exit(1);
    }
    if (!known_hmi) {
        known_hmi = hmi;
    } else {
        if (known_hmi != hmi) {
            print(style::error,
                  "{}: Fatal Error: Got fed mix of module index; hmi={} instead of "
                  "initial {} are your routing "
                  "crossed?",
                  _port,
                  hmi,
                  known_hmi);
            std::exit(1);
        }
    }
    if (!header.dls.energy) {
        if (first_acquisition) {
            print(style::warning,
                  "Warning: Did not get energy in addJsonHeader packet\n");
        }
    }

    // Paranoia: Look for pedestal flag changing partway through stream.
    // This is unlikely to be from the detector, but bad handling of
    // acquisition separation in the logic of this program.
    if ((header.dls.pedestal && !is_pedestal_mode && num_images_seen > 0)
        || is_pedestal_mode && !header.dls.pedestal && num_images_seen > 0) {
        print(style::error,
              "hm {}: Error: Pedestal flag toggled midway through stream ({} "
              "images seen)! "
              "Ignoring data.\n",
              known_hmi.value(),
              num_images_seen);
        return false;
    }

    // Handle Setting data on first image in an acquisition
    if (num_images_seen == 0) {
        exposure_ns = header.expLength * 100;

        is_pedestal_mode = header.dls.pedestal;
        if (is_pedestal_mode) {
            if (!header.dls.pedestal_frames) {
                print(style::error,
                      "Error: Pedestal mode on but no pedestal_frames set\n");
                return false;
            }
            if (!header.dls.pedestal_loops) {
                print(style::error,
                      "Error: Pedestal mode on but no pedestal_loops set\n");
                return false;
            }
            if (first_acquisition) {
                print("Starting pedestal measurement run\n");
            }
        } else {
            if (!pedestals.has_pedestals(exposure_ns, known_hmi.value())) {
                print(_args.require_pedestals ? style::error : style::warning,
                      "Warning: Do not have pedestals for {:.2f} ms HMI={}, cannot "
                      "correct.\n",
                      exposure_ns / 1000000.0,
                      known_hmi.value());
                if (_args.require_pedestals) {
                    return false;
                }
                // We aren't *requiring* pedestals, but we need to have some
                std::vector<PedestalsLibrary::pedestal_t> fake_pedestals(
                    HM_PIXELS * GAIN_MODES.size());

                pedestals.register_pedestals(
                    exposure_ns,
                    known_hmi.value(),
                    {fake_pedestals.data(), HM_PIXELS},
                    {fake_pedestals.data() + HM_PIXELS, HM_PIXELS},
                    {fake_pedestals.data() + HM_PIXELS * 2, HM_PIXELS});
            }
        }
    }

    ++num_images_seen;
    highest_image_seen = std::max(highest_image_seen, header.frameIndex + 1);
    if (hm_frameNumber != 0 && header.frameNumber > hm_frameNumber + 1) {
        auto num_skipped = header.frameNumber - hm_frameNumber - 1;
        print(style::warning,
              "hm {}: Warning: Skipped {} frames\n",
              known_hmi.value(),
              num_skipped);
    }
    return true;
}

#pragma region Process Frame
auto DataStreamHandler::process_frame(const SLSHeader &header,
                                      std::span<uint16_t> &frame) -> void {
    auto time_frame = Timer();
    auto energy = header.dls.energy.value_or(12.4);

    if (header.dls.raw) {
        // We want raw, uncorrected data. Just copy it over.
        cudaMemcpyAsync(dev_output_buffer, frame.data(), HM_PIXELS, stream);
    } else if (is_pedestal_mode) {
        // We also want to keep raw pedestal data uncorrected
        cudaMemcpyAsync(dev_output_buffer, frame.data(), HM_PIXELS, stream);
        // Work out what we expect the gain mode to be for this frame.
        // We don't want to count pixels in frames that aren't what they
        // are supposed to be forced to.
        const auto ploops = header.dls.pedestal_loops.value();
        const auto pframes = header.dls.pedestal_frames.value();

        int gain_mode = header.frameIndex >= (ploops * pframes) ? 2 : 1;

        if (header.frameIndex % pframes != (pframes - 1)) {
            // Only the Nframes-1-indexed images have the gain mode forced.
            gain_mode = 0;
        }

        call_jungfrau_pedestal_accumulate(
            stream, frame.data(), pedestal_n, pedestal_x, pedestal_x_sq, gain_mode);
    } else {
        auto timer_corr = Timer();
        call_jungfrau_image_corrections(
            stream,
            gains.get_gpu_ptrs(known_hmi.value()),
            pedestals.get_gpu_ptrs(exposure_ns, known_hmi.value()),
            frame.data(),
            dev_output_buffer,
            energy);
        stats_correct += timer_corr.get_elapsed_seconds();
    }
    // Construct the HDF5 header so that we can do direct chunk write
    // on the other end of the pipe
    // first 12 bytes are uint64_t BE array size and uint32_t BE block size
    // these are the precomputed values
    uint64_t &header_uncompress_size_ref =
        *reinterpret_cast<uint64_t *>(compression_buffer.data());
    uint32_t &header_block_size_ref =
        *reinterpret_cast<uint32_t *>(compression_buffer.data() + 8);
    header_uncompress_size_ref = __builtin_bswap64(2 * 256 * 1024);
    header_block_size_ref = __builtin_bswap32(8192);

    auto timer_bs = Timer();
    launch_bitshuffle(stream, dev_output_buffer, dev_bitshuffle_buffer_out);
    // Copy this back from the device for LZ4
    cudaMemcpyAsync(
        bitshuffled_buffer.get(), dev_bitshuffle_buffer_out, HM_PIXELS * 2, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    stats_bs += timer_bs.get_elapsed_seconds();

    auto time_lz4 = Timer();
    size_t current_index = 12;
    for (size_t block = 0; block < HM_PIXELS * 2 / 8192; ++block) {
        auto size = LZ4_compress_default(
            reinterpret_cast<char *>(bitshuffled_buffer.get() + 2 * block * 4096),
            reinterpret_cast<char *>(compression_buffer.data()) + current_index + 4,
            8192,
            compression_buffer.size() - current_index - 4);
        uint32_t &header_block_size_ref =
            *reinterpret_cast<uint32_t *>(compression_buffer.data() + current_index);
        header_block_size_ref = __builtin_bswap32(size);
        current_index += size + 4;
    }
    stats_lz4_time += time_lz4.get_elapsed_seconds();

    // static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
    //                         )
    //                         .count())
    // / 1000.0;

    zmq::multipart_t send_msgs;
    json send_header;
    send_header["frameIndex"] = header.frameIndex;
    send_header["row"] = header.row;
    send_header["column"] = header.column;
    send_header["shape"] = header.shape;
    send_header["bitmode"] = header.bitmode;
    send_header["expLength"] = header.expLength;
    send_header["acquisition"] = acquisition_number.load();
    send_msgs.push_back(zmq::message_t(send_header.dump()));
    send_msgs.push_back(zmq::message_t(compression_buffer.data(), current_index));
    auto time_push = Timer();
    if (send_onwards && zmq::send_multipart(send, send_msgs) == std::nullopt) {
        print(style::warning,
              "{}:{}: Warning: Failed to send onward message. Disabling send until end "
              "of "
              "acquisition.\n",
              _port,
              header.udp_port);
        // Don't send any more this acquisition.
        send_onwards = false;
    }
    stats_push += time_push.get_elapsed_seconds();
    stats_process_frame_time += time_frame.get_elapsed_seconds();
}

#pragma region End Acquisition

auto DataStreamHandler::end_acquisition() -> void {
    is_first_validation_this_acquisition.store(false);
    if (num_images_seen != highest_image_seen) {
        print(style::warning,
              "hm{:02}: Incomplete image set, recieved {}/{} expected images\n",
              known_hmi.value(),
              num_images_seen,
              highest_image_seen);
    }

    if (is_pedestal_mode) {
        std::vector<std::byte> pedestal_mask(HM_PIXELS);
        std::vector<PedestalsLibrary::pedestal_t> new_pedestals(HM_PIXELS
                                                                * GAIN_MODES.size());

        call_jungfrau_pedestal_finalize(stream,
                                        pedestal_n,
                                        pedestal_x,
                                        new_pedestals.data(),
                                        reinterpret_cast<bool *>(pedestal_mask.data()));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        pedestals.register_pedestals(exposure_ns,
                                     known_hmi.value(),
                                     {new_pedestals.data(), HM_PIXELS},
                                     {new_pedestals.data() + HM_PIXELS, HM_PIXELS},
                                     {new_pedestals.data() + HM_PIXELS * 2, HM_PIXELS});
    }
    num_images_seen = 0;
    highest_image_seen = 0;
    is_pedestal_mode = false;
    send_onwards = true;
    exposure_ns = 0;
}

#pragma region Main Loop

// auto zmq_listen(std::stop_token stop,
//                 std::barrier<> &sync_barrier,
//                 const Arguments &args,
//                 const GainData &gains,
//                 PedestalsLibrary &pedestals,
//                 uint16_t port) -> void {
//     // For now, each thread gets it's own context. We can experiment
//     // with shared later. The Guide (not that one) suggests one IO
//     // thread per GB/s of data, and we have 2 GB/s per module (e.g.
//     // each IO thread can cope with one half-module).
//     zmq::context_t ctx;

//     zmq::socket_t sub{ctx, zmq::socket_type::sub};
//     sub.connect(fmt::format("tcp://{}:{}", args.zmq_host, port));
//     sub.set(zmq::sockopt::subscribe, "");

//     // Set up the results sending port. Let's reuse the context, and
//     // we can wait and see if this needs more resources.
//     zmq::socket_t send{ctx, zmq::socket_type::push};
//     send.set(zmq::sockopt::sndhwm, 50000);
//     send.set(zmq::sockopt::sndbuf, 128 * 1024 * 1024);
//     send.set(zmq::sockopt::sndtimeo, 10000);
//     send.bind(
//         fmt::format("tcp://0.0.0.0:{}", port - args.zmq_port + args.zmq_send_port));

//     CudaStream stream;
//     DataStreamHandler handler(args, port, stream, gains, pedestals, send);

//     auto output_data = std::make_unique<uint16_t[]>(HM_HEIGHT * HM_WIDTH);

//     while (!stop.stop_requested()) {
//         // Make sure all of our threads are in the same phase
//         sync_barrier.arrive_and_wait();
//         // Wait for the next message. Count waiting so we know when we are idle.
//         // Between acquisitions we wait as long as it takes
//         sub.set(zmq::sockopt::rcvtimeo, -1);
//         size_t num_frames_seen = 0;
//         double time_waiting = 0;
//         double time_frame = 0;
//         auto time_acq = std::optional<Timer>();
//         // Loop over images within an acquisition
//         while (!stop.stop_requested()) {
//             std::vector<zmq::message_t> recv_msgs;
//             auto wait_time = Timer();
//             ++threads_waiting;
//             const auto ret = zmq::recv_multipart(sub, std::back_inserter(recv_msgs));
//             --threads_waiting;
//             in_acquisition = true;
//             if (!time_acq.has_value()) {
//                 time_acq = {Timer()};
//             }
//             time_waiting += wait_time.get_elapsed_seconds();
//             auto frame_time = Timer();
//             // All subsequent waits on this series of images can timeout
//             sub.set(zmq::sockopt::rcvtimeo, static_cast<int>(args.zmq_timeout));

//             if (!ret) {
//                 // If here, then we had a timeout waiting for images.
//                 print(style::error,
//                       "{}: HMI={} Error: Timeout waiting for more images/end "
//                       "notification\n",
//                       port,
//                       handler.known_hmi.value());
//                 break;
//             }

//             if (num_frames_seen == 0) {
//                 print("{}: {}", port, recv_msgs[0].to_string_view());
//             }
//             auto header =
//                 json::parse(recv_msgs[0].to_string_view()).template get<SLSHeader>();
//             if (ret == 1 && header.bitmode == 0) {
//                 print("{}: Received end packet\n", port);
//                 break;
//             } else if (ret > 2) {
//                 print(style::error,
//                       "{}: Error: Got unexpected multipart message length {}\n{}",
//                       port,
//                       ret.value(),
//                       recv_msgs[0].to_string_view());
//                 continue;
//             }
//             if (port == args.zmq_port) {
//                 acq_progress = header.progress;
//             }
//             ++num_frames_seen;
//             // We have a standard image packet
//             // Validate the header, and skip this image if invalid
//             if (!handler.validate_header(header)) {
//                 continue;
//             }
//             std::span<uint16_t> data = {
//                 reinterpret_cast<uint16_t *>(recv_msgs[1].data()),
//                 recv_msgs[1].size() / 2};
//             handler.process_frame(header, data);
//             time_frame += frame_time.get_elapsed_seconds();
//         }
//         bool was_pedestals = handler.is_pedestal_mode;
//         handler.end_acquisition();
//         // Now, wait until all frames have completed
//         sync_barrier.arrive_and_wait();
//         in_acquisition = false;
//         print(
//             "{}: Time waiting for LZ4: {:.2f}S Process: {:.2f}S Push: {:.2f}S Frame: "
//             "{:.2f}S Wait: {:.2f}S Corr: {:.2f}S BS: {:.2f}S\n",
//             port,
//             handler.stats_lz4_time,
//             handler.stats_process_frame_time,
//             handler.stats_push,
//             time_frame,
//             time_waiting,
//             handler.stats_correct,
//             handler.stats_bs);
//         // If we are the first port
//         if (port == args.zmq_port) {
//             print("Acquisition {} complete in {:.2f}S\n",
//                   acquisition_number,
//                   time_acq.value().get_elapsed_seconds());
//             ++acquisition_number;
//             acq_progress = 0;
//             if (was_pedestals) {
//                 pedestals.save_pedestals();
//             }
//         }
//         time_acq = std::nullopt;
//     }
// }
//    sls::Receiver r(argc, argv);
// auto zmq_listen(std::stop_token stop,
//                 std::barrier<> &sync_barrier,
//                 const Arguments &args,
//                 const GainData &gains,
//                 PedestalsLibrary &pedestals,
//                 uint16_t port) -> void {

struct AcqContext {
    uint32_t bitmode;
    std::array<uint32_t, 2> detshape;
    std::array<DataStreamHandler *, 2> handlers;
    bool is_first_receiver;
    PedestalsLibrary &pedestals;
    uint16_t port;
    std::array<uint16_t, 2> udp_ports;
    // zmq::socket_t &zmq_send;
    /// Bit depth, from start acquisition header
    /// Detector shape, from start acquisition header
};

// ━  ┃  ┏ ┳ ┓ ┏ ┯ ┓ ┏ ┳ ┓ ┏ ┯ ┓
// ┅  ┇  ┣ ╋ ┫ ┣ ┿ ┫ ┠ ╂ ┨ ┠ ┼ ┨
// ┉  ┋  ┗ ┻ ┛ ┗ ┷ ┛ ┗ ┻ ┛ ┗ ┷ ┛

auto header_from_framedata(const slsDetectorDefs::sls_receiver_header &recHeader,
                           const slsDetectorDefs::dataCallbackHeader &dataHeader,
                           const AcqContext &ctx) -> SLSHeader {
    SLSHeader out;
    out.acqIndex = dataHeader.acqIndex;
    out.addJsonHeader = dataHeader.addJsonHeader;
    out.bitmode = ctx.bitmode;
    out.column = recHeader.detHeader.column;
    out.detshape = ctx.detshape;
    out.detType = recHeader.detHeader.detType;
    out.dls = DLSHeaderAdditions::from_map(dataHeader.addJsonHeader);
    out.expLength = recHeader.detHeader.expLength;
    out.frameIndex = dataHeader.frameIndex;
    out.frameNumber = recHeader.detHeader.frameNumber;
    out.progress = dataHeader.progress;
    out.row = recHeader.detHeader.row;
    out.shape = {static_cast<uint32_t>(dataHeader.shape.x),
                 static_cast<uint32_t>(dataHeader.shape.y)};
    out.udp_port = dataHeader.udpPort;
    return out;
}

int StartAcq(const slsDetectorDefs::startCallbackHeader header, void *objectPointer) {
    auto &ctx = *reinterpret_cast<AcqContext *>(objectPointer);
    assert(header.udpPort.size() == 2);
    ctx.udp_ports = {static_cast<uint16_t>(header.udpPort[0]),
                     static_cast<uint16_t>(header.udpPort[1])};
    --threads_waiting;
    print(
        "┏━━ Start Acquisition on receiver TCP port: {} (tid:{})\n\
┃ UDP Ports:      {}\n\
┃ Dynamic Range:  {}\n\
┃ Detector Shape: {} x {}\n\
┃ File Path:      {}\n\
┃ File Name:      {}\n\
┃ File Index:     {}\n\
┃ Quad:           {}\n\
┗ Additional Header: {}\n",
        ctx.port,
        std::this_thread::get_id(),
        header.udpPort,
        header.dynamicRange,
        header.detectorShape.x,
        header.detectorShape.y,
        header.filePath,
        header.fileName,
        header.fileIndex,
        header.quad,
        header.addJsonHeader);

    ctx.bitmode = header.dynamicRange;
    ctx.detshape = {static_cast<uint32_t>(header.detectorShape.x),
                    static_cast<uint32_t>(header.detectorShape.y)};
    if (ctx.is_first_receiver) {
        acq_progress = 0.0f;
    }
    return 0;
}

// /** Acquisition Finished Call back */
void EndAcq(const slsDetectorDefs::endCallbackHeader header, void *objectPointer) {
    auto &ctx = *reinterpret_cast<AcqContext *>(objectPointer);
    // std::vector<uint32_t> udpPort;
    //         std::vector<uint64_t> completeFrames;
    //         std::vector<uint64_t> lastFrameIndex;
    print(
        "┏━━ End Acquisition on receiver TCP port: {} (tid:{})\n"
        "┃ UDP Ports:        {}\n"
        "┃ Complete Frames:  {}\n"
        "┗ Last Frame Index: {}\n",
        ctx.port,
        std::this_thread::get_id(),
        header.udpPort,
        header.completeFrames,
        header.lastFrameIndex);

    bool was_pedestals = ctx.handlers[0]->is_pedestal_mode;
    ctx.handlers[0]->end_acquisition();
    ctx.handlers[1]->end_acquisition();

    if (ctx.is_first_receiver) {
        print("Acquisition {} complete in {:.2f}S\n", acquisition_number, 0.0f);
        //   time_acq.value().get_elapsed_seconds());
        ++acquisition_number;
        // acq_progress = 0;
        if (was_pedestals) {
            ctx.pedestals.save_pedestals();
        }
    }
    ++threads_waiting;
}

void GotData(slsDetectorDefs::sls_receiver_header &header,
             slsDetectorDefs::dataCallbackHeader callbackHeader,
             char *dataPointer,
             size_t &imageSize,
             void *objectPointer) {
    // NOTE: THIS FUNCTION IS CALLED FROM A THREAD PER STREAM
    auto &ctx = *reinterpret_cast<AcqContext *>(objectPointer);
    auto sls_header = header_from_framedata(header, callbackHeader, ctx);
    // Find the handler for this UDP port
    auto &handler = *ctx.handlers[callbackHeader.udpPort == ctx.udp_ports[0] ? 0 : 1];
    // print(
    //     "┏━━ Got Frame on receiver TCP port {} (tid:{})\n"
    //     "┃ UDP Ports:         {}\n"
    //     "┗ Detector row, col: {}, {} ({}, {})\n",
    //     // "┃ Complete Frames:  {}\n"
    //     // "┗ Last Frame Index: {}\n",
    //     ctx.port,
    //     std::this_thread::get_id(),
    //     callbackHeader.udpPort,
    //     header.detHeader.row,
    //     header.detHeader.column,
    //     sls_header.row,
    //     sls_header.column);
    if (!handler.validate_header(sls_header)) {
        print("{}: Invalid header\n", callbackHeader.udpPort);
        return;
    }
    std::span<uint16_t> data = {reinterpret_cast<uint16_t *>(dataPointer),
                                imageSize / 2};
    handler.process_frame(sls_header, data);

    if (ctx.is_first_receiver) {
        acq_progress = sls_header.progress;
    }
}

auto start_zmq_sender(zmq::context_t &context, uint16_t port) -> zmq::socket_t {
    zmq::socket_t send{context, zmq::socket_type::push};
    send.set(zmq::sockopt::sndhwm, 50000);
    send.set(zmq::sockopt::sndbuf, 128 * 1024 * 1024);
    send.set(zmq::sockopt::sndtimeo, 10000);
    auto zmq_bind_spec = fmt::format("tcp://0.0.0.0:{}", port);
    send.bind(zmq_bind_spec);
    print("Binding sending ZMQ to {}\n", zmq_bind_spec);
    return send;
}
auto start_receiver(std::stop_token stop,
                    std::barrier<> &sync_barrier,
                    const Arguments &args,
                    const GainData &gains,
                    PedestalsLibrary &pedestals,
                    uint16_t port) -> void {
    // This will be handled in start/end acq callbacks
    ++threads_waiting;

    sls::Receiver r(port);

    // For now, each thread gets it's own context. We can experiment
    // with shared later. The Guide (not that one) suggests one IO
    // thread per GB/s of data, and we have 2 GB/s per module (e.g.
    // each IO thread can cope with one half-module).
    zmq::context_t ctx, ctx2;
    auto send_a = start_zmq_sender(ctx, (port - args.rx_port) * 2 + args.zmq_send_port);
    auto send_b =
        start_zmq_sender(ctx, (port - args.rx_port) * 2 + args.zmq_send_port + 1);

    CudaStream stream, stream2;
    DataStreamHandler handler(args, port, stream, gains, pedestals, send_a);
    DataStreamHandler handler2(args, port, stream2, gains, pedestals, send_b);

    AcqContext context{.handlers = {&handler, &handler2},
                       .is_first_receiver = (port == args.rx_port),
                       .pedestals = pedestals,
                       .port = port};

    r.registerCallBackStartAcquisition(StartAcq, &context);
    r.registerCallBackAcquisitionFinished(EndAcq, &context);
    r.registerCallBackRawDataReady(GotData, &context);

    // Keep the receiver alive as long as we aren't stopping
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(80ms);
    }
}
#pragma region Launcher

auto do_live(Arguments &args) -> void {
    print(
        "                   __   _\n"
        "                  / /  (_)  _____\n"
        "                 / /__/ / |/ / -_)\n"
        "                /____/_/|___/\\__/\n\n");

    auto gain_maps = getenv_or("GAIN_MAPS", GAIN_MAPS).value();
    print("GPU:      {}\n", args.cuda_device_signature);
    if (args.detector == JF1M) {
        print("Detector: {}\n", JF1M_Display);
    } else if (args.detector == JF9M_SIM) {
        print("Detector: {}\n", JF9M_SIM_Display);
    } else {
        print("Detector: {}\n", styled(args.detector, emphasis::bold));
    }

    // Load calibration data into device memory for efficient access
    auto gains = GainData(gain_maps, args.detector);
    gains.upload();

    auto pedestals = PedestalsLibrary(args.detector);

    print("Starting up listeners on TCP ports {}-{}\n",
          args.rx_port,
          args.rx_port + args.rx_listeners - 1);
    if (!args.require_pedestals) {
        print(style::warning, "Running without requiring pedestal data\n");
        // Now we know how many workers, we can construct the global barrier
        auto barrier = std::barrier{args.rx_listeners};
        {
            std::vector<std::jthread> threads;
            for (uint16_t port = args.rx_port; port < args.rx_port + args.rx_listeners;
                 ++port) {
                threads.emplace_back(start_receiver,
                                     global_stop.get_token(),
                                     std::ref(barrier),
                                     args,
                                     std::cref(gains),
                                     std::ref(pedestals),
                                     port);
                std::jthread &thread = threads.back();
                std::string name = fmt::format("listen_{}", port);
                pthread_setname_np(thread.native_handle(), name.c_str());
            }
            while (true) {
                if (!args.no_progress) {
                    if (threads_waiting == args.rx_listeners && !in_acquisition) {
                        spinner("All listeners waiting");
                    } else {
                        auto msg = fmt::format(
                            "  Progress {:3}: {:3.2f} %                  \r",
                            acquisition_number,
                            acq_progress);
                        std::cout << msg << std::flush;
                    }
                }
                std::this_thread::sleep_for(80ms);
            }
        }
        // Only happens if we change to terminate
        print("All processing complete.\n");
    }
}