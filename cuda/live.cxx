
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <pthread.h>
#include <sls/Receiver.h>

#include <algorithm>
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
#include <variant>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "bitshuffle.h"
#include "blockingconcurrentqueue.h"
#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "concurrentqueue.h"
#include "constants.hpp"
#include "cuda_common.hpp"
#include "cuda_profiler_api.h"
#include "hdf5_tools.hpp"
#include "kernels.h"
#include "lz4.h"
#include "readerwriterqueue.h"

using namespace fmt;
using json = nlohmann::json;
using moodycamel::BlockingConcurrentQueue;
using moodycamel::BlockingReaderWriterQueue;
using moodycamel::ConcurrentQueue;
using moodycamel::ReaderWriterQueue;

using namespace std::chrono_literals;

/// Location of some pedestal data to load. Make this automatic/specifiable later
const auto PEDESTAL_DATA = std::filesystem::path{"/dev/shm/pedestals.h5"};

std::stop_source global_stop;

/// Count how many threads are waiting, so we know if everything is idle
std::atomic_int threads_waiting_proc{0};

/// How many threads are currently actively processing frame data
std::atomic_int threads_receiving{0};

std::atomic_bool in_acquisition{false};
std::atomic_int acquisition_number{0};
std::atomic<float> acq_progress{0};
std::vector<float> per_rec_progress{};
std::atomic<float> total_processing_time;

/// Fetch and update an atomic float with the maximum of two values
inline float atomic_fetch_max(std::atomic<float> &obj, float arg) noexcept {
    float old = obj.load(std::memory_order_relaxed);
    while (old < arg
           && !obj.compare_exchange_weak(
               old, arg, std::memory_order_release, std::memory_order_relaxed)) {
        // old is updated by compare_exchange_weak automatically
    }
    return old;
}

/// Get an environment variable if present, with optional default
auto getenv_or(std::string name, std::optional<std::string> _default = std::nullopt)
    -> std::optional<std::string> {
    auto data = std::getenv(name.c_str());
    if (data == nullptr) {
        return _default;
    }
    return {data};
}

auto rfc3339_now() -> std::string {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto time_t_now = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_now);
    return fmt::format("{}.{:03d}{}",
                       buf,
                       static_cast<int>(ms.count()),
                       tm_now.tm_gmtoff == 0
                           ? "Z"
                           : fmt::format("{:+03d}:{:02d}",
                                         tm_now.tm_gmtoff / 3600,
                                         std::abs((tm_now.tm_gmtoff % 3600) / 60)));
}

auto get_thread_name() -> std::string {
    char name[16];
    pthread_getname_np(pthread_self(), name, 16);
    return std::string(name);
}

auto thread_id_str() -> std::string {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

auto log_prefix() -> std::string {
    return format("{} {:16}::{}", rfc3339_now(), get_thread_name(), thread_id_str());
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

#pragma region Header Parsing

auto read_boolish(std::string value) -> bool {
    if (value.size() == 0) {
        return false;
    }
    if (value != "true" && value != "false") {
        throw std::runtime_error(
            fmt::format("Got non-boolish json value: '{}'", value));
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
            double wavelength_angstrom = std::strtod(value.c_str(), nullptr);
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
    // uint32_t bitmode;
    // std::array<uint32_t, 2> detshape;
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
        std::optional<PedestalData> pd_;
        try {
            pd_ = PedestalData(path, _detector);
        } catch (std::runtime_error err) {
            print(style::warning,
                  "Warning: Could not load pedestal data file {}\n",
                  path);
            return false;
        }
        auto pd = std::move(pd_).value();
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

/// @brief The basic unit of communication between threads inside DataStreamHandler
struct FrameData {
    slsDetectorDefs::sls_detector_header header;
    std::optional<slsDetectorDefs::dataCallbackHeader> callbackHeader;
    std::vector<uint8_t> data;
    std::optional<std::tuple<size_t, size_t>> is_pedestals;
};

namespace acqstate {
struct Starting {};
struct ImageReceived {};
struct Ended {};

}  // namespace acqstate

using AcquisitionState =
    std::variant<acqstate::Starting, acqstate::ImageReceived, acqstate::Ended>;
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

    DataStreamHandler(
        const Detector &detector,
        uint16_t udp_port,
        uint16_t zmq_port,
        const GainData &gains,
        PedestalsLibrary &pedestals,
        std::shared_ptr<BlockingConcurrentQueue<AcquisitionState>> feedback)
        : _detector(detector),
          _port(udp_port),
          gains(gains),
          pedestals(pedestals),
          _frames(std::make_shared<BlockingReaderWriterQueue<FrameData>>(32)),
          _feedback(feedback) {
        // Work out the maximum size the compressed data can be, add 12 for the HDF5 header
        size_t compress_size = LZ4_compressBound(sizeof(pixel_t) * HM_PIXELS) + 12;
        compression_buffer = std::vector<std::byte>(compress_size);
        assert(compression_buffer.size() == compress_size);

        // Build the ZMQ sender, to send results onwards
        _zmq_sender = start_zmq_sender(_zmq_context, zmq_port);

        // Create the data buffers used by processing
        pedestal_n = make_cuda_malloc<uint32_t>(GAIN_MODES.size() * HM_PIXELS);
        pedestal_x = make_cuda_malloc<uint32_t>(GAIN_MODES.size() * HM_PIXELS);
        pedestal_x_sq = make_cuda_malloc<uint64_t>(GAIN_MODES.size() * HM_PIXELS);
        reset_pedestal_buffers();
        dev_bitshuffle_buffer_out = make_cuda_malloc<std::byte>(HM_PIXELS * 2);
    }
    ~DataStreamHandler() {}

    /// @brief Pass a frame into the DataStreamHandler for processing.
    ///
    /// This should ideally do as little processing as possible to safely
    /// hand off the data to the processing/sending thread.
    ///
    /// @param header           The raw per-frame header
    /// @param callbackHeader   If present, the additional header passed to slsReceiver hooks.
    ///                         This might not be present, depending on if the handler is being
    ///                         run as part of an slsReceiver or raw frame receiver.
    /// @param data             The frame data.
    /// @returns                If the frame was deemed to be valid (pass early checks)
    auto pass_frame_into_handler(
        const slsDetectorDefs::sls_detector_header &header,
        const std::optional<slsDetectorDefs::dataCallbackHeader> callbackHeader,
        std::span<std::byte> data) -> bool;

    /// @brief Start listening for frame messages sent to the handler by pass_frame_into_handler
    ///
    /// @param stop     The stop-token, to request clean shutdown.
    auto listen(std::stop_token stop) -> void;

    auto frame_queue() -> std::shared_ptr<BlockingReaderWriterQueue<FrameData>> {
        return _frames;
    }
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

    const Detector _detector;
    uint16_t _port;
    const CudaStream stream;
    const GainData &gains;
    PedestalsLibrary &pedestals;
    zmq::context_t _zmq_context;
    std::optional<zmq::socket_t> _zmq_sender;
    std::shared_ptr<BlockingReaderWriterQueue<FrameData>> _frames;
    std::shared_ptr<BlockingConcurrentQueue<AcquisitionState>> _feedback;

    //
    // Data Buffers
    //
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

auto DataStreamHandler::pass_frame_into_handler(
    const slsDetectorDefs::sls_detector_header &header,
    const std::optional<slsDetectorDefs::dataCallbackHeader> callbackHeader,
    std::span<std::byte> data) -> bool {
    print("{}: Got image {}\n", _port, header.frameNumber);
    return true;
}

auto DataStreamHandler::listen(std::stop_token stop) -> void {
    FrameData data;
    while (!stop.stop_requested()) {
        if (!_frames->wait_dequeue_timed(data, 10ms)) {
            continue;
        }
        print("{}: Got data for frame {}\n", _port, data.header.frameNumber);
    }
}
#pragma region Validate Header

auto DataStreamHandler::validate_header(const SLSHeader &header) -> bool {
    // Once per acquisition, the first thread through gets this flag
    bool _expected = true;

    // Validate this matches our expectations
    if (header.shape != std::array{HM_WIDTH, HM_HEIGHT}) {
        print(style::error,
              "{}: Error: Got wrong sized image ({}), expected (1024,256)",
              _port,
              header.shape);
        return false;
    }
    uint32_t det_w = std::get<0>(DETECTOR_SIZE.at(_detector));
    uint32_t det_h = std::get<1>(DETECTOR_SIZE.at(_detector)) * 2;

    // Handle knowing which module we handle
    auto hmi = header.column * det_h + header.row;
    if (!known_hmi) {
        known_hmi = hmi;
    } else {
        if (known_hmi != hmi) {
            print(style::error,
                  "{}: Fatal Error: Got fed mix of module index; hmi={} instead of "
                  "initial {} on frame {} are your routing "
                  "crossed?",
                  _port,
                  hmi,
                  known_hmi,
                  header.frameIndex);
            return false;
            std::exit(1);
        }
    }
    if (!header.dls.energy) {
        print(style::warning, "Warning: Did not get energy in addJsonHeader packet\n");
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
        } else {
            if (!pedestals.has_pedestals(exposure_ns, known_hmi.value())
                && !header.dls.raw) {
                print(style::error,
                      "Warning: Do not have pedestals for {:.2f} ms HMI={}, cannot "
                      "correct.\n",
                      exposure_ns / 1000000.0,
                      known_hmi.value());
                return false;
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

    uint32_t det_w = std::get<0>(DETECTOR_SIZE.at(_detector));
    uint32_t det_h = std::get<1>(DETECTOR_SIZE.at(_detector)) * 2;

    zmq::multipart_t send_msgs;
    json send_header;
    send_header["frameIndex"] = header.frameIndex;
    send_header["row"] = header.row;
    send_header["column"] = header.column;
    send_header["shape"] = header.shape;
    send_header["detshape"] = std::array{det_w, det_h};
    send_header["bitmode"] = 16;
    send_header["expLength"] = header.expLength;
    send_header["acquisition"] = acquisition_number.load();
    send_msgs.push_back(zmq::message_t(send_header.dump()));
    send_msgs.push_back(zmq::message_t(compression_buffer.data(), current_index));
    auto time_push = Timer();
    if (send_onwards
        && zmq::send_multipart(_zmq_sender.value(), send_msgs) == std::nullopt) {
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
        // Reset the buffers, ready for next time
        reset_pedestal_buffers();
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

auto header_from_framedata(const slsDetectorDefs::sls_receiver_header &recHeader,
                           const slsDetectorDefs::dataCallbackHeader &dataHeader)
    -> SLSHeader {
    SLSHeader out;
    out.acqIndex = dataHeader.acqIndex;
    out.addJsonHeader = dataHeader.addJsonHeader;
    out.column = recHeader.detHeader.column;
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

/// @brief Hold contextual information across callbacks from slsReceiver
struct SLSReceiverContext {
    uint16_t tcp_port;
    std::array<uint16_t, 2> udp_ports;
    std::array<std::shared_ptr<DataStreamHandler>, 2> handlers;
    std::array<float, 2> cumulative_time;
    std::optional<std::tuple<size_t, size_t>> is_pedestals;
    /// Was there something wrong, that we need to skip this acquisition?
    bool skip_acquisition;
};

int StartAcq(const slsDetectorDefs::startCallbackHeader header, void *objectPointer) {
    auto &ctx = *reinterpret_cast<SLSReceiverContext *>(objectPointer);
    ctx.skip_acquisition = false;
    assert(header.udpPort.size() == 2);
    ctx.udp_ports = {static_cast<uint16_t>(header.udpPort[0]),
                     static_cast<uint16_t>(header.udpPort[1])};
    --threads_waiting_proc;
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
        ctx.tcp_port,
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

    return 0;
}

// /** Acquisition Finished Call back */
void EndAcq(const slsDetectorDefs::endCallbackHeader header, void *objectPointer) {
    auto &ctx = *reinterpret_cast<SLSReceiverContext *>(objectPointer);
    print(
        "┏━━ End Acquisition on receiver TCP port: {} (tid:{})\n"
        "┃ UDP Ports:        {}\n"
        "┃ Complete Frames:  {}\n"
        "┃ Last Frame Index: {}\n"
        "┗ Total processing time/frame: {:.2f} {:.2f} ms\n",
        ctx.tcp_port,
        std::this_thread::get_id(),
        header.udpPort,
        header.completeFrames,
        header.lastFrameIndex,

        ctx.cumulative_time[0] * 1000.0 / header.completeFrames[0],
        ctx.cumulative_time[1] * 1000.0 / header.completeFrames[1]);

    total_processing_time += ctx.cumulative_time[0] + ctx.cumulative_time[1];

    bool was_pedestals = ctx.is_pedestals.has_value();

    ++threads_waiting_proc;
}

void GotData(slsDetectorDefs::sls_receiver_header &header,
             slsDetectorDefs::dataCallbackHeader callbackHeader,
             char *dataPointer,
             size_t &imageSize,
             void *objectPointer) {
    auto process_timer = Timer();
    threads_receiving += 1;
    // NOTE: THIS FUNCTION IS CALLED FROM A THREAD PER STREAM
    auto &ctx = *reinterpret_cast<SLSReceiverContext *>(objectPointer);
    // Handle skipping this acquisition, if an unrecoverable error previously occured
    if (ctx.skip_acquisition) {
        threads_receiving -= 1;
        return;
    }

    auto port_instance = callbackHeader.udpPort == ctx.udp_ports[0] ? 0 : 1;
    auto &handler = *ctx.handlers[port_instance];

    // assert(handler.try_emplace(header, {callbackHeader}, )
    ctx.skip_acquisition = !handler.pass_frame_into_handler(
        header.detHeader,
        {callbackHeader},
        std::span(reinterpret_cast<std::byte *>(dataPointer), imageSize));
    threads_receiving -= 1;
}

auto start_receiver(std::stop_token stop,
                    std::shared_ptr<DataStreamHandler> handler_a,
                    std::shared_ptr<DataStreamHandler> handler_b,
                    uint16_t port) -> void {
    sls::Receiver r(port);
    SLSReceiverContext context{.tcp_port = port, .handlers = {handler_a, handler_b}};

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
    } else if (args.detector == JF9M) {
        print("Detector: {}\n", JF9M_Display);
    } else {
        print("Detector: {}\n", styled(args.detector, emphasis::bold));
    }

    // Load calibration data into device memory for efficient access
    auto gains = GainData(gain_maps, args.detector);
    gains.upload();

    auto pedestals = PedestalsLibrary(args.detector);

    auto feedback = std::make_shared<BlockingConcurrentQueue<AcquisitionState>>(32);

    print("Starting up listeners on TCP ports {}-{}\n",
          args.rx_port,
          args.rx_port + args.rx_listeners - 1);
    // Now we know how many workers, we can construct the global barrier
    auto barrier = std::barrier{args.rx_listeners};
    {
        std::vector<std::jthread> threads;
        for (uint16_t port = args.rx_port; port < args.rx_port + args.rx_listeners;
             ++port) {
            uint16_t expected_udp_port = (port - args.rx_port) * 2 + 30000;
            uint16_t zmq_port = (port - args.rx_port) * 2 + args.zmq_send_port + 1;

            // Make two handlers for this receiver port
            auto handler_a = std::make_shared<DataStreamHandler>(
                args.detector, expected_udp_port, zmq_port, gains, pedestals, feedback);
            auto handler_b = std::make_shared<DataStreamHandler>(args.detector,
                                                                 expected_udp_port + 1,
                                                                 zmq_port + 1,
                                                                 gains,
                                                                 pedestals,
                                                                 feedback);

            // Launch the processing threads
            threads.emplace_back(
                &DataStreamHandler::listen, handler_a.get(), global_stop.get_token());
            pthread_setname_np(threads.back().native_handle(),
                               fmt::format("process_{}", expected_udp_port).c_str());
            threads.emplace_back(
                &DataStreamHandler::listen, handler_b.get(), global_stop.get_token());
            pthread_setname_np(
                threads.back().native_handle(),
                fmt::format("process_{}", expected_udp_port + 1).c_str());

            // Launch receiver threads
            threads.emplace_back(
                start_receiver, global_stop.get_token(), handler_a, handler_b, port);
            std::jthread &thread = threads.back();
            std::string name = fmt::format("listen_{}", port);
            pthread_setname_np(thread.native_handle(), name.c_str());
        }
        // Do the live update handling - we manage state in this thread
        AcquisitionState state;
        size_t currently_active = 0;
        size_t num_images_seen = 0;
        size_t total_expected_images = 0;
        size_t acquisition_number = 0;
        auto last_print = std::chrono::steady_clock::now();

        while (true) {
            if (!args.no_progress) {
                if (feedback->wait_dequeue_timed(state, 80ms)) {
                    if (std::holds_alternative<acqstate::Starting>(state)) {
                        auto msg = std::get<acqstate::Starting>(state);
                        currently_active += 1;

                    } else if (std::holds_alternative<acqstate::ImageReceived>(state)) {
                        auto msg = std::get<acqstate::ImageReceived>(state);
                        num_images_seen += 1;
                    } else if (std::holds_alternative<acqstate::Ended>(state)) {
                        auto msg = std::get<acqstate::Ended>(state);
                        num_images_seen = 0;
                        currently_active -= 1;
                    }
                }
                // Have we had enough time elapse for an update?
                auto elapsed = std::chrono::steady_clock::now() - last_print;
                if (elapsed < 80ms) {
                    continue;
                }
                // We're due an output update
                if (currently_active == 0) {
                    spinner("All listeners waiting");
                } else {
                    // In the middle of a collection, work out what to print
                    auto msg =
                        format(" {}: {} images\r", acquisition_number, num_images_seen);
                    std::cout << msg << std::flush;
                }
            }
        }
        // Only happens if we change to terminate
        print("All processing complete.\n");
    }
}