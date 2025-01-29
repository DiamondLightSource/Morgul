
#include <fmt/ranges.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
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
const auto PEDESTAL_DATA = std::filesystem::path{
    "/scratch/nickd/PEDESTALS/jf1md_0.5ms_2024-10-03_12-42-49_pedestal.h5"};

std::stop_source global_stop;

/// Count how many threads are waiting, so we know if everything is idle
std::atomic_int threads_waiting{0};
/// Used to identify the first validation each acquisition, to avoid spamming
std::atomic_bool is_first_validation_this_acquisition{false};
std::atomic_int acquisition_number{0};

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

struct DLSHeaderAdditions {
    bool pedestal = false;
    std::optional<double> energy;
};

class SLSHeader {
  public:
    uint32_t jsonversion;
    uint32_t bitmode;
    uint64_t fileIndex;
    std::array<uint32_t, 2> detshape;
    std::array<uint32_t, 2> shape;
    uint32_t size;
    size_t acqIndex;
    size_t frameIndex;
    double progress;
    std::string fname;
    uint32_t data;
    uint32_t completeImage;
    size_t frameNumber;
    uint32_t expLength;
    uint32_t packetNumber;
    uint32_t timestamp;
    uint32_t modId;
    uint32_t row;
    uint32_t column;
    uint64_t detSpec1;
    uint32_t detSpec2;
    uint32_t detSpec3;
    uint32_t detSpec4;
    uint32_t detType;
    uint32_t version;
    uint32_t flipRows;
    uint32_t quad;
    std::optional<json> addJsonHeader;
    /// DLS-specific additional headers that may be present in addJsonHeader
    DLSHeaderAdditions dls;
};
void from_json(const json &j, DLSHeaderAdditions &d) {
    if (j.contains("pedestal")) {
        j.at("pedestal").get_to(d.pedestal);
    }
    if (j.contains("energy")) {
        auto value = j.at("energy").template get<std::string>();
        d.energy = std::strtod(value.c_str(), nullptr);
    }
}

void from_json(const json &j, SLSHeader &h) {
    j.at("jsonversion").get_to(h.jsonversion);
    j.at("bitmode").get_to(h.bitmode);
    j.at("fileIndex").get_to(h.fileIndex);
    j.at("size").get_to(h.size);
    j.at("acqIndex").get_to(h.acqIndex);
    j.at("frameIndex").get_to(h.frameIndex);
    j.at("progress").get_to(h.progress);
    j.at("fname").get_to(h.fname);
    j.at("data").get_to(h.data);
    j.at("completeImage").get_to(h.completeImage);
    j.at("frameNumber").get_to(h.frameNumber);
    j.at("expLength").get_to(h.expLength);
    j.at("packetNumber").get_to(h.packetNumber);
    j.at("timestamp").get_to(h.timestamp);
    j.at("modId").get_to(h.modId);
    j.at("row").get_to(h.row);
    j.at("column").get_to(h.column);
    j.at("detSpec1").get_to(h.detSpec1);
    j.at("detSpec2").get_to(h.detSpec2);
    j.at("detSpec3").get_to(h.detSpec3);
    j.at("detSpec4").get_to(h.detSpec4);
    j.at("detType").get_to(h.detType);
    j.at("version").get_to(h.version);
    j.at("flipRows").get_to(h.flipRows);
    j.at("quad").get_to(h.quad);
    j.at("detshape")[0].get_to(h.detshape[0]);
    j.at("detshape")[1].get_to(h.detshape[1]);
    j.at("shape")[0].get_to(h.shape[0]);
    j.at("shape")[1].get_to(h.shape[1]);
    if (j.contains("addJsonHeader")) {
        h.addJsonHeader = j.at("addJsonHeader");
        h.dls = j["addJsonHeader"].template get<DLSHeaderAdditions>();
    }
}

class DataStreamHandler {
  public:
    // Once we receive an HMI, we must always receive the same one
    std::optional<uint32_t> known_hmi;

    // Keep track of how many images we have seen/the highest index.
    // since the last end-packet.
    int num_images_seen = 0;
    int highest_image_seen = 0;
    // Keep track of the last frame number seen, so we know if a frame was skipped
    uint64_t hm_frameNumber = 0;

    DataStreamHandler(const Arguments &args,
                      uint16_t port,
                      const CudaStream &stream,
                      const GainData &gains,
                      const PedestalData &pedestals,
                      zmq::socket_t &send_socket)
        : _args(args),
          _port(port),
          stream(stream),
          gains(gains),
          pedestals(pedestals),
          send(send_socket) {
        is_first_validation_this_acquisition.store(true);
        // Work out the maximum size the compressed data can be, add 12 for the HDF5 header
        size_t compress_size = LZ4_compressBound(2 * HM_HEIGHT * HM_WIDTH) + 12;
        compression_buffer = std::make_unique<std::byte[]>(compress_size);
    }

    auto validate_header(const SLSHeader &header) -> bool;
    auto process_frame(const SLSHeader &header, std::span<uint16_t> &frame) -> void;
    auto end_acquisition() -> void;

  private:
    const Arguments &_args;
    uint16_t _port;
    const CudaStream &stream;
    const GainData &gains;
    const PedestalData &pedestals;
    zmq::socket_t &send;
    std::unique_ptr<uint16_t[]> corrected_buffer =
        std::make_unique<uint16_t[]>(HM_HEIGHT * HM_WIDTH);
    std::unique_ptr<std::byte[]> compression_buffer;
};

auto DataStreamHandler::validate_header(const SLSHeader &header) -> bool {
    // Once per acquisition, the first thread through gets this flag
    bool _expected = true;
    bool first_acquisition =
        is_first_validation_this_acquisition.compare_exchange_strong(_expected, false);

    // Validate this matches our expectations
    if (header.shape != std::array{1024u, 256u}) {
        if (first_acquisition) {
            print(style::error,
                  "{}: Error: Got wrong sized image ({}), expected (1024,256)",
                  _port,
                  header.shape);
        }
        return false;
    }
    uint32_t det_x = std::get<0>(DETECTOR_SIZE.at(_args.detector));
    uint32_t det_y = std::get<1>(DETECTOR_SIZE.at(_args.detector)) * 2;
    if (header.detshape != std::array{det_x, det_y}) {
        if (first_acquisition) {
            print(style::error,
                  "{}: Error: Got wrong sized detector {}; expected {},{}",
                  _port,
                  header.detshape,
                  det_x,
                  det_y);
        }
        return false;
    }
    auto hmi = header.column * det_y + header.row;
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

    // If here, then we have an expected next packet
    ++num_images_seen;
    highest_image_seen =
        std::max(highest_image_seen, {static_cast<int>(header.frameIndex + 1)});
    if (hm_frameNumber != 0 && header.frameNumber > hm_frameNumber + 1) {
        auto num_skipped = header.frameNumber - hm_frameNumber - 1;
        print(style::warning,
              "hm {}: Warning: Skipped {} frames\n",
              known_hmi.value(),
              num_skipped);
    }
    return true;
}

auto DataStreamHandler::process_frame(const SLSHeader &header,
                                      std::span<uint16_t> &frame) -> void {
    auto energy = header.dls.energy.value_or(12.4);
    call_jungfrau_image_corrections(stream,
                                    gains.get_gpu_ptrs(known_hmi.value()),
                                    pedestals.get_gpu_ptrs(known_hmi.value()),
                                    frame.data(),
                                    corrected_buffer.get(),
                                    energy);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Construct the HDF5 header so that we can do direct chunk write
    // on the other end of the pipe
    // first 12 bytes are uint64_t BE array size and uint32_t BE block size
    // these are the precomputed values
    uint64_t &uncompress_size = *reinterpret_cast<uint64_t *>(compression_buffer.get());
    uint32_t &block_size = *reinterpret_cast<uint32_t *>(compression_buffer.get() + 8);
    // unsigned long long *alias64 = (unsigned long long *)scratch;

    uncompress_size = __builtin_bswap64(2 * 256 * 1024);
    block_size = __builtin_bswap32(8192);

    auto size = bshuf_compress_lz4(corrected_buffer.get(),
                                   compression_buffer.get() + 12,
                                   HM_HEIGHT * HM_WIDTH,
                                   2,
                                   4096);

    zmq::multipart_t send_msgs;
    // Form the multipart message to match Graeme's implementation
    // Header, if we wanted to send more information
    // json out_header;
    // out_header["frameIndex"] = header.frameIndex;
    // out_header["hmi"] = known_hmi.value();
    // send_msgs.push_back(zmq::message_t(out_header.dump()));
    std::vector<int> send_hdr = {static_cast<int>(header.frameIndex)};
    send_msgs.push_back(zmq::message_t(send_hdr));

    send_msgs.push_back(zmq::message_t(compression_buffer.get(), size + 12));
    zmq::send_multipart(send, send_msgs);
}

auto DataStreamHandler::end_acquisition() -> void {
    is_first_validation_this_acquisition.store(false);
}

auto zmq_listen(std::stop_token stop,
                std::barrier<> &sync_barrier,
                const Arguments &args,
                const GainData &gains,
                const PedestalData &pedestals,
                uint16_t port) -> void {
    // For now, each thread gets it's own context. We can experiment
    // with shared later. The Guide (not that one) suggests one IO
    // thread per GB/s of data, and we have 2 GB/s per module (e.g.
    // each IO thread can cope with one half-module).
    zmq::context_t ctx;

    zmq::socket_t sub{ctx, zmq::socket_type::sub};
    sub.connect(fmt::format("tcp://{}:{}", args.zmq_host, port));
    sub.set(zmq::sockopt::subscribe, "");

    // Set up the results sending port. Let's reuse the context, and
    // we can wait and see if this needs more resources.
    zmq::socket_t send{ctx, zmq::socket_type::push};
    send.set(zmq::sockopt::sndhwm, 50000);
    send.set(zmq::sockopt::sndbuf, 128 * 1024 * 1024);
    send.bind(
        fmt::format("tcp://0.0.0.0:{}", port - args.zmq_port + args.zmq_send_port));

    CudaStream stream;
    DataStreamHandler handler(args, port, stream, gains, pedestals, send);

    auto output_data = std::make_unique<uint16_t[]>(HM_HEIGHT * HM_WIDTH);

    while (!stop.stop_requested()) {
        // Wait for the next message. Count waiting so we know when we are idle.
        sync_barrier.arrive_and_wait();
        // Between acquisitions we wait as long as it takes
        sub.set(zmq::sockopt::rcvtimeo, -1);
        // Loop over images within an acquisition
        while (!stop.stop_requested()) {
            std::vector<zmq::message_t> recv_msgs;
            ++threads_waiting;
            const auto ret = zmq::recv_multipart(sub, std::back_inserter(recv_msgs));
            --threads_waiting;
            // All subsequent waits on this series of images can timeout
            sub.set(zmq::sockopt::rcvtimeo, static_cast<int>(args.zmq_timeout));

            if (!ret) {
                // If here, then we had a timeout waiting for images.
                print(style::error,
                      "{}: HMI={} Error: Timeout waiting for more images/end "
                      "notification\n",
                      port,
                      handler.known_hmi.value());
                break;
            }
            print("{}: {}", port, recv_msgs[0].to_string_view());
            auto header =
                json::parse(recv_msgs[0].to_string_view()).template get<SLSHeader>();
            if (ret == 1 && header.bitmode == 0) {
                print("{}: Received end packet\n", port);
                break;
            } else if (ret > 2) {
                print(style::error,
                      "{}: Error: Got unexpected multipart message length {}\n{}",
                      port,
                      ret.value(),
                      recv_msgs[0].to_string_view());
                continue;
            }

            // We have a standard image packet
            // Validate the header, and skip this image if invalid
            if (!handler.validate_header(header)) {
                continue;
            }
            std::span<uint16_t> data = {
                reinterpret_cast<uint16_t *>(recv_msgs[1].data()),
                recv_msgs[1].size() / 2};
            handler.process_frame(header, data);
        }
        handler.end_acquisition();
        // Now, wait until all frames have completed
        sync_barrier.arrive_and_wait();
        // If we are the first port
        if (port == args.zmq_port) {
            print("Acquisition {} complete\n", acquisition_number);
            ++acquisition_number;
        }
    }
}

auto do_live(Arguments &args) -> void {
    print(
        "                   __   _\n"
        "                  / /  (_)  _____\n"
        "                 / /__/ / |/ / -_)\n"
        "                /____/_/|___/\\__/\n\n");

    auto gain_maps = getenv_or("GAIN_MAPS", GAIN_MAPS).value();
    print("GPU:      {}\n", args.cuda_device_signature);
    print("Detector: {}\n", styled(args.detector, emphasis::bold));

    // Load calibration data into device memory for efficient access
    auto gains = GainData(gain_maps, args.detector);
    gains.upload();

    auto pedestals = PedestalData(PEDESTAL_DATA, args.detector);
    pedestals.upload();

    print("Connecting to {}\n",
          styled(fmt::format("tcp://{}:{}-{}",
                             args.zmq_host,
                             args.zmq_port,
                             args.zmq_port + args.zmq_listeners - 1),
                 style::url));
    // Now we know how many workers, we can construct the global barrier
    // worker_sync = std::make_unique<std::barrier<>>(args.zmq_listeners);
    auto barrier = std::barrier{args.zmq_listeners};
    {
        std::vector<std::jthread> threads;
        for (uint16_t port = args.zmq_port; port < args.zmq_port + args.zmq_listeners;
             ++port) {
            threads.emplace_back(zmq_listen,
                                 global_stop.get_token(),
                                 std::ref(barrier),
                                 args,
                                 std::cref(gains),
                                 std::cref(pedestals),
                                 port);
        }
        while (true) {
            while (threads_waiting == args.zmq_listeners) {
                spinner("All listeners waiting");
                std::this_thread::sleep_for(80ms);
            }
        }
    }
    // Only happens if we change to terminate
    print("All processing complete.\n");
}
