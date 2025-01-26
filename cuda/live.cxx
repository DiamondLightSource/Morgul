
#include <fmt/ranges.h>

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <optional>
#include <stop_token>
#include <thread>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"
#include "hdf5_tools.hpp"
#include "kernels.h"

using namespace fmt;
using json = nlohmann::json;
using namespace std::chrono_literals;

/// Location of some pedestal data to load. Make this automatic/specifiable later
const auto PEDESTAL_DATA = std::filesystem::path{
    "/scratch/nickd/PEDESTALS/jf1md_0.5ms_2024-10-03_12-42-49_pedestal.h5"};

std::stop_source global_stop;

/// Count how many threads are waiting, so we know if everything is idle
std::atomic_int threads_waiting;

/// Get an environment variable if present, with optional default
auto getenv_or(std::string name, std::optional<std::string> _default = std::nullopt)
    -> std::optional<std::string> {
    auto data = std::getenv(name.c_str());
    if (data == nullptr) {
        return _default;
    }
    return {data};
}

void wait_spinner(const std::string_view &message) {
    static int index = 0;
    std::this_thread::sleep_for(80ms);
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
    print("  {} {}\r", message, ball[index]);
    std::cout << std::flush;
}

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
    std::string addJsonHeader;
};
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
        j.at("addJsonHeader").get_to(h.addJsonHeader);
    }
}

auto zmq_listen(std::stop_token stop,
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

    // Once we receive an HMI, we must always receive the same one
    std::optional<uint32_t> known_hmi;

    // Keep track of how many images we have seen/the highest index.
    // since the last end-packet.
    int num_images_seen = 0;
    int highest_image_seen = 0;

    // Use our own stream
    CudaStream stream;

    while (!stop.stop_requested()) {
        std::vector<zmq::message_t> recv_msgs;
        // Wait for the next message. Count waiting so we know when we are idle.
        ++threads_waiting;
        const auto ret = zmq::recv_multipart(sub, std::back_inserter(recv_msgs));
        --threads_waiting;
        assert(ret);
        auto json = json::parse(recv_msgs[0].to_string_view());
        auto header = json.template get<SLSHeader>();
        if (ret == 1) {
            if (header.bitmode == 0) {
                print("{}: Got end packet; saw {} / {} images.\n",
                      port,
                      num_images_seen,
                      highest_image_seen);
            }
            num_images_seen = 0;
            highest_image_seen = 0;
            continue;
        } else if (ret != 2) {
            print(style::error,
                  "{}: Error: Got unexpected multipart message length {}\n",
                  port,
                  ret);
            continue;
        }
        // We have a standard image packet

        // Validate this matches our expectations
        if (header.shape != std::array{1024u, 256u}) {
            print(style::error,
                  "{}: Error: Got wrong sized image ({}), expected (1024,256)",
                  port,
                  header.shape);
            continue;
        }
        uint32_t det_x = std::get<0>(DETECTOR_SIZE.at(args.detector));
        uint32_t det_y = std::get<1>(DETECTOR_SIZE.at(args.detector)) * 2;
        if (header.detshape != std::array{det_x, det_y}) {
            print(style::error,
                  "{}: Error: Got wrong sized detector {}; expected {},{}",
                  port,
                  header.detshape,
                  det_x,
                  det_y);
            continue;
        }
        print("{}: {}", port, recv_msgs[0].to_string_view());
        auto hmi = header.column * det_y + header.row;
        if (!known_hmi) {
            known_hmi = hmi;
        } else {
            if (known_hmi != hmi) {
                print(style::error,
                      "{}: Fatal Error: Got fed mix of module index; are your routing "
                      "crossed?",
                      port);
                std::exit(1);
            }
        }
        // If here, then we have an expected next packet
        // send it to the GPU for processing
        call_jungfrau_image_corrections(stream,
                                        gains.get_gpu_ptrs(hmi),
                                        pedestals.get_gpu_ptrs(hmi),
                                        static_cast<uint16_t *>(recv_msgs[1].data()),
                                        0);
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
    {
        std::vector<std::jthread> threads;
        for (uint16_t port = args.zmq_port; port < args.zmq_port + args.zmq_listeners;
             ++port) {
            threads.emplace_back(zmq_listen,
                                 global_stop.get_token(),
                                 args,
                                 std::cref(gains),
                                 std::cref(pedestals),
                                 port);
        }
        while (true) {
            while (threads_waiting == args.zmq_listeners) {
                wait_spinner("All listeners waiting");
            }
        }
    }
    // Only happens if we change to terminate
    print("All processing complete.\n");
}
