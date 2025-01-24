
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
#include "hdf5_tools.hpp"

using namespace fmt;
using json = nlohmann::json;
using namespace std::chrono_literals;

const auto PREVIOUS_PEDESTAL = std::filesystem::path{"/dev/shm/current_pedestal.h5"};

std::stop_source global_stop;

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

auto zmq_listen(std::stop_token stop, const Arguments &args, uint16_t port) -> void {
    // For now, each thread gets it's own context. We can experiment
    // with shared later. The Guide (not that one) suggests one IO
    // thread per GB/s of data, and we have 2 GB/s per module (e.g.
    // each IO thread can cope with one half-module).
    zmq::context_t ctx;
    zmq::socket_t sub{ctx, zmq::socket_type::sub};
    sub.connect(format("tcp://{}:{}", args.zmq_host, port));
    sub.set(zmq::sockopt::subscribe, "");

    int num_images_seen = 0;
    int highest_image_seen = 0;
    while (true) {
        std::vector<zmq::message_t> recv_msgs;
        ++threads_waiting;
        const auto ret = zmq::recv_multipart(sub, std::back_inserter(recv_msgs));
        --threads_waiting;
        if (!ret) {
            print(style::error,
                  "{}: Error: Got unexpected multipart message length {}\n",
                  port,
                  ret);
            throw std::runtime_error("Unexpected multipart message");
        }
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
        } else if (ret == 2) {
            ++num_images_seen;
            highest_image_seen =
                std::max(highest_image_seen, static_cast<int>(header.frameIndex + 1));
            // print("{}: Received {},{}#{:5}: {}",
            //       port,
            //       header.column,
            //       header.row,
            //       header.frameIndex,
            //       recv_msgs[0].to_string_view());
            // print("       And size: {}\n", recv_msgs[1].size());
        } else {
            print(style::error,
                  "{}: Error: Got unexpected multipart message length {}\n",
                  port,
                  ret);
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
    // print("Using Gains:    {}\n", styled(gain_maps, style::path));
    auto gains = GainData(gain_maps, args.detector);
    gains.upload();

    print("Connecting to {}\n",
          styled(format("tcp://{}:{}-{}",
                        args.zmq_host,
                        args.zmq_port,
                        args.zmq_port + args.zmq_listeners - 1),
                 style::url));
    {
        std::vector<std::jthread> threads;
        for (int port = args.zmq_port; port < args.zmq_port + args.zmq_listeners;
             ++port) {
            threads.emplace_back(zmq_listen, global_stop.get_token(), args, port);
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