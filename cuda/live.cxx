
#include <stop_token>
#include <thread>
#include <zmq.hpp>

#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "hdf5_tools.hpp"

using namespace fmt;

std::stop_source global_stop;

auto zmq_listen(std::stop_token stop, uint16_t port) -> void {
    print("{}: In listening thread.\n", port);
}

auto do_live(Arguments &args) -> void {
    print(
        "                   __   _\n"
        "                  / /  (_)  _____\n"
        "                 / /__/ / |/ / -_)\n"
        "                /____/_/|___/\\__/\n\n");

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
            threads.emplace_back(zmq_listen, global_stop.get_token(), port);
        }
    }
    print("All processing complete.\n");
}