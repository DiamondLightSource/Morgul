#pragma once

#include <string>
#include <vector>

#include "constants.hpp"

struct Arguments {
    bool verbose = false;
    int cuda_device_index = 0;

    std::string command;
    std::vector<std::string> sources;
    Detector detector;
    std::string cuda_device_signature;
    std::string zmq_host;
    uint16_t zmq_port;
    uint16_t zmq_listeners;
    uint16_t zmq_timeout;
};

auto do_correct(Arguments &args) -> void;
auto do_live(Arguments &args) -> void;