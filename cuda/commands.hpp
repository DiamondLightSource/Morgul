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
};

auto do_correct(Arguments &args) -> void;