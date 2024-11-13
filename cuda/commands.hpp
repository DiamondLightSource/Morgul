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
};

auto do_correct(Arguments &args) -> void;