#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <tuple>

const auto GAIN_MAPS = std::filesystem::path{"/dls_sw/apps/jungfrau/calibration"};

using Detector = std::string;
constexpr static Detector JF1M{"JF1M"};

constexpr static std::tuple<uint16_t, uint16_t> MODULE_SHAPE{1024, 512};
constexpr static std::tuple<uint16_t, uint16_t> HALF_MODULE_SHAPE{1024, 256};

/// Maps detector name to known module names and positions
const std::map<Detector, std::map<std::string, std::tuple<int, int>>> KNOWN_DETECTORS =
    {{JF1M, {{"M420", {0, 0}}, {"M418", {0, 1}}}}};

/// Size of detector, in (columns, rows)
const std::map<Detector, std::tuple<int, int>> DETECTOR_SIZE = {{JF1M, {1, 2}}};

/// Names for gain modes
const std::vector<uint8_t> GAIN_MODES = {0, 1, 2};

// template <
// [jf1md-00]
// position = bottom
// module = M420

// [jf1md-01]
// position = top
// module = M418

// [jf4mpsi-00]
// position = "00"
// module = M120

// [jf4mpsi-01]
// position = "01"
// module = M210

// [jf4mpsi-02]
// position = "10"
// module = M202

// [jf4mpsi-03]
// position = "11"
// module = M115

// [jf4mpsi-04]
// position = "20"
// module = M049

// [jf4mpsi-05]
// position = "21"
// module = M043

// [jf4mpsi-06]
// position = "30"
// module = M060

// [jf4mpsi-07]
// position = "31"
// module = M232

// [Grey-Area.local]
// calibration = /Users/graeme/data/jungfrau-1m-bench/calibrations

// [Ethics-Gradient]
// calibration = /Users/graeme/data/jungfrau-1m-bench/calibrations

// [diamond.ac.uk]
// calibration = /dls_sw/apps/jungfrau/calibration
