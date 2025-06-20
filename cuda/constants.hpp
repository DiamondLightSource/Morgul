#pragma once

#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>

const auto GAIN_MAPS = std::filesystem::path{"/scratch/nickd/GAINS"};

using Detector = std::string;

// Give ourselves a way to identify old cuda e.g. less than stellar constexpre support
#if __CUDACC_VER_MAJOR__ == 12 && __CUDAACC_VER_MINOR__ <= 2
#define OLD_CUDA
#endif

constexpr static std::tuple<uint16_t, uint16_t> MODULE_SHAPE{1024, 512};
constexpr static std::tuple<uint16_t, uint16_t> HALF_MODULE_SHAPE{1024, 256};

constexpr static uint32_t HM_WIDTH = std::get<0>(HALF_MODULE_SHAPE);
constexpr static uint32_t HM_HEIGHT = std::get<1>(HALF_MODULE_SHAPE);
constexpr static uint32_t HM_PIXELS = HM_WIDTH * HM_HEIGHT;

using pixel_t = uint16_t;

#ifndef OLD_CUDA
constexpr static Detector JF1M{"JF1M"};
constexpr static Detector JF9M_SIM{"JF9M-SIM"};

/// Maps detector name to known module names and positions
const std::map<Detector, std::map<std::string, std::tuple<int, int>>> KNOWN_DETECTORS =
    {{JF1M, {{"M420", {0, 0}}, {"M418", {0, 1}}}},
     {JF9M_SIM,
      {
          {"SIM_M00", {0, 0}},
          {"SIM_M01", {0, 1}},
          {"SIM_M02", {0, 2}},
          {"SIM_M03", {0, 3}},
          {"SIM_M04", {0, 4}},
          {"SIM_M05", {0, 5}},
          {"SIM_M10", {0, 6}},
          {"SIM_M11", {0, 7}},
          {"SIM_M12", {0, 8}},
          {"SIM_M13", {0, 9}},
          {"SIM_M14", {0, 10}},
          {"SIM_M15", {0, 11}},
          {"SIM_M20", {0, 12}},
          {"SIM_M21", {0, 13}},
          {"SIM_M22", {0, 14}},
          {"SIM_M23", {0, 15}},
          {"SIM_M24", {0, 16}},
          {"SIM_M25", {0, 17}},
      }}};

/// Size of detector, in (columns, rows)
const std::map<Detector, std::tuple<int, int>> DETECTOR_SIZE = {{JF1M, {1, 2}},
                                                                {JF9M_SIM, {1, 18}}};

static const std::string JF1M_Display{
    "\e[1m\e[38;5;198mJ\e[39m\e[38;5;163mF\e[39m\e[38;5;129m1\e[39m\e[38;5;93mM\e["
    "39m\e[38;5;33m\e[39m\e[0m\n"};

static const std::string JF9M_SIM_Display{
    "\e[38;2;207;174;2mJ\e[39m\e[38;2;235;133;15mF\e[39m\e[38;2;251;91;40m9\e[39m\e[38;"
    "2;254;53;76mM\e[39m\e[38;2;243;24;116m-\e[39m\e[38;2;219;5;158mS\e[39m\e[38;2;185;"
    "1;197mI\e[39m\e[38;2;145;10;228mM\e[39m\e[38;2;103;32;248m\e[39m"};

#endif

/// Names for gain modes
constexpr std::array<uint8_t, 3> GAIN_MODES{0, 1, 2};

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
