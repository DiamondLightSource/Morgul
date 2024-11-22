#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

#include "array2d.hpp"
#include "constants.hpp"

struct CalibrationDataPath {
    std::filesystem::path pedestal;
    std::optional<std::filesystem::path> mask;
    std::filesystem::path gain;
};
auto get_applicable_calibration_paths(float exposure_time, uint64_t timestamp)
    -> CalibrationDataPath;

enum class ModuleMode {
    FULL,
    HALF,
};
auto module_mode_from(std::string_view value) -> ModuleMode;

class PedestalData {
    typedef double pedestal_t;

  public:
    PedestalData(std::filesystem::path path, Detector detector);
    auto get_pedestal(size_t halfmodule_index, uint8_t gain_mode) const
        -> const Array2D<pedestal_t>& {
        return _modules.at(halfmodule_index).at(gain_mode);
    }
    auto exposure_time() const {
        return _exposure_time;
    }

  private:
    std::filesystem::path _path;
    ModuleMode _module_mode;
    std::map<size_t, std::map<uint8_t, Array2D<pedestal_t>>> _modules;
    float _exposure_time;
};

class GainData {
    typedef double gain_t;

  public:
    GainData(std::filesystem::path path, Detector detector);

  private:
    std::filesystem::path _path;
    std::map<size_t, std::map<uint8_t, Array2D<gain_t>>> _modules;
};
