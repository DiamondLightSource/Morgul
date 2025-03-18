#pragma once

#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

#include "array2d.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"

struct CalibrationDataPath {
    std::filesystem::path pedestal;
    std::optional<std::filesystem::path> mask;
    std::filesystem::path gain;
};
auto get_applicable_calibration_paths(
    float exposure_time,
    std::chrono::utc_time<std::chrono::seconds> timestamp) -> CalibrationDataPath;

enum class ModuleMode {
    FULL,
    HALF,
};
auto module_mode_from(std::string_view value) -> ModuleMode;

class PedestalData {
  public:
    using pedestal_t = float;
    using GainModePointers =
        std::array<shared_device_ptr<pedestal_t[]>, GAIN_MODES.size()>;
    PedestalData(std::filesystem::path path, Detector detector);
    auto get_pedestal(size_t halfmodule_index, uint8_t gain_mode) const
        -> const Array2D<pedestal_t>& {
        return _modules.at(halfmodule_index).at(gain_mode);
    }
    auto exposure_time() const {
        return _exposure_time;
    }

    void upload();

    auto get_gpu_ptrs(size_t hmi) const {
        assert(_gpu_data);
        return _gpu_modules.at(hmi);
    }

  private:
    std::optional<size_t> _gpu_pitch;
    std::filesystem::path _path;
    ModuleMode _module_mode;
    std::map<size_t, std::map<uint8_t, Array2D<pedestal_t>>> _modules;
    std::map<size_t, GainModePointers> _gpu_modules;
    float _exposure_time;
};

class GainData {
  public:
    using gain_t = double;
    using GainModePointers = std::array<shared_device_ptr<gain_t[]>, GAIN_MODES.size()>;

    GainData(std::filesystem::path path, Detector detector);
    void upload();
    auto pitch() {
        assert(_gpu_data);
        return _gpu_pitch;
    }
    auto get_gpu_ptrs(size_t hmi) const {
        assert(_gpu_data);
        return _gpu_modules.at(hmi);
    }

  private:
    std::optional<size_t> _gpu_pitch;
    std::filesystem::path _path;
    std::map<size_t, std::map<uint8_t, Array2D<gain_t>>> _modules;
    std::map<size_t, GainModePointers> _gpu_modules;
};
