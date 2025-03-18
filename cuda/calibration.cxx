#include "calibration.hpp"

#include <date/date.h>
#include <date/tz.h>
#include <fmt/chrono.h>
#include <fmt/std.h>
#include <glob.h>

#include <chrono>
#include <iostream>
#include <sstream>
// #include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <ranges>
#include <zeus/expected.hpp>

#include "common.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"
#include "hdf5_tools.hpp"

using namespace fmt;

/// Read the calibration log to find the correct calibration data sets
auto get_applicable_calibration_paths(
    float exposure_time,
    std::chrono::utc_time<std::chrono::seconds> timestamp) -> CalibrationDataPath {
    const auto calibration_log = std::getenv("JUNGFRAU_CALIBRATION_LOG");
    if (calibration_log == nullptr) {
        throw std::runtime_error(
            "Can not find calibration data; Please set JUNGFRAU_CALIBRATION_LOG.");
    }

    // const auto ts_latest = std::chrono::sys_time(std::chrono::seconds(timestamp));
    // print("Calibration time point: {}\n",
    //       fmt::styled(ts_latest, fg(fmt::terminal_color::cyan)));
    std::string record_kind, record_timestamp;
    std::filesystem::path record_path;
    float record_exposure;
    auto file = std::ifstream(calibration_log);

    // std::chrono::sys_time<std::chrono::milliseconds> ts;
    // std::chrono::time_point<std::chrono::utc_clock, std::chrono::microseconds> ts;
    std::chrono::sys_time<std::chrono::microseconds> ts_parse;
    std::chrono::utc_time<std::chrono::microseconds> ts;

    using CalibTS = std::tuple<decltype(ts), std::filesystem::path>;
    std::optional<CalibTS> most_recent_pedestal = std::nullopt;
    std::optional<CalibTS> most_recent_mask = std::nullopt;

    int line = 1;
    while (file >> record_kind >> record_timestamp >> record_exposure >> record_path) {
        std::stringstream parse_ss{record_timestamp};
        parse_ss >> date::parse("%FT%T%z", ts_parse);

        if (parse_ss.fail()) {
            throw std::runtime_error(
                fmt::format("Error Reading {}:{}: Failed to parse timestamp '{}'",
                            calibration_log,
                            line,
                            record_timestamp));
        }
        // convert this to the UTC clock time, for comparison
        // ts = std::chrono::clock_cast<std::chrono::utc_clock>(ts_parse);
        // WARNING: Disabled for GCC 11 but probably needs conversion
        // ts = std::chrono::utc_clock::from_sys(ts_parse);

        if (ts < timestamp) {
            if (record_kind == "PEDESTAL") {
                if (std::fabs(record_exposure - exposure_time) > 1e-6) {
                    continue;
                }
                if (most_recent_pedestal) {
                    if (ts > std::get<0>(most_recent_pedestal.value())) {
                        most_recent_pedestal = {ts, record_path};
                    }
                } else {
                    most_recent_pedestal = {ts, record_path};
                }
            } else if (record_kind == "MASK") {
                if (most_recent_mask) {
                    if (ts > std::get<0>(most_recent_mask.value())) {
                        most_recent_mask = {ts, record_path};
                    }
                } else {
                    most_recent_mask = {ts, record_path};
                }
            }
        }
        line += 1;
    }
    if (!most_recent_pedestal) {
        throw std::runtime_error(
            "Error: Could not find a matching pedestal calibration");
    }
    // if (!most_recent_mask) {
    //     throw std::runtime_error("Error: Could not find a matching mask calibration");
    // }
    if (timestamp - std::get<0>(most_recent_pedestal.value())
        > std::chrono::hours(24)) {
        print(style::warning,
              "Warning: Calibration time point is over 24 hours older than data. "
              "Continuing, but this might not work well.\n");
    }
    return {
        .pedestal = std::get<1>(most_recent_pedestal.value()),
        // .mask = std::get<1>(most_recent_mask.value()),
        .gain = GAIN_MAPS,
    };
}

auto module_mode_from(std::string_view value) -> ModuleMode {
    if (value == "full") {
        return ModuleMode::FULL;
    } else if (value == "half") {
        return ModuleMode::HALF;
    }
    throw std::runtime_error(
        fmt::format("Got invalid or not understood module mode '{}'", value));
}

using zeus::expected, zeus::unexpected;

template <typename T>
auto draw_image_data(const Array2D<T> &data,
                     size_t x,
                     size_t y,
                     size_t width,
                     size_t height) -> void {
    draw_image_data(data.data(), x, y, width, height, data.stride(), data.height());
}

PedestalData::PedestalData(std::filesystem::path path, Detector detector)
    : _path(path) {
    auto file = H5Cleanup<H5Fclose>(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    if (file == H5I_INVALID_HID) {
        throw std::runtime_error("Failed to open Pedestal file");
    }
    _module_mode = module_mode_from(
        read_single_hdf5_value<std::string>(file, "/module_mode").value());
    _exposure_time = read_single_hdf5_value<float>(file, "/exptime").value();
    print("Reading pedestals for exposure time {} ms from {}\n",
          styled(_exposure_time * 1000, style::number),
          styled(path, style::path));
    auto [n_cols, n_rows] = DETECTOR_SIZE.at(detector);

    // We want to support two forms of pedestal file;
    if (_module_mode == ModuleMode::FULL) {
        // Original Morgul:
        // - <ModuleName>/pedestal_{0,1,2} (1024x512)
        //         const std::map<Detector, std::map<std::string, std::tuple<int, int>>> KNOWN_DETECTORS =
        auto det_modules = KNOWN_DETECTORS.at(detector);
        for (const auto &[module_name, position] : det_modules) {
            auto [mod_col, mod_row] = position;
            uint8_t hmindex = 2 * n_rows * mod_col + 2 * mod_row;

            // Read the data for this module out of the file
            for (auto mode : GAIN_MODES) {
                auto name = fmt::format("{}/pedestal_{}", module_name, mode);
                auto table = read_2d_dataset<pedestal_t>(file, name).value();

                // Let's split this dataset
                auto [top, bottom] = split_module(table);
                _modules[hmindex][mode] = std::move(top);
                _modules[hmindex + 1][mode] = std::move(bottom);
            }
        }
    } else {
        // Halfmodule morgul:
        // - /pedestal_{0,1,2} (1024x512)
        //         const std::map<Detector, std::map<std::string, std::tuple<int, int>>> KNOWN_DETECTORS =
        for (size_t hmi : std::ranges::views::iota(0, n_cols * n_rows * 2)) {
            for (auto gain_mode : GAIN_MODES) {
                auto dataset_name =
                    fmt::format("hmi_{:02d}/pedestal_{}", hmi, gain_mode);

                _modules[hmi][gain_mode] =
                    read_2d_dataset<pedestal_t>(file, dataset_name).value();
            }
        }
    }
}

auto PedestalData::upload() -> void {
    // auto width = std::get<0>(MODULE_SHAPE);
    // auto height = std::get<1>(HALF_MODULE_SHAPE);
    size_t num_modules = _modules.size();

    if (_module_mode != ModuleMode::HALF) {
        print(style::error, "Error: Only support GPU upload on halfmodule pedestals");
        std::exit(1);
    }

    // Copy each halfmodules gain modes into device memory
    for (const auto &[hmi, gain_modes] : _modules) {
        GainModePointers dev;
        for (int i = 0; i < GAIN_MODES.size(); ++i) {
            auto [ptr, pitch] =
                make_cuda_pitched_malloc<pedestal_t>(HM_WIDTH, HM_HEIGHT);
            dev[i] = ptr;
            _gpu_pitch = pitch;

            auto &data = gain_modes.at(GAIN_MODES[i]);
            assert(HM_WIDTH == data.width());
            assert(HM_HEIGHT == data.height());
            assert(1024 == data.stride());
            cudaMemcpy(dev[i], data.data().data(), HM_HEIGHT * HM_WIDTH);
        }
        _gpu_modules[hmi] = dev;
    }
    // We can avoid carrying round pitch if we know the pitched array is unpadded
    if (_gpu_pitch != 1024) {
        print(style::error,
              "Error: Expected module pedestals to have unpadded pitch. Instead have "
              "{}.",
              _gpu_pitch);
        std::exit(1);
    }
}

GainData::GainData(std::filesystem::path path, Detector detector) : _path(path) {
    // Get the module names
    auto modules = KNOWN_DETECTORS.at(detector);
    // auto det_modules = KNOWN_DETECTORS.at(detector);
    constexpr size_t num_module_pixels =
        std::get<0>(MODULE_SHAPE) * std::get<1>(MODULE_SHAPE);
    constexpr size_t num_pixels = num_module_pixels * GAIN_MODES.size();
    auto raw_data = std::vector<gain_t>(num_pixels);

    auto [n_cols, n_rows] = DETECTOR_SIZE.at(detector);

    for (const auto &[module_name, module_position] : modules) {
        auto expected_map = path / fmt::format("{}_fullspeed", module_name) / "*.bin";
        // The actual gain map name may vary; pick up the only .bin file via glob
        glob_t glob_results;
        int glob_ret = glob(expected_map.c_str(), 0, nullptr, &glob_results);
        if (glob_ret != 0) {
            globfree(&glob_results);
            throw std::runtime_error("glob for gain .bin failed");
        }
        if (glob_results.gl_pathc > 1) {
            globfree(&glob_results);
            throw std::runtime_error(
                fmt::format("Got more than one result for {}", expected_map));
        }
        // We have one result, and we want to use it
        std::filesystem::path module_gain_map(glob_results.gl_pathv[0]);
        // Check the file size

        size_t expected_size = num_pixels * sizeof(gain_t);
        if (expected_size > std::filesystem::file_size(module_gain_map)) {
            throw std::runtime_error(fmt::format(
                "Gain map {} on disk are an unexpected size (expected at least {})",
                module_gain_map,
                expected_size));
        }
        globfree(&glob_results);
        print("Loading gain map for {} from: {}\n",
              module_name,
              styled(module_gain_map, style::path));
        auto gainfile = std::ifstream(module_gain_map, std::ios_base::binary);
        if (!gainfile) {
            throw std::runtime_error("Could not open gain map for reading");
        }
        gainfile.read(reinterpret_cast<char *>(raw_data.data()), expected_size);

        auto [mod_col, mod_row] = module_position;
        size_t halfmodule_index = 2 * n_rows * mod_col + 2 * mod_row;

        // Split each of these into six halfmodules
        for (uint8_t gain_mode : std::ranges::views::iota(0, int(GAIN_MODES.size()))) {
            auto gain_top = std::make_unique<gain_t[]>(num_module_pixels / 2);
            auto gain_bot = std::make_unique<gain_t[]>(num_module_pixels / 2);
            std::fill(gain_top.get(), gain_top.get() + num_module_pixels / 2, 666);
            std::fill(gain_bot.get(), gain_bot.get() + num_module_pixels / 2, 666);
            std::copy(raw_data.begin() + num_module_pixels * gain_mode,
                      raw_data.begin() + num_module_pixels * gain_mode
                          + num_module_pixels / 2,
                      gain_top.get());
            std::copy(raw_data.begin() + num_module_pixels * gain_mode
                          + num_module_pixels / 2,
                      raw_data.begin() + num_module_pixels * (gain_mode + 1),
                      gain_bot.get());
            _modules[halfmodule_index][GAIN_MODES[gain_mode]] =
                Array2D(std::move(gain_top),
                        std::get<0>(MODULE_SHAPE),
                        std::get<1>(MODULE_SHAPE) / 2);
            _modules[halfmodule_index + 1][GAIN_MODES[gain_mode]] =
                Array2D(std::move(gain_bot),
                        std::get<0>(MODULE_SHAPE),
                        std::get<1>(MODULE_SHAPE) / 2);
        }
    }
}

auto GainData::upload() -> void {
    size_t num_modules = _modules.size() * GAIN_MODES.size();

    for (const auto &[hmi, gain_modes] : _modules) {
        GainModePointers dev;
        for (int i = 0; i < GAIN_MODES.size(); ++i) {
            auto [ptr, pitch] = make_cuda_pitched_malloc<gain_t>(HM_WIDTH, HM_HEIGHT);
            dev[i] = ptr;
            _gpu_pitch = pitch;

            auto &data = gain_modes.at(GAIN_MODES[i]);
            assert(HM_WIDTH == data.width());
            assert(HM_HEIGHT == data.height());
            assert(1024 == data.stride());
            cudaMemcpy(dev[i], data.data().data(), HM_HEIGHT * HM_WIDTH);
        }
        _gpu_modules[hmi] = dev;
    }

    // We can avoid carrying round pitch if we know the pitched array is unpadded
    if (_gpu_pitch != 1024) {
        print(style::error,
              "Error: Expected module gains to have unpadded pitch. Instead have "
              "{}.",
              _gpu_pitch);
        std::exit(1);
    }
}
