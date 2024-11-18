#include <date/date.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <hdf5.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <zeus/expected.hpp>

#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"

// namespace py = pybind11;
//
using namespace fmt;
using zeus::expected, zeus::unexpected;

// using namespace H5;

struct CalibrationDataPath {
    std::filesystem::path pedestal;
    std::optional<std::filesystem::path> mask;
    std::filesystem::path gain;
};

/// Read the calibration log to find the correct calibration data sets
auto get_applicable_calibration_paths(float exposure_time, uint64_t timestamp)
    -> CalibrationDataPath {
    const auto calibration_log = std::getenv("JUNGFRAU_CALIBRATION_LOG");
    if (calibration_log == nullptr) {
        throw std::runtime_error(
            "Can not find calibration data; Please set JUNGFRAU_CALIBRATION_LOG.");
    }

    const auto ts_latest = std::chrono::sys_time(std::chrono::seconds(timestamp));
    // print("Calibration time point: {}\n",
    //       fmt::styled(ts_latest, fg(fmt::terminal_color::cyan)));
    std::string record_kind, record_timestamp;
    std::filesystem::path record_path;
    float record_exposure;
    auto file = std::ifstream(calibration_log);

    // std::chrono::sys_time<std::chrono::milliseconds> ts;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> ts;
    using CalibTS = std::tuple<decltype(ts), std::filesystem::path>;
    std::optional<CalibTS> most_recent_pedestal = std::nullopt;
    std::optional<CalibTS> most_recent_mask = std::nullopt;

    int line = 1;
    while (file >> record_kind >> record_timestamp >> record_exposure >> record_path) {
        std::stringstream parse_ss{record_timestamp};
        parse_ss >> date::parse("%FT%T%z", ts);
        if (parse_ss.fail()) {
            throw std::runtime_error(
                fmt::format("Error Reading {}:{}: Failed to parse timestamp '{}'",
                            calibration_log,
                            line,
                            record_timestamp));
        }
        if (ts < ts_latest) {
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
    if (ts_latest - std::get<0>(most_recent_pedestal.value())
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

auto H5Iget_name(hid_t identifier) -> std::optional<std::string> {
    ssize_t name_size = H5Iget_name(identifier, NULL, 0);
    if (name_size == 0) {
        return std::nullopt;
    }
    std::string name;
    name.reserve(name_size + 1);
    H5Iget_name(identifier, name.data(), name_size + 1);
    return name;
}

template <typename T>
auto read_single_hdf5_value(hid_t root_group, std::string path)
    -> expected<T, std::string>;
//     {
//     hid_t dataset;
//     if ((dataset = H5Dopen(root_group, path.c_str(), H5P_DEFAULT)) == H5I_INVALID_HID) {
//         throw std::runtime_error(fmt::format("Invalid HDF5 group: {}", path));
//     }
//     hid_t datatype = H5Dget_type(dataset);
//     size_t datatype_size = H5Tget_size(datatype);
//     hid_t dataspace = H5Dget_space(dataset);
//     size_t num_elements = H5Sget_simple_extent_npoints(dataspace);

//     if (num_elements > 1) {
//         return unexpected(fmt::format("More than one element reading {}/{}",
//                                       H5Iget_name(dataset).value()));
//     } else {
//     }
//     H5Dclose(dataset);
// }

/// Convenience class to ensure an HDF5 closing routine is called properly
template <herr_t(D)(hid_t)>
struct H5Cleanup {
    H5Cleanup(hid_t id) : id(id) {}
    ~H5Cleanup() {
        if (id >= 0) {
            D(id);
        }
    }
    operator hid_t() const {
        return id;
    }
    hid_t id;
};

template <>
auto read_single_hdf5_value(hid_t root_group, const std::string path)
    -> expected<std::string, std::string> {
    auto dataset = H5Cleanup<H5Dclose>(H5Dopen(root_group, path.c_str(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return unexpected(fmt::format("Invalid HDF5 group: {}", path));
    }
    auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    if (H5Tget_class(datatype) != H5T_STRING) {
        return unexpected("Dataset type class is not string!");
    }

    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    H5S_class_t dataspace_type = H5Sget_simple_extent_type(dataspace);
    if (dataspace_type != H5S_SCALAR) {
        return unexpected(
            fmt::format("Do not know how to read non-scalar dataset {}/{}",
                        H5Iget_name(dataset).value(),
                        path));
    }
    bool is_var = H5Tis_variable_str(datatype);
    if (!is_var) {
        return unexpected("Unhandled fixed-length string");
    }
    char *buffer = nullptr;

    if (H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer) < 0) {
        return unexpected("Failed to read");
    }
    std::string output{buffer};
    // We need to explicitly reclaim the automatically allocated space
    if (H5Treclaim(datatype, dataspace, H5P_DEFAULT, &buffer) < 0) {
        return unexpected("Failed to reclaim");
    }

    return output;
}

template <typename T>
class Array2D {
  public:
    Array2D(std::unique_ptr<T> data,
            size_t width,
            size_t height,
            size_t stride,
            bool ragged = false)
        : _width(width), _height(height), _stride(stride), _data(data) {}

    auto data() -> std::span<T> {
        return std::span<T>(_data.get(), _stride * _height);
    }

  private:
    size_t _width, _height, _stride;
    std::unique_ptr<T> _data;
};

template <typename T>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> expected<Array2D<T>, std::string> {
    auto dataset =
        H5Cleanup<H5Dclose>(H5Dopen(root_group, path_to_dataset.data(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return unexpected(fmt::format("Invalid HDF5 group: {}", path_to_dataset));
    }
    auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    if (datatype < 0) {
        return unexpected("Could not get data type");
    }
    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    if (dataspace < 0) {
        return unexpected("Could not get data space");
    }
}

enum class ModuleMode {
    FULL,
    HALF,
};
auto module_mode_from(std::string_view value) -> ModuleMode {
    if (value == "full") {
        return ModuleMode::FULL;
    } else if (value == "half") {
        return ModuleMode::HALF;
    }
    throw std::runtime_error(
        fmt::format("Got invalid or not understood module mode '{}'", value));
}

class PedestalData {
  public:
    PedestalData(std::filesystem::path path, Detector detector) : _path(path) {
        // auto file = H5File(path, H5F_ACC_RDONLY);
        auto file =
            H5Cleanup<H5Fclose>(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
        if (file == H5I_INVALID_HID) {
            throw std::runtime_error("Failed to open Pedestal file");
        }
        _module_mode = module_mode_from(
            read_single_hdf5_value<std::string>(file, "/module_mode").value());

        print("Module mode: {}\n", _module_mode == ModuleMode::FULL ? "Full" : "Half");

        // We want to support two forms of pedestal file;
        // HMI Morgul
        // - HMI_ID/pedestal_{0,1,2} (1024x256)
        auto [n_cols, n_rows] = DETECTOR_SIZE.at(detector);
        if (_module_mode == ModuleMode::FULL) {
            // Original Morgul:
            // - <ModuleName>/pedestal_{0,1,2} (1024x512)
            //         const std::map<Detector, std::map<std::string, std::tuple<int, int>>> KNOWN_DETECTORS =
            // {{JF1M, {{"M420", {0, 0}}, {"M418", {0, 1}}}}};
            auto det_modules = KNOWN_DETECTORS.at(detector);
            for (const auto &[module_name, position] : det_modules) {
                auto [mod_col, mod_row] = position;
                uint8_t index = n_rows * mod_col + mod_row;
                print("Module {}(i={})\n", module_name, index);

                // Read the data for this module out of the file
                for (auto mode : GAIN_MODES) {
                    auto name = fmt::format("{}/pedestal_{}", module_name, mode);
                    auto table = read_2d_dataset<double>(file, name);
                }
            }
        } else {
            throw std::runtime_error("Halfmodules not handled");
        }
        // return "Some";
    }

  private:
    std::filesystem::path _path;
    ModuleMode _module_mode;
};

auto do_correct(Arguments &args) -> void {
    print(
        "        ____ ___   ____  ____ ___  ____ / /_\n"
        "       / __// _ \\ / __/ / __// -_)/ __// __/\n"
        "       \\__/ \\___//_/   /_/   \\__/ \\__/ \\__/\n");

    // Open the data files to work out what calibration we need

    auto cal = get_applicable_calibration_paths(0.001, 1731413311);

    if (cal.mask) {
        print("Using Mask:     {}\n", fmt::styled(cal.mask, style::path));
    } else {
        print("Using Mask:     {}\n",
              styled("No Mask Data, temporarily accepting", style::error));
    }
    print("Using Pedestal: {}\n", fmt::styled(cal.pedestal, style::path));
    print("Using Gains:    {}\n", fmt::styled(cal.gain, style::path));

    // Read pedestal data into memory
    auto pedestal_data = PedestalData{cal.pedestal, args.detector};

    // for (auto &src : args.sources) {
    //     print(" - {}\n", src);
    // }
}
