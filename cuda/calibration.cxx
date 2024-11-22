#include "calibration.hpp"

#include <date/date.h>
#include <fmt/std.h>
#include <glob.h>
#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <ranges>
#include <zeus/expected.hpp>

#include "common.hpp"
#include "constants.hpp"
using namespace fmt;

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

template <typename T>
auto read_single_hdf5_value(hid_t root_group, std::string path)
    -> expected<T, std::string> {
    auto dataset = H5Cleanup<H5Dclose>(H5Dopen(root_group, path.data(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return unexpected(fmt::format("Invalid HDF5 group: {}", path));
    }
    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    if (dataspace < 0) {
        return unexpected("Could not get data space");
    }
    auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    if (datatype < 0) {
        return unexpected("Could not get data type");
    }
    H5S_class_t dataspace_type = H5Sget_simple_extent_type(dataspace);
    if (dataspace_type != H5S_SCALAR) {
        return unexpected(
            fmt::format("Do not know how to read non-scalar dataset {}/{}",
                        H5Iget_name(dataset).value(),
                        path));
    }

    // Check for basic data type mismatches
    auto dt_class = H5Tget_class(datatype);
    if (dt_class == H5T_INTEGER && !std::is_integral_v<T>) {
        return unexpected("Trying to read integer type into non-integer");
    } else if (dt_class == H5T_FLOAT && !std::is_floating_point_v<T>) {
        return unexpected("Trying to read floating point value into integer");
    }
    if (dt_class != H5T_INTEGER && dt_class != H5T_FLOAT) {
        return unexpected(
            "Unexpected data class type; can only handle integer/non-integer");
    }

    size_t datatype_size = H5Tget_size(datatype);
    auto native_type =
        H5Cleanup<H5Tclose>(H5Tget_native_type(datatype, H5T_DIR_DEFAULT));
    size_t native_size = H5Tget_size(native_type);

    hid_t read_datatype = datatype;
    if (dt_class == H5T_INTEGER) {
        // Validate data type conversions for now. This is a bit annoying
        // but probably safer than just blindly assuming the conversion works.
        if ((native_type == H5T_NATIVE_CHAR || native_type == H5T_NATIVE_SHORT
             || native_type == H5T_NATIVE_INT || native_type == H5T_NATIVE_LONG
             || native_type == H5T_NATIVE_LLONG)
            && !std::is_signed_v<T>) {
            return unexpected("Will not copy signed to unsigned");
        }
        if ((native_type == H5T_NATIVE_UCHAR || native_type == H5T_NATIVE_USHORT
             || native_type == H5T_NATIVE_UINT || native_type == H5T_NATIVE_ULONG
             || native_type == H5T_NATIVE_ULLONG)
            && std::is_signed_v<T>) {
            return unexpected("Will not copy unsigned data into signed");
        }
        if (datatype_size != sizeof(T)) {
            return unexpected(
                fmt::format("Data type size mismatch: Trying to copy size {} into {}",
                            datatype_size,
                            sizeof(T)));
        }
    }

    // If native type double, we want to read that, even if we've been asked
    // for a float by the template instantiator. The caller probably doesn't
    // care if the data is declared as float or double internally.
    typename std::conditional<std::is_same_v<T, float>, double, T>::type output;

    if (H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &output) < 0) {
        throw std::runtime_error("Failed to read dataset");
        return unexpected("Failed to read dataset");
    }

    return output;
}

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

// template<typename T>
// auto format_as(const Array2D<T> &a) { return fmt::format}

template <typename T>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> expected<Array2D<T>, std::string> {
    auto dataset =
        H5Cleanup<H5Dclose>(H5Dopen(root_group, path_to_dataset.data(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return unexpected(fmt::format("Invalid HDF5 group: {}", path_to_dataset));
    }
    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    if (dataspace < 0) {
        return unexpected("Could not get data space");
    }
    auto rank = H5Sget_simple_extent_ndims(dataspace);
    if (rank != 2) {
        return unexpected("Dataset is not 2D");
    }
    // auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    // if (datatype < 0) {
    //     return unexpected("Could not get data type");
    // }
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace, dims, nullptr);

    hid_t hdf5_type;
    if constexpr (std::is_same_v<T, float>) {
        hdf5_type = H5T_NATIVE_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        hdf5_type = H5T_NATIVE_DOUBLE;
    } else {
        static_assert(false, "Unrecognised data type for reading 2D array");
    }
    // std::vector<T> buffer(dims[0] * dims[1]);
    auto data = std::make_unique<T[]>(dims[0] * dims[1]);
    if (H5Dread(dataset, hdf5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.get()) < 0) {
        return unexpected("Failed to read data");
    }
    return Array2D(std::move(data), dims[1], dims[0]);
}

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
    print("Got exposure time: {}\n", _exposure_time);
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
        print("Loading gain map for {} from {}\n", module_name, module_gain_map);
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
