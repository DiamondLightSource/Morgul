
// #include <fmt/color.h>
// #include <fmt/core.h>
// #include <fmt/format.h>
// #include <fmt/ranges.h>

// #include <algorithm>
// #include <cstdlib>
// #include <filesystem>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <memory>
// #include <optional>
// #include <sstream>
// #include <string_view>
// #include <type_traits>

#include <chrono>

#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "hdf5_tools.hpp"

using fmt::print;
using std::chrono::sys_time, std::chrono::milliseconds;
using std::filesystem::path;

class DataFile {
  public:
    typedef std::chrono::time_point<std::chrono::system_clock,
                                    std::chrono::duration<double>>
        timestamp_t;

    DataFile(const path &path) {
        file = H5Cleanup<H5Fclose>(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
        _exposure_time = read_single_hdf5_value<double>(file, "exptime").value();
        _timestamp = timestamp_t(std::chrono::duration<double>(
            read_single_hdf5_value<double>(file, "timestamp").value()));
    }

    auto exposure_time() {
        return _exposure_time;
    }
    auto timestamp() {
        return _timestamp;
    }

  private:
    H5Cleanup<H5Fclose> file;
    double _exposure_time;
    timestamp_t _timestamp;
};
auto format_frequency(double value_hz) {
    if (value_hz > 999) {
        return fmt::format("{:.1f} KHz", styled(value_hz / 1000, style::number));
    } else {
        return fmt::format("{:.0f} Hz", styled(value_hz, style::number));
    }
}
auto do_correct(Arguments &args) -> void {
    print(
        "        ____ ___   ____  ____ ___  ____ / /_\n"
        "       / __// _ \\ / __/ / __// -_)/ __// __/\n"
        "       \\__/ \\___//_/   /_/   \\__/ \\__/ \\__/\n\n");

    print("Using {}\n", args.cuda_device_signature);

    // Open the data files to work out what calibration we need
    std::optional<float> common_exposure_time;
    std::optional<DataFile::timestamp_t> common_timestamp;

    for (auto &datafile_path : args.sources) {
        print("Reading source {}\n", styled(datafile_path, style::path));
        auto file = DataFile(datafile_path);
        if (!common_exposure_time) {
            common_exposure_time = file.exposure_time();
            common_timestamp = file.timestamp();

        } else {
            if (file.exposure_time() - common_exposure_time.value() > 1e-9) {
                throw std::runtime_error(fmt::format(
                    "Error: Mismatched exposure times; got both {} s and {} s",
                    common_exposure_time.value(),
                    file.exposure_time()));
            }
            if (file.timestamp() != common_timestamp.value()) {
                throw std::runtime_error(fmt::format(
                    "Error: Mismatched collection timestamps; got both {} s and {} s",
                    common_exposure_time.value(),
                    file.exposure_time()));
            }
        }
    }

    print("Common exposure time: {} s ({})\n",
          styled(common_exposure_time.value(), style::number),
          format_frequency(1.0 / common_exposure_time.value()));

    print("Common timestamp:     {}\n",
          styled(common_timestamp.value(), style::number));
    auto cal = get_applicable_calibration_paths(
        common_exposure_time.value(),
        std::chrono::time_point_cast<std::chrono::seconds>(common_timestamp.value()));
    // auto cal = get_applicable_calibration_paths(0.001, 1731413311);

    if (cal.mask) {
        print("Using Mask:     {}\n", fmt::styled(cal.mask, style::path));
    } else {
        print("Using Mask:     {}\n",
              styled("No Mask Data, proceeding without", style::warning));
    }
    print("Using Pedestal: {}\n", fmt::styled(cal.pedestal, style::path));
    print("Using Gains:    {}\n", fmt::styled(cal.gain, style::path));

    // Read pedestal data into memory
    auto pedestal_data = PedestalData{cal.pedestal, args.detector};
    auto gain_data = GainData(cal.gain, args.detector);
}
