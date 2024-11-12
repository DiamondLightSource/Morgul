#include <date/date.h>
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>

#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"

using namespace fmt;

struct CalibrationData {
    std::filesystem::path pedestal;
    std::filesystem::path mask;
    std::filesystem::path gain;
};

auto get_applicable_calibration(float exposure_time, uint64_t timestamp)
    -> CalibrationData {
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
    if (!most_recent_mask) {
        throw std::runtime_error("Error: Could not find a matching mask calibration");
    }
    if (ts_latest - std::get<0>(most_recent_pedestal.value())
        > std::chrono::hours(24)) {
        print(style::warning,
              "Warning: Calibration time point is over 24 hours older than data. "
              "Continuing, but this might not work well.\n");
    }
    return {
        .pedestal = std::get<1>(most_recent_pedestal.value()),
        .mask = std::get<1>(most_recent_mask.value()),
        .gain = GAIN_MAPS,
    };
}

auto do_correct(Arguments &args) -> void {
    print(
        "        ____ ___   ____  ____ ___  ____ / /_\n"
        "       / __// _ \\ / __/ / __// -_)/ __// __/\n"
        "       \\__/ \\___//_/   /_/   \\__/ \\__/ \\__/\n");
    auto cal = get_applicable_calibration(0.001, 1731413311);

    // mt::styled(1.23, fmt::fg(fmt::color::green)
    print("Using Mask:     {}\n", fmt::styled(cal.mask, style::path));
    print("Using Pedestal: {}\n", fmt::styled(cal.pedestal, style::path));
    print("Using Gains:    {}\n", fmt::styled(cal.gain, style::path));

    // for (auto &src : args.sources) {
    //     print(" - {}\n", src);
    // }
}
