
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

#include "calibration.hpp"
#include "commands.hpp"
#include "common.hpp"
#include "constants.hpp"

using namespace fmt;

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
              styled("No Mask Data, proceeding without", style::warning));
    }
    print("Using Pedestal: {}\n", fmt::styled(cal.pedestal, style::path));
    print("Using Gains:    {}\n", fmt::styled(cal.gain, style::path));

    // Read pedestal data into memory
    auto pedestal_data = PedestalData{cal.pedestal, args.detector};
    auto gain_data = GainData(cal.gain, args.detector);
}
