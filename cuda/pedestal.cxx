
#include <filesystem>

#include "commands.hpp"
#include "common.hpp"
#include "hdf5_tools.hpp"

using namespace fmt;

auto do_pedestal(Arguments &args) -> void {
    print(
        "                        __          __        __\n"
        "        ____  ___  ____/ /__  _____/ /_____ _/ /\n"
        "       / __ \\/ _ \\/ __  / _ \\/ ___/ __/ __ `/ /\n"
        "      / /_/ /  __/ /_/ /  __(__  ) /_/ /_/ / /\n"
        "     / .___/\\___/\\__,_/\\___/____/\\__/\\__,_/_/\n"
        "    /_/\n\n");
    print("Pedestal frames per loop: {}\n", styled(args.pedestal.loops, style::number));
    print("Pedestal loops per gain:  {}\n",
          styled(args.pedestal.frames, style::number));
    print("Writing output to:        {}\n",
          styled(args.pedestal.output_filename, style::path));

    // First, expand any of the sources that are directories
    for (auto &datafile_path : args.sources) {
        if (std::filesystem::is_directory(datafile_path)) {
            print("Expanding Directory {}\n", styled(datafile_path, style::path));
        } else {
            print("Reading file {}\n", styled(datafile_path, style::path));
        }
    }
}
