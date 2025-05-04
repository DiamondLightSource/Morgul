
#include "commands.hpp"
#include "common.hpp"

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
}
