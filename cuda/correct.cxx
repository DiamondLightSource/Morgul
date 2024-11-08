#include <fmt/core.h>

#include "commands.hpp"

using namespace fmt;

auto do_correct(Arguments &args) -> void {
    print("Running correction parser:\n");
    for (auto &src : args.sources) {
        print(" - {}\n", src);
    }
}
