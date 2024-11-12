#include <fmt/core.h>
#include <fmt/ostream.h>

#include <argparse/argparse.hpp>

#include "commands.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"

using namespace fmt;

/// Parse the CLI arguments into an easily-passable arguments object
auto do_argument_parsing(int argc, char **argv) -> Arguments {
    auto parser = CUDAArgumentParser();
    parser.add_argument("--detector").default_value("JF1M");
    auto correct_parser = argparse::ArgumentParser("correct");
    correct_parser.add_argument("SOURCES")
        .help("Raw data files to run corrections on")
        .nargs(argparse::nargs_pattern::at_least_one);

    parser.add_subparser(correct_parser);
    auto cuargs = parser.parse_args(argc, argv);
    Arguments args = {
        .verbose = cuargs.verbose,
        .cuda_device_index = cuargs.device_index,
        .detector = parser.get<std::string>("--detector"),
    };

    if (!KNOWN_DETECTORS.contains(args.detector)) {
        print("Error: Unknown detector '{}'\n", args.detector);
        std::exit(1);
    }
    print(
        " ███▄ ▄███▓ ▒█████   ██▀███    ▄████  █    ██  ██▓    \n"
        "▓██▒▀█▀ ██▒▒██▒  ██▒▓██ ▒ ██▒ ██▒ ▀█▒ ██  ▓██▒▓██▒    \n"
        "▓██    ▓██░▒██░  ██▒▓██ ░▄█ ▒▒██░▄▄▄░▓██  ▒██░▒██░    \n"
        "▒██    ▒██ ▒██   ██░▒██▀▀█▄  ░▓█  ██▓▓▓█  ░██░▒██░    \n"
        "▒██▒   ░██▒░ ████▓▒░░██▓ ▒██▒░▒▓███▀▒▒▒█████▓ ░██████▒\n"
        "░ ▒░   ░  ░░ ▒░▒░▒░ ░ ▒▓ ░▒▓░ ░▒   ▒ ░▒▓▒ ▒ ▒ ░ ▒░▓  ░\n"
        "░  ░      ░  ░ ▒ ▒░   ░▒ ░ ▒░  ░   ░ ░░▒░ ░ ░ ░ ░ ▒  ░\n"
        "░      ░   ░ ░ ░ ▒    ░░   ░ ░ ░   ░  ░░░ ░ ░   ░ ░   \n"
        "       ░       ░ ░     ░           ░    ░         ░  ░\n");
    if (parser.is_subcommand_used(correct_parser)) {
        args.command = {"correct"};
        args.sources = correct_parser.get<std::vector<std::string>>("SOURCES");
    } else {
        print("{}\n", parser.help().str());
        std::exit(1);
    }

    return args;
}

int main(int argc, char **argv) {
    auto args = do_argument_parsing(argc, argv);

    if (args.command == "correct") {
        do_correct(args);
    }

    return 0;
}