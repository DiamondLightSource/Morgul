#include <fmt/core.h>
#include <fmt/ostream.h>

#include <argparse/argparse.hpp>

#include "commands.hpp"
#include "constants.hpp"
#include "cuda_common.hpp"

using namespace fmt;

/// Parse the CLI arguments into an easily-passable arguments object
auto do_argument_parsing(int argc, char **argv) -> Arguments {
    Arguments args;

    auto parser = CUDAArgumentParser();
    parser.add_argument("--detector").default_value(JF1M).store_into(args.detector);
    auto correct_parser = argparse::ArgumentParser("correct");
    correct_parser.add_argument("SOURCES")
        .help("Raw data files to run corrections on")
        .nargs(argparse::nargs_pattern::at_least_one)
        .store_into(args.sources);
    parser.add_subparser(correct_parser);

    auto live_parser = argparse::ArgumentParser("live");
    live_parser.add_argument("LISTENERS")
        .help("How many listeners to run (e.g. how many PUB sources)")
        .store_into(args.zmq_listeners);
    live_parser.add_argument("--zmq-port")
        .help(
            "The first port to listen to. Automatically incremented by 1 for each "
            "listener.")
        .default_value(static_cast<decltype(Arguments::zmq_port)>(30001))
        .store_into(args.zmq_port)
        .metavar("NUM");
    live_parser.add_argument("--zmq-host")
        .help("The ZMQ host to connect to")
        .default_value(std::string{"0.0.0.0"})
        .store_into(args.zmq_host);
    live_parser.add_argument("--zmq-timeout")
        .help(
            "The time (in milliseconds) to wait for more frames in the current "
            "acquisition before giving up. 0 means wait forever.")
        .default_value(static_cast<decltype(args.zmq_timeout)>(10000))
        .store_into(args.zmq_timeout)
        .metavar("MS");

    parser.add_subparser(live_parser);

    auto cuargs = parser.parse_args(argc, argv);
    args.verbose = cuargs.verbose;
    args.cuda_device_index = cuargs.device_index;
    args.cuda_device_signature =
        fmt::format("{} (CUDA {}.{})",
                    fmt::styled(cuargs.device.name, fmt::emphasis::bold),
                    cuargs.device.major,
                    cuargs.device.minor);

    if (!KNOWN_DETECTORS.contains(args.detector)) {
        print("Error: Unknown detector '{}'\n", args.detector);
        std::exit(1);
    }
    if (parser.is_subcommand_used(correct_parser)) {
        args.command = {"correct"};
    } else if (parser.is_subcommand_used(live_parser)) {
        args.command = {"live"};
    } else {
        print("{}\n", parser.help().str());
        std::exit(1);
    }

    return args;
}

int main(int argc, char **argv) {
    auto args = do_argument_parsing(argc, argv);
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
    if (args.command == "correct") {
        do_correct(args);
    } else if (args.command == "live") {
        do_live(args);
    }

    return 0;
}