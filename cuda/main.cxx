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
    parser.add_argument("--detector").default_value(JF1M);
    auto correct_parser = argparse::ArgumentParser("correct");
    correct_parser.add_argument("SOURCES")
        .help("Raw data files to run corrections on")
        .nargs(argparse::nargs_pattern::at_least_one);
    parser.add_subparser(correct_parser);

    auto live_parser = argparse::ArgumentParser("live");
    live_parser.add_argument("LISTENERS")
        .help("How many listeners to run (e.g. how many PUB sources)")
        .scan<'i', decltype(Arguments::zmq_listeners)>();
    live_parser.add_argument("--zmq-port")
        .help(
            "The first port to listen to. Automatically incremented by 1 for each "
            "listener.")
        .scan<'i', decltype(Arguments::zmq_port)>()
        .default_value(30001);
    live_parser.add_argument("--zmq-host")
        .help("The ZMQ host to connect to")
        .default_value(std::string{"0.0.0.0"});

    parser.add_subparser(live_parser);

    auto cuargs = parser.parse_args(argc, argv);
    Arguments args = {.verbose = cuargs.verbose,
                      .cuda_device_index = cuargs.device_index,
                      .detector = parser.get<std::string>("--detector"),
                      .cuda_device_signature = fmt::format(
                          "{} (CUDA {}.{})",
                          fmt::styled(cuargs.device.name, fmt::emphasis::bold),
                          cuargs.device.major,
                          cuargs.device.minor),
                      .zmq_host = parser.get<std::string>("--zmq-host"),
                      .zmq_port = parser.get<uint16_t>("--zmq-port")};

    if (!KNOWN_DETECTORS.contains(args.detector)) {
        print("Error: Unknown detector '{}'\n", args.detector);
        std::exit(1);
    }
    if (parser.is_subcommand_used(correct_parser)) {
        args.command = {"correct"};
        args.sources = correct_parser.get<std::vector<std::string>>("SOURCES");
    } else if (parser.is_subcommand_used(live_parser)) {
        args.command = {"live"};
        args.zmq_listeners =
            correct_parser.get<decltype(Arguments::zmq_listeners)>("LISTENERS");
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