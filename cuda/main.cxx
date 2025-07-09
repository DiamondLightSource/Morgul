#include <fmt/core.h>
#include <fmt/ostream.h>

#include <argparse/argparse.hpp>

#include "commands.hpp"
#include "constants.hpp"
#include "cuda_argparse.hpp"
#include "cuda_common.hpp"

using namespace fmt;
using namespace std::literals::string_literals;

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
            "The first port to connect to. Automatically incremented by 1 for each "
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
    live_parser.add_argument("--zmq-send-port")
        .help(
            "The ZMQ base port to send corrected data onwards, one per listener. Binds "
            "to tcp://0.0.0.0/ as a PUSH socket.")
        .default_value(static_cast<decltype(Arguments::zmq_send_port)>(31001))
        .store_into(args.zmq_send_port)
        .metavar("NUM");
    live_parser.add_argument("--no-progress")
        .help("Don't show progress messages or spinners")
        .flag()
        .store_into(args.no_progress);
    live_parser.add_argument("--no-require-pedestals")
        .help("Allows 'correcting' data without pedestals, for testing purposes.")
        .default_value(true)
        .implicit_value(false)
        .store_into(args.require_pedestals);
    parser.add_subparser(live_parser);

    auto pedestal_parser = argparse::ArgumentParser("pedestal");
    pedestal_parser.add_argument("SOURCES")
        .help("Raw data files to use as the source of pedestal data")
        .nargs(argparse::nargs_pattern::at_least_one)
        .store_into(args.sources);
    pedestal_parser.add_argument("-l", "--loops")
        .help("The number of loops per gain mode")
        .default_value(static_cast<size_t>(200))
        .store_into(args.pedestal.loops)
        .metavar("NUM");
    pedestal_parser.add_argument("-f", "--frames")
        .help(
            "The number of frames in a single loop. The last frame will be in the "
            "special gain mode.")
        .default_value(static_cast<size_t>(20))
        .store_into(args.pedestal.frames)
        .metavar("NUM");
    pedestal_parser.add_argument("-o")
        .help("Output file to write pedestal tables to")
        .default_value("pedestal.h5"s)
        .store_into(args.pedestal.output_filename)
        .metavar("FILENAME");

    parser.add_subparser(pedestal_parser);

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
    } else if (parser.is_subcommand_used(pedestal_parser)) {
        args.command = {"pedestal"};
    } else {
        print("{}\n", parser.help().str());
        std::exit(1);
    }
    // Don't know why this readout isn't working via store_into
    args.require_pedestals = live_parser.get<bool>("--no-require-pedestals");

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
    } else if (args.command == "pedestal") {
        do_pedestal(args);
    }

    return 0;
}