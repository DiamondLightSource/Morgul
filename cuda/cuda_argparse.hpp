#pragma once

#include <cuda_runtime.h>
#include <fmt/color.h>
#include <fmt/core.h>

#include <argparse/argparse.hpp>

#include "cuda_common.hpp"

class CUDAArgumentParser;

struct CUDAArguments {
  public:
    bool verbose = false;

    int device_index = 0;
    std::optional<size_t> image_number;

    cudaDeviceProp device;

  private:
    friend class CUDAArgumentParser;
};

class CUDAArgumentParser : public argparse::ArgumentParser {
  public:
    CUDAArgumentParser(std::string version = "0.1.0")
        : ArgumentParser("", version, argparse::default_arguments::help) {
        this->add_argument("-v", "--verbose")
            .help("Verbose output")
            .implicit_value(false)
            .action([&](const std::string &value) { _arguments.verbose = true; });

        this->add_argument("-d", "--device")
            .help("Index of the CUDA device device to target.")
            .default_value(0)
            .metavar("INDEX")
            .action([&](const std::string &value) {
                _arguments.device_index = std::stoi(value);
                return _arguments.device_index;
            });
        this->add_argument("--list-devices")
            .help("List the order of CUDA devices, then quit.")
            .implicit_value(false)
            .action([](const std::string &value) {
                int deviceCount;
                if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
                    fmt::print("\033[1;31mError: Could not get GPU count ({})\033[0m\n",
                               cudaGetErrorString(cudaGetLastError()));
                    std::exit(1);
                }

                fmt::print("System GPUs:\n");
                for (int device = 0; device < deviceCount; ++device) {
                    cudaDeviceProp deviceProp;
                    cudaGetDeviceProperties(&deviceProp, device);
                    fmt::print("  {:2d}: {} (PCI {}:{}:{}, CUDA {}.{})\n",
                               device,
                               fmt::styled(deviceProp.name, fmt::emphasis::bold),
                               deviceProp.pciDomainID,
                               deviceProp.pciBusID,
                               deviceProp.pciDeviceID,
                               deviceProp.major,
                               deviceProp.minor);
                }

                std::exit(0);
            });
    }

    auto parse_args(int argc, char **argv) -> CUDAArguments {
        // Convert these to std::string
        std::vector<std::string> args{argv, argv + argc};
        // Look for a "common.args" file in the current folder. If
        // present, add each line as an argument.
        std::ifstream common("common.args");
        std::filesystem::path argfile{"common.args"};
        if (std::filesystem::exists(argfile)) {
            fmt::print("File {} exists, loading default args:\n",
                       fmt::styled(argfile.string(), fmt::emphasis::bold));
            std::fstream f{argfile};
            std::string arg;
            while (std::getline(f, arg)) {
                // Make sure this argument isn't already set
                // if(std::find(vector.begin(), vector.end(), item)!=vector.end()){
                // Found the item
                if (std::find(args.begin(), args.end(), arg) != args.end()) {
                    continue;
                }
                if (arg.size() > 0) {
                    fmt::print("    {}\n", arg);
                    args.push_back(arg);
                }
            }
        }

        try {
            ArgumentParser::parse_args(args);
        } catch (std::runtime_error &e) {
            fmt::print(
                "{}: {}\n{}\n",
                fmt::styled("Error",
                            fmt::emphasis::bold | fmt::fg(fmt::terminal_color::red)),
                fmt::styled(e.what(), fmt::fg(fmt::terminal_color::red)),
                ArgumentParser::usage());
            std::exit(1);
        }

        // cudaDeviceProp deviceProp;
        if (cudaSetDevice(_arguments.device_index) != cudaSuccess) {
            fmt::print(
                "\033[1;31m{}\033[0m\033[31m: Could not select device ({})\033[0m\n",
                "Error",
                cuda_error_string(cudaGetLastError()));
            std::exit(1);
        }
        if (cudaGetDeviceProperties(&_arguments.device, _arguments.device_index)
            != cudaSuccess) {
            fmt::print(fmt::fg(fmt::terminal_color::red),
                       "{}: Could not inspect GPU ({})\n",
                       fmt::styled("Error", fmt::emphasis::bold),
                       cuda_error_string(cudaGetLastError()));
            std::exit(1);
        }
        // fmt::print("Using {} (CUDA {}.{})\n\n",
        //            fmt::styled(_arguments.device.name, fmt::emphasis::bold),
        //            _arguments.device.major,
        //            _arguments.device.minor);

        // If we activated h5read, then handle hdf5 verbosity
        // if (_activated_h5read && !_arguments.verbose) {
        //     _hdf5::H5Eset_auto((_hdf5::hid_t)0, NULL, NULL);
        // }

        return _arguments;
    }

    auto cuda_arguments() {
        return _arguments;
    }
    // void add_h5read_arguments() {
    //     bool implicit_sample = std::getenv("H5READ_IMPLICIT_SAMPLE") != NULL;

    //     auto &group = add_mutually_exclusive_group(!implicit_sample);
    //     group.add_argument("--sample")
    //       .help(
    //         "Don't load a data file, instead use generated test data. If "
    //         "H5READ_IMPLICIT_SAMPLE is set, then this is assumed, if a file is
    //         not " "provided.")
    //       .implicit_value(true);
    //     group.add_argument("file")
    //       .metavar("FILE.nxs")
    //       .help("Path to the Nexus file to parse")
    //       .action([&](const std::string &value) { _arguments.file = value; });
    //     _activated_h5read = true;
    // }

  private:
    CUDAArguments _arguments{};
};
