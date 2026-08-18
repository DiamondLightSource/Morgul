#pragma once

#include <atomic>
#include <chrono>
#include <optional>
#include <string>

/// Fetch and update an atomic float with the maximum of two values
inline float atomic_fetch_max(std::atomic<float> &obj, float arg) noexcept;

/// Get an environment variable if present, with optional default
auto getenv_or(std::string name, std::optional<std::string> _default = std::nullopt)
    -> std::optional<std::string>;

/// @brief Read the value of an EPICS PV, using the caget command-line tool.
///
/// @param pv       The PV to read. This is passed through a shell, so must not
///                 be built out of untrusted input.
/// @param timeout  How long, in seconds, to let caget wait for a reply.
/// @returns The value of the PV, or nullopt if it could not be read.
auto caget(const std::string &pv, double timeout = 2.0) -> std::optional<std::string>;

auto rfc3339_now() -> std::string;

auto get_thread_name() -> std::string;
auto thread_id_str() -> std::string;
auto log_prefix() -> std::string;
/// Print a spinner animation, then return the number of characters printed
int spinner(const std::string_view &message);

class Timer {
  public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    auto get_elapsed_seconds() -> double {
        std::chrono::duration<double, std::milli> dt =
            std::chrono::high_resolution_clock::now() - start;
        return dt.count() / 1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point start;
};

auto read_boolish(std::string value) -> bool;