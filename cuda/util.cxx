
#include "util.hpp"

#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <pthread.h>

#include <chrono>
#include <iostream>

using namespace fmt;

/// Fetch and update an atomic float with the maximum of two values
inline float atomic_fetch_max(std::atomic<float> &obj, float arg) noexcept {
    float old = obj.load(std::memory_order_relaxed);
    while (old < arg
           && !obj.compare_exchange_weak(
               old, arg, std::memory_order_release, std::memory_order_relaxed)) {
        // old is updated by compare_exchange_weak automatically
    }
    return old;
}

/// Get an environment variable if present, with optional default
auto getenv_or(std::string name, std::optional<std::string> _default)
    -> std::optional<std::string> {
    auto data = std::getenv(name.c_str());
    if (data == nullptr) {
        return _default;
    }
    return {data};
}

auto rfc3339_now() -> std::string {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto time_t_now = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_now);
    return fmt::format("{}.{:03d}{}",
                       buf,
                       static_cast<int>(ms.count()),
                       tm_now.tm_gmtoff == 0
                           ? "Z"
                           : fmt::format("{:+03d}:{:02d}",
                                         tm_now.tm_gmtoff / 3600,
                                         std::abs((tm_now.tm_gmtoff % 3600) / 60)));
}

auto get_thread_name() -> std::string {
    char name[16];
    pthread_getname_np(pthread_self(), name, 16);
    return std::string(name);
}

auto thread_id_str() -> std::string {
    std::ostringstream oss;
    oss << std::this_thread::get_id();
    return oss.str();
}

auto log_prefix() -> std::string {
    return format("{} {:16}::{}", rfc3339_now(), get_thread_name(), thread_id_str());
}

/// Print a spinner animation, then return the number of characters printed
int spinner(const std::string_view &message) {
    static int index = 0;
    std::vector<std::string> ball = {
        "( ●    )",
        "(  ●   )",
        "(   ●  )",
        "(    ● )",
        "(     ●)",
        "(    ● )",
        "(   ●  )",
        "(  ●   )",
        "( ●    )",
        "(●     )",
    };
    index = (index + 1) % ball.size();
    std::string msg = fmt::format("  {} {}\r", message, ball[index]);
    std::cerr << msg << std::flush;
    return msg.size();
}

auto read_boolish(std::string value) -> bool {
    if (value.size() == 0) {
        return false;
    }
    if (value != "true" && value != "false") {
        throw std::runtime_error(
            fmt::format("Got non-boolish json value: '{}'", value));
    }
    return value == "true";
}
