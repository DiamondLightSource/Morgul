#pragma once

#include <fmt/base.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/std.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Setting up some common printing styles
namespace style {
static constexpr auto error = fmt::emphasis::bold | fmt::fg(fmt::terminal_color::red);
static constexpr auto warning =
    fmt::emphasis::bold | fmt::fg(fmt::terminal_color::yellow);
static constexpr auto path = fmt::fg(fmt::terminal_color::magenta);
static constexpr auto number = fmt::emphasis::bold | fmt::fg(fmt::terminal_color::cyan);
static constexpr auto url = fmt::emphasis::bold | fmt::fg(fmt::terminal_color::blue);

}  // namespace style

/// Custom format support for utc_clock timepoints - the GCC 13 stdlib
/// does not have a complete enough <chrono> for the built-in fmt:: support.
template <typename T>
struct fmt::formatter<std::chrono::time_point<std::chrono::utc_clock, T>>
    : formatter<string_view> {
    auto format(std::chrono::time_point<std::chrono::utc_clock, T> c,
                format_context &ctx) const -> format_context::iterator {
        std::stringstream ss;
        ss << fmt::format(fmt::runtime("{:%Y-%m-%d %H:%M:%S}"), c);
        return formatter<string_view>::format(ss.str(), ctx);
    }
};

/// Draw a subset of the pixel values for a 2D image array
/// fast, slow, width, height - describe the bounding box to draw
/// data_width, data_height - describe the full data array size
template <typename T>
void draw_image_data(const T *data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    std::string format_type = "";
    if constexpr (std::is_integral<T>::value) {
        format_type = "d";
    } else {
        format_type = ".1f";
    }

    // Walk over the data and get various metadata for generation
    // Maximum value
    T accum = 0;
    // Maximum format width for each column
    std::vector<int> col_widths;
    for (int col = fast; col < fast + width; ++col) {
        size_t maxw = fmt::formatted_size("{:3}", col);
        for (int row = slow; row < std::min(slow + height, data_height); ++row) {
            auto val = data[col + data_width * row];
            auto fmt_spec = fmt::format("{{:{}}}", format_type);
            maxw = std::max(maxw, fmt::formatted_size(fmt::runtime(fmt_spec), val));
            accum = std::max(accum, val);
        }
        col_widths.push_back(maxw);
    }
    bool is_top = slow == 0;
    bool is_left = fast == 0;
    bool is_right = fast + width >= data_width;

    // Draw a row header
    fmt::print("x =       ");
    for (int i = 0; i < width; ++i) {
        auto x = i + fast;
        fmt::print("{:{}} ", x, col_widths[i]);
    }
    fmt::print("\n         ");
    if (is_top) {
        if (is_left) {
            fmt::print("╔");
        } else {
            fmt::print("╒");
        }
    } else {
        if (is_left) {
            fmt::print("╓");
        } else {
            fmt::print("┌");
        }
    }

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < col_widths[i]; ++j) {
            fmt::print("{}", is_top ? "═" : "─");
        }
        fmt::print("{}", is_top ? "═" : "─");
    }
    if (is_top) {
        if (is_right) {
            fmt::print("╗");
        } else {
            fmt::print("╕");
        }
    } else {
        if (is_right) {
            fmt::print("╖");
        } else {
            fmt::print("┐");
        }
    }
    fmt::print("\n");
    for (int y = slow; y < std::min(slow + height, data_height); ++y) {
        if (y == slow) {
            fmt::print("y = {:4d} {}", y, is_left ? "║" : "│");
        } else {
            fmt::print("    {:4d} {}", y, is_left ? "║" : "│");
        }
        for (int i = fast; i < fast + width; ++i) {
            // Calculate color
            // Black, 232->255, White
            // Range of 24 colors, not including white. Split into 25 bins, so
            // that we have a whole black top bin
            auto dat = data[i + data_width * y];
            int color = 255 - ((float)dat / (float)accum) * 24;
            if (color <= 231) color = 0;
            if (accum == 0) color = 255;
            // Avoid type comparison warnings when operating on unsigned
            if constexpr (std::is_signed<decltype(dat)>::value) {
                if (dat < 0) {
                    color = 9;
                }
            }
            if (dat == accum && accum != 0) {
                fmt::print("\033[0m\033[1m");
            } else {
                fmt::print("\033[38;5;{}m", color);
            }
            auto fmt_spec =
                fmt::format("{{:{}{}}} ", col_widths[i - fast], format_type);
            fmt::print(fmt::runtime(fmt_spec), dat);
            if (dat == accum) {
                fmt::print("\033[0m");
            }
        }
        fmt::print("\033[0m{}\n", is_right ? "║" : "│");
    }
}
template <typename T, typename U>
void draw_image_data(const std::unique_ptr<T, U> &data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    draw_image_data(static_cast<T *>(data.get()),
                    fast,
                    slow,
                    width,
                    height,
                    data_width,
                    data_height);
}
template <typename T>
void draw_image_data(const std::span<T> data,
                     size_t fast,
                     size_t slow,
                     size_t width,
                     size_t height,
                     size_t data_width,
                     size_t data_height) {
    draw_image_data(data.data(), fast, slow, width, height, data_width, data_height);
}

template <typename T = uint8_t>
auto GBps(float time_ms, size_t number_objects) -> float {
    return 1000 * number_objects * sizeof(T) / time_ms / 1e9;
}

template <typename T, typename U>
bool compare_results(const T *left,
                     const size_t left_pitch,
                     const U *right,
                     const size_t right_pitch,
                     std::size_t width,
                     std::size_t height,
                     size_t *mismatch_x = nullptr,
                     size_t *mismatch_y = nullptr) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            T lval = left[left_pitch * y + x];
            U rval = right[right_pitch * y + x];
            if (lval != rval) {
                if (mismatch_x != nullptr) {
                    *mismatch_x = x;
                }
                if (mismatch_y != nullptr) {
                    *mismatch_y = y;
                }
                fmt::print("First mismatch at ({}, {}), Left {} != {} Right\n",
                           x,
                           y,
                           (int)lval,
                           (int)rval);
                return false;
            }
        }
    }
    return true;
}

template <typename T, typename I, typename I2 = size_t>
auto count_nonzero(const T *data, I width, I height, I2 pitch = 0) -> size_t {
    if (pitch == 0) pitch = width;
    size_t strong = 0;
    for (size_t row = 0; row < height; ++row) {
        for (size_t col = 0; col < width; ++col) {
            if (data[row * pitch + col]) {
                strong += 1;
            }
        }
    }
    return strong;
}
template <typename T, typename I, typename I2 = size_t>
auto count_nonzero(const std::span<const T> data, I width, I height, I2 pitch = 0)
    -> size_t {
    return count_nonzero(data.data(), width, height, pitch);
}
