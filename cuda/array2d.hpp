#pragma once
#include <span>

#include "common.hpp"

template <typename T>
class Array2D {
  public:
    Array2D() : _width(0), _height(0), _stride(0) {}

    Array2D(std::unique_ptr<T[]> data, size_t width, size_t height)
        : _width(width), _height(height), _stride(width), _data(std::move(data)) {}

    Array2D(std::unique_ptr<T[]> data, size_t width, size_t height, size_t stride)
        : _width(width), _height(height), _stride(stride), _data(std::move(data)) {}

    auto data() const -> std::span<T> {
        return std::span<T>(_data.get(), _stride * _height);
    }
    auto width() const {
        return _width;
    }
    auto height() const {
        return _height;
    }
    auto stride() const {
        return _stride;
    }

    auto type() const {
        if constexpr (std::is_same_v<T, float>) {
            return "float";
        } else if constexpr (std::is_same_v<T, double>) {
            return "double";
        } else {
            return "(please add type)";
        }
    }

  private:
    size_t _width, _height, _stride;
    std::unique_ptr<T[]> _data;
};

template <typename T>
auto split_module(const Array2D<T> &input) -> std::tuple<Array2D<T>, Array2D<T>> {
    if (input.height() != 512 || input.width() != 1024) {
        throw std::runtime_error(
            "Have been asked to split something that is not a standard module");
    }
    auto data_top = std::make_unique<T[]>(input.height() / 2 * input.stride());
    auto data_bottom = std::make_unique<T[]>(input.height() / 2 * input.stride());

    auto input_data = input.data();

    std::copy(input.data().data(),
              input.data().data() + (input.stride() * input.height() / 2),
              data_top.get());
    std::copy(input.data().data() + (input.stride() * input.height() / 2),
              &input.data().back(),
              data_bottom.get());

    return {
        Array2D<T>(
            std::move(data_top), input.width(), input.height() / 2, input.stride()),
        Array2D<T>(
            std::move(data_bottom), input.width(), input.height() / 2, input.stride())};
}

template <typename T>
struct fmt::formatter<Array2D<T>> : formatter<std::string_view> {
    auto format(const Array2D<T> &array, format_context &ctx) const
        -> format_context::iterator {
        std::string name = fmt::format(
            "<Array2D {} x {} {}>", array.width(), array.height(), array.type());
        return formatter<std::string_view>::format(name, ctx);
    }
};