#pragma once

#include <fmt/core.h>
#include <hdf5.h>

#include <optional>
#include <string>
#include <type_traits>
#include <zeus/expected.hpp>

#include "array2d.hpp"

auto H5Iget_name(hid_t identifier) -> std::optional<std::string>;

/// Convenience class to ensure an HDF5 closing routine is called properly
template <herr_t(D)(hid_t)>
struct H5Cleanup {
    H5Cleanup() : id(-1) {}
    H5Cleanup(hid_t id) : id(id) {}
    H5Cleanup(const H5Cleanup&) = delete;
    H5Cleanup(H5Cleanup&& other) : id(other.id) {
        other.id = -1;
    }
    H5Cleanup& operator=(H5Cleanup&& other) {
        std::swap((*this).id, other.id);
        return *this;
    }
    ~H5Cleanup() {
        if (id >= 0) {
            D(id);
        }
    }
    operator hid_t() const {
        return id;
    }
    hid_t id;
};

template <typename T>
auto read_single_hdf5_value(hid_t root_group, std::string path)
    -> zeus::expected<T, std::string> {
    auto dataset = H5Cleanup<H5Dclose>(H5Dopen(root_group, path.data(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return zeus::unexpected(fmt::format("Invalid HDF5 group: {}", path));
    }
    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    if (dataspace < 0) {
        return zeus::unexpected("Could not get data space");
    }
    auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    if (datatype < 0) {
        return zeus::unexpected("Could not get data type");
    }
    H5S_class_t dataspace_type = H5Sget_simple_extent_type(dataspace);
    if (dataspace_type != H5S_SCALAR) {
        return zeus::unexpected(
            fmt::format("Do not know how to read non-scalar dataset {}/{}",
                        H5Iget_name(dataset).value(),
                        path));
    }

    // Check for basic data type mismatches
    auto dt_class = H5Tget_class(datatype);
    if (dt_class == H5T_INTEGER && !std::is_integral_v<T>) {
        return zeus::unexpected("Trying to read integer type into non-integer");
    } else if (dt_class == H5T_FLOAT && !std::is_floating_point_v<T>) {
        return zeus::unexpected("Trying to read floating point value into integer");
    }
    if (dt_class != H5T_INTEGER && dt_class != H5T_FLOAT) {
        return zeus::unexpected(
            "Unexpected data class type; can only handle integer/non-integer");
    }

    size_t datatype_size = H5Tget_size(datatype);
    auto native_type =
        H5Cleanup<H5Tclose>(H5Tget_native_type(datatype, H5T_DIR_DEFAULT));
    size_t native_size = H5Tget_size(native_type);

    hid_t read_datatype = datatype;
    if (dt_class == H5T_INTEGER) {
        // Validate data type conversions for now. This is a bit annoying
        // but probably safer than just blindly assuming the conversion works.
        if ((native_type == H5T_NATIVE_CHAR || native_type == H5T_NATIVE_SHORT
             || native_type == H5T_NATIVE_INT || native_type == H5T_NATIVE_LONG
             || native_type == H5T_NATIVE_LLONG)
            && !std::is_signed_v<T>) {
            return zeus::unexpected("Will not copy signed to unsigned");
        }
        if ((native_type == H5T_NATIVE_UCHAR || native_type == H5T_NATIVE_USHORT
             || native_type == H5T_NATIVE_UINT || native_type == H5T_NATIVE_ULONG
             || native_type == H5T_NATIVE_ULLONG)
            && std::is_signed_v<T>) {
            return zeus::unexpected("Will not copy unsigned data into signed");
        }
        if (datatype_size != sizeof(T)) {
            return zeus::unexpected(
                fmt::format("Data type size mismatch: Trying to copy size {} into {}",
                            datatype_size,
                            sizeof(T)));
        }
    }

    // If native type double, we want to read that, even if we've been asked
    // for a float by the template instantiator. The caller probably doesn't
    // care if the data is declared as float or double internally.
    typename std::conditional<std::is_same_v<T, float>, double, T>::type output;

    if (H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &output) < 0) {
        throw std::runtime_error("Failed to read dataset");
        return zeus::unexpected("Failed to read dataset");
    }

    return output;
}

template <typename T>
inline hid_t H5T;

template <>
inline hid_t H5T<float> = H5T_NATIVE_FLOAT;

template <typename T>
auto write_scalar_hdf5_value(hid_t root_group, std::string path, T value) -> void {
    auto dataspace = H5Cleanup<H5Sclose>(H5Screate(H5S_SCALAR));
    auto dataset = H5Cleanup<H5Dclose>(H5Dcreate(root_group,
                                                 path.c_str(),
                                                 H5T<float>,
                                                 dataspace,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT));
    // dataspace
    auto ret = H5Dwrite(dataset, H5T<float>, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
}

template <>
auto write_scalar_hdf5_value<std::string>(hid_t root_group,
                                          std::string path,
                                          std::string value) -> void;

template <>
auto read_single_hdf5_value(hid_t root_group, const std::string path)
    -> zeus::expected<std::string, std::string>;

template <typename T>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<T>, std::string>;
template <>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<float>, std::string>;
template <>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<double>, std::string>;
