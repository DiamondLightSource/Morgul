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
    H5Cleanup(const H5Cleanup &) = delete;
    H5Cleanup(H5Cleanup &&other) : id(other.id) {
        other.id = -1;
    }
    H5Cleanup &operator=(H5Cleanup &&other) {
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

using H5Fcleanup = H5Cleanup<H5Fclose>;
using H5Dcleanup = H5Cleanup<H5Dclose>;
using H5Scleanup = H5Cleanup<H5Sclose>;

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

    auto native_type =
        H5Cleanup<H5Tclose>(H5Tget_native_type(datatype, H5T_DIR_DEFAULT));
    size_t native_size = H5Tget_size(native_type);

    // Read as the native type, not the on-disk one; passing the on-disk type as
    // the memory type means HDF5 does no conversion at all, so e.g. big-endian
    // data would come back byte-swapped.
    hid_t read_datatype = native_type;
    if (dt_class == H5T_INTEGER) {
        // Validate data type conversions for now. This is a bit annoying
        // but probably safer than just blindly assuming the conversion works.
        // Note: the native type is a freshly allocated id rather than the
        // predefined H5T_NATIVE_* constant, so it must be interrogated rather
        // than compared against them.
        auto sign = H5Tget_sign(native_type);
        if (sign == H5T_SGN_ERROR) {
            return zeus::unexpected("Could not get integer data type sign");
        }
        if (sign == H5T_SGN_2 && !std::is_signed_v<T>) {
            return zeus::unexpected("Will not copy signed to unsigned");
        }
        if (sign == H5T_SGN_NONE && std::is_signed_v<T>) {
            return zeus::unexpected("Will not copy unsigned data into signed");
        }
        if (native_size != sizeof(T)) {
            return zeus::unexpected(
                fmt::format("Data type size mismatch: Trying to copy size {} into {}",
                            native_size,
                            sizeof(T)));
        }
    } else {
        // Anything that is not integer or float was rejected above, so this is
        // H5T_FLOAT. We always read those into a double, so we must ask HDF5 to
        // convert to that; otherwise reading e.g. a 4-byte float dataset only
        // fills half of the output and we get garbage.
        read_datatype = H5T_NATIVE_DOUBLE;
    }

    // If native type double, we want to read that, even if we've been asked
    // for a float by the template instantiator. The caller probably doesn't
    // care if the data is declared as float or double internally. This must
    // cover every floating point T, to match the H5T_NATIVE_DOUBLE above.
    std::conditional_t<std::is_floating_point_v<T>, double, T> output;

    if (H5Dread(dataset, read_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &output) < 0) {
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
                                                 H5T<T>,
                                                 dataspace,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT,
                                                 H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        throw std::runtime_error(fmt::format("Failed to create dataset {}", path));
    }
    if (H5Dwrite(dataset, H5T<T>, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value) < 0) {
        throw std::runtime_error(fmt::format("Failed to write dataset {}", path));
    }
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

/// @brief Turn off HDF5 automatic error stack printing for scope duration
class H5ErrorSilencer {
  public:
    H5ErrorSilencer() {
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    }
    ~H5ErrorSilencer() {
        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }
    H5ErrorSilencer(const H5ErrorSilencer &) = delete;
    H5ErrorSilencer &operator=(const H5ErrorSilencer &) = delete;

  private:
    H5E_auto2_t old_func;
    void *old_client_data;
};
