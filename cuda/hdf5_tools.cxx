#include "hdf5_tools.hpp"

auto H5Iget_name(hid_t identifier) -> std::optional<std::string> {
    ssize_t name_size = H5Iget_name(identifier, NULL, 0);
    if (name_size == 0) {
        return std::nullopt;
    }
    std::string name;
    name.reserve(name_size + 1);
    H5Iget_name(identifier, name.data(), name_size + 1);
    return name;
}

template <>
auto read_single_hdf5_value(hid_t root_group, const std::string path)
    -> zeus::expected<std::string, std::string> {
    auto dataset = H5Cleanup<H5Dclose>(H5Dopen(root_group, path.c_str(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return zeus::unexpected(fmt::format("Invalid HDF5 group: {}", path));
    }
    auto datatype = H5Cleanup<H5Tclose>(H5Dget_type(dataset));
    if (H5Tget_class(datatype) != H5T_STRING) {
        return zeus::unexpected("Dataset type class is not string!");
    }

    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    H5S_class_t dataspace_type = H5Sget_simple_extent_type(dataspace);
    if (dataspace_type != H5S_SCALAR) {
        return zeus::unexpected(
            fmt::format("Do not know how to read non-scalar dataset {}/{}",
                        H5Iget_name(dataset).value(),
                        path));
    }
    bool is_var = H5Tis_variable_str(datatype);
    if (!is_var) {
        return zeus::unexpected("Unhandled fixed-length string");
    }
    char *buffer = nullptr;

    if (H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &buffer) < 0) {
        return zeus::unexpected("Failed to read");
    }
    std::string output{buffer};
    // We need to explicitly reclaim the automatically allocated space
    if (H5Treclaim(datatype, dataspace, H5P_DEFAULT, &buffer) < 0) {
        return zeus::unexpected("Failed to reclaim");
    }

    return output;
}

template <typename T>
auto _read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<T>, std::string> {
    auto dataset =
        H5Cleanup<H5Dclose>(H5Dopen(root_group, path_to_dataset.data(), H5P_DEFAULT));
    if (dataset == H5I_INVALID_HID) {
        return zeus::unexpected(fmt::format("Invalid HDF5 group: {}", path_to_dataset));
    }
    auto dataspace = H5Cleanup<H5Sclose>(H5Dget_space(dataset));
    if (dataspace < 0) {
        return zeus::unexpected("Could not get data space");
    }
    auto rank = H5Sget_simple_extent_ndims(dataspace);
    if (rank != 2) {
        return zeus::unexpected("Dataset is not 2D");
    }
    hsize_t dims[2];
    H5Sget_simple_extent_dims(dataspace, dims, nullptr);

    hid_t hdf5_type;
    if constexpr (std::is_same_v<T, float>) {
        hdf5_type = H5T_NATIVE_FLOAT;
    } else if constexpr (std::is_same_v<T, double>) {
        hdf5_type = H5T_NATIVE_DOUBLE;
    } else {
        static_assert(false, "Unrecognised data type for reading 2D array");
    }
    // std::vector<T> buffer(dims[0] * dims[1]);
    auto data = std::make_unique<T[]>(dims[0] * dims[1]);
    if (H5Dread(dataset, hdf5_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.get()) < 0) {
        return zeus::unexpected("Failed to read data");
    }
    return Array2D(std::move(data), dims[1], dims[0]);
}

template <>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<float>, std::string> {
    return _read_2d_dataset<float>(root_group, path_to_dataset);
}
template <>
auto read_2d_dataset(hid_t root_group, std::string_view path_to_dataset)
    -> zeus::expected<Array2D<double>, std::string> {
    return _read_2d_dataset<double>(root_group, path_to_dataset);
}
