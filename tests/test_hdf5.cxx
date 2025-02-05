#include <fmt/core.h>
#include <gtest/gtest.h>

#include "hdf5_tools.hpp"

using testing::TempDir;
using namespace fmt;

// Demonstrate some basic assertions.
TEST(HDF5, TestWriteScalarString) {
    std::string path = TempDir() + "test.h5";
    {
        auto file = H5Fcleanup(
            H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        write_scalar_hdf5_value<std::string>(file, "/test", "This is test");
    }
    {
        auto file =
            H5Cleanup<H5Fclose>(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
        auto value = read_single_hdf5_value<std::string>(file, "/test");
        EXPECT_TRUE(value.has_value());
        EXPECT_STREQ(value.value().c_str(), "This is test");
    }
}