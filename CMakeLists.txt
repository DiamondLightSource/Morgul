cmake_minimum_required(VERSION 3.24...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+
# - 3.24 Added FetchContent find_package behaviour

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(morgul LANGUAGES C CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)
include(EnsureSubmodule)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
#target_compile_options(libmorgul PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
    message(STATUS "Detected aarch64; setting native CPU")
#    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:CXX>>:-Ofast>)
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:CUDA>,$<COMPILE_LANGUAGE:CXX>>:-mcpu=native>)
    list(REMOVE_ITEM CXX_FLAGS_RELWITHDEBINFO -O2)
    list(REMOVE_ITEM CUDA_FLAGS_RELWITHDEBINFO -O2)
endif()

# Dependency fetching
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.2.0
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(fmt)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.12.0
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/p-ranav/argparse
    GIT_TAG 3eda91b2e1ce7d569f84ba295507c4cd8fd96910 # v3.2
)
FetchContent_MakeAvailable(argparse)
FetchContent_Declare(
    date_src
    GIT_REPOSITORY https://github.com/HowardHinnant/date.git
    GIT_TAG        v3.0.3
)
FetchContent_MakeAvailable(date_src)
FetchContent_Declare(
    zeus_expected
    GIT_REPOSITORY https://github.com/zeus-cpp/expected.git
    GIT_TAG v1.2.0
)
set(BUILD_TESTING OFF)
FetchContent_MakeAvailable(zeus_expected)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/e235eb34c6c4fed790ccdad4b16394301360dcd4.zip
)
FetchContent_MakeAvailable(googletest)

find_package(LZ4 REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)
find_package(HDF5 REQUIRED)
find_package(lodepng REQUIRED)
find_package(cppzmq REQUIRED)

ensure_submodule(modules/bitshuffle)
add_library(
    bitshuffle
    modules/bitshuffle/src/bitshuffle.c
    modules/bitshuffle/src/bitshuffle_core.c
    modules/bitshuffle/src/iochain.c
    modules/bitshuffle/lz4/lz4.c
)
target_include_directories(bitshuffle PUBLIC modules/bitshuffle/src modules/bitshuffle/lz4)

add_library(
    libmorgul
    STATIC
    cuda/correct.cxx
    cuda/calibration.cxx
    cuda/hdf5_tools.cxx
    cuda/live.cxx
    cuda/kernels.cu
)
target_link_libraries(
    libmorgul
    PUBLIC
    argparse
    bitshuffle
    cppzmq
    CUDA::cudart
    date::date
    fmt
    hdf5::hdf5
    lodepng
    LZ4::LZ4
    nlohmann_json::nlohmann_json
    zeus::expected
)
target_compile_options(libmorgul PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(libmorgul PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_compile_options(libmorgul PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_include_directories(libmorgul INTERFACE cuda)
# target_compile_options(morgul-cuda PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:--generate-line-info>")

add_executable(morgul-cuda cuda/main.cxx)
target_link_libraries(morgul-cuda PRIVATE libmorgul)

enable_testing()
add_executable(
    morgul-tests
    tests/test_hdf5.cxx
)
target_link_libraries(morgul-tests GTest::gtest_main libmorgul fmt)
include(GoogleTest)
gtest_discover_tests(morgul-tests)
