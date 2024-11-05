cmake_minimum_required(VERSION 3.20...3.30)
# Constraints on minimum version:
# - h5read: FindHDF5 only creates interface targets on 3.20+

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(morgul LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS yes)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(SetDefaultBuildRelWithDebInfo)
include(AlwaysColourCompilation)


# include_directories(include)

# Dependency fetching
set(FETCHCONTENT_QUIET OFF)
include(FetchContent)
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.0.2
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(fmt)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY  https://github.com/p-ranav/argparse
    GIT_TAG 68fd0277eea5412aff9b91c0b70efc698359dae0 # 6.3.1
)
FetchContent_MakeAvailable(argparse)

find_package(LZ4 REQUIRED)
find_package(Bitshuffle REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED)
find_package(lodepng REQUIRED)

add_executable(morgul-cuda cuda/process.cxx)
target_compile_options(morgul-cuda PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
target_link_libraries(
    morgul-cuda
    PRIVATE
    CUDA::cudart
    fmt
    hdf5::hdf5
    nlohmann_json::nlohmann_json
    lodepng
)
