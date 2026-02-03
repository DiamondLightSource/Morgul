# Morgul Project Analysis

A comprehensive analysis of the Morgul processing toolset for Diamond Jungfrau detectors.

## Executive Summary

**Morgul** is a high-performance data processing pipeline for Jungfrau X-ray detectors at Diamond Light Source. The project contains two main components:

- **morgul-cuda**: The primary, modern CUDA-based live correction system (~3,254 lines of C++/CUDA)
- **morgul (Python)**: Legacy offline processing toolset for calibration, conversion, and visualization

---

## Project Structure Overview

```mermaid
graph TB
    subgraph Project["Morgul Project"]
        subgraph CUDA["cuda/ (Primary)"]
            main[main.cxx<br/>CLI Entry Point]
            live[live.cxx<br/>Live Streaming]
            correct[correct.cxx<br/>Offline Correction]
            pedestal[pedestal.cxx<br/>Pedestal Generation]
            kernels[kernels.cu<br/>CUDA Kernels]
            calibration[calibration.cxx/hpp<br/>Calibration Data]
            cuda_common[cuda_common.hpp<br/>CUDA Utilities]
            hdf5_tools[hdf5_tools.cxx/hpp<br/>HDF5 I/O]
            constants[constants.hpp<br/>Detector Config]
        end

        subgraph Python["morgul/ (Legacy)"]
            morgul_py[morgul.py<br/>CLI App]
            config[config.py]
            py_correct[morgul_correct.py]
            py_pedestal[morgul_pedestal.py]
            py_mask[morgul_mask.py]
            py_nxmx[morgul_nxmx.py]
        end

        subgraph Build["Build System"]
            cmake[CMakeLists.txt]
            pyproject[pyproject.toml]
        end

        subgraph Modules["External Modules"]
            bitshuffle[bitshuffle/]
            sls[slsDetectorPackage/]
        end
    end

    main --> live
    main --> correct
    main --> pedestal
    live --> kernels
    correct --> kernels
    pedestal --> kernels
    kernels --> cuda_common
    live --> calibration
    correct --> calibration
    calibration --> hdf5_tools
    live --> constants

    cmake --> CUDA
    pyproject --> Python
```

---

## CUDA Architecture (Primary Focus)

### File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `main.cxx` | 138 | CLI argument parsing and command dispatch |
| `live.cxx` | 953 | Live streaming acquisition and processing |
| `correct.cxx` | 112 | Offline correction workflow |
| `pedestal.cxx` | 33 | Pedestal generation workflow |
| `kernels.cu` | 180 | CUDA computation kernels |
| `kernels.h` | 30 | Kernel declarations |
| `calibration.cxx` | 330 | Calibration data loading/GPU upload |
| `calibration.hpp` | 81 | Calibration data structures |
| `cuda_common.hpp` | 400 | CUDA memory management & utilities |
| `cuda_argparse.hpp` | ~100 | CUDA device selection |
| `hdf5_tools.cxx/hpp` | 281 | HDF5 I/O abstractions |
| `constants.hpp` | 150 | Detector configurations |
| `array2d.hpp` | 78 | 2D array container |
| `common.hpp` | 238 | General utilities |
| `commands.hpp` | 30 | Command argument structures |

### Command Flow

```mermaid
flowchart LR
    CLI[main.cxx<br/>CLI Parser]

    CLI -->|"morgul-cuda live"| LIVE[live.cxx<br/>Live Mode]
    CLI -->|"morgul-cuda correct"| CORRECT[correct.cxx<br/>Offline Mode]
    CLI -->|"morgul-cuda pedestal"| PEDESTAL[pedestal.cxx<br/>Pedestal Mode]

    subgraph Processing
        LIVE --> KERNELS[kernels.cu]
        CORRECT --> KERNELS
        PEDESTAL --> KERNELS
    end

    subgraph Output
        LIVE --> ZMQ[ZMQ Stream]
        CORRECT --> HDF5[HDF5 Files]
        PEDESTAL --> HDF5
    end
```

---

## Live Processing Pipeline

The live processing system (`live.cxx`) is the heart of morgul-cuda, handling real-time data from Jungfrau detectors.

### Live Data Flow

```mermaid
flowchart TB
    subgraph Input["Data Input"]
        DET[Jungfrau Detector]
        SLS[SLS Receivers<br/>UDP Packets]
    end

    subgraph Headers["Header Parsing"]
        SLSHDR[SLSHeader<br/>Binary Metadata]
        DLSHDR[DLSHeaderAdditions<br/>JSON Metadata]
    end

    subgraph Cache["Calibration Cache"]
        PEDLIB[PedestalsLibrary<br/>Thread-Safe Cache]
        GAINLIB[GainData<br/>GPU-Resident]
    end

    subgraph Processing["GPU Processing"]
        UPLOAD[Upload to GPU<br/>Pinned Memory]
        CORRECT[jungfrau_image_corrections<br/>Kernel]
        COMPRESS[LZ4 Compression]
    end

    subgraph Output["Output"]
        ZMQOUT[ZMQ Publisher<br/>To Writers]
        SHMPEDESTAL["Shared Memory<br/>pedestals.h5"]
    end

    DET --> SLS
    SLS --> SLSHDR
    SLS --> DLSHDR
    SLSHDR --> UPLOAD
    DLSHDR --> PEDLIB
    PEDLIB --> CORRECT
    GAINLIB --> CORRECT
    UPLOAD --> CORRECT
    CORRECT --> COMPRESS
    COMPRESS --> ZMQOUT
    PEDLIB -.->|Save| SHMPEDESTAL
```

### Multi-Threaded Architecture

```mermaid
flowchart TB
    subgraph Listeners["Per-Port Listeners"]
        L1[Listener 1<br/>Port N]
        L2[Listener 2<br/>Port N+1]
        L3[Listener 3<br/>Port N+2]
        LN[Listener N<br/>...]
    end

    subgraph Shared["Shared Resources"]
        PEDCACHE[PedestalsLibrary<br/>Mutex Protected]
        GAINCACHE[GainData Cache]
        CUDADEV[CUDA Device]
    end

    subgraph Handlers["DataStreamHandler"]
        H1[Handler 1<br/>Half-Module 0]
        H2[Handler 2<br/>Half-Module 1]
        H3[Handler 3<br/>Half-Module 2]
    end

    L1 --> H1
    L2 --> H2
    L3 --> H3

    H1 --> PEDCACHE
    H2 --> PEDCACHE
    H3 --> PEDCACHE

    H1 --> CUDADEV
    H2 --> CUDADEV
    H3 --> CUDADEV
```

---

## CUDA Kernel Architecture

### Image Correction Kernel

The main processing kernel `jungfrau_image_corrections` performs per-pixel correction:

```mermaid
flowchart LR
    subgraph Input["Input (per pixel)"]
        RAW[16-bit Raw Value<br/>14-bit data + 2-bit gain]
    end

    subgraph Extract["Extraction"]
        VALUE[14-bit ADC Value]
        GAIN[2-bit Gain Mode<br/>0, 1, or 2]
    end

    subgraph Calibration["Calibration Lookup"]
        PED[Pedestal Value<br/>per pixel, per gain]
        GAINVAL[Gain Factor<br/>per pixel, per gain]
    end

    subgraph Compute["Computation"]
        SUB["Subtract Pedestal<br/>(raw - pedestal)"]
        DIV["Apply Gain<br/>/ (gain × energy_keV)"]
    end

    subgraph Output["Output"]
        PHOTON[uint16_t<br/>Photon Count]
    end

    RAW --> VALUE
    RAW --> GAIN
    VALUE --> SUB
    GAIN --> PED
    GAIN --> GAINVAL
    PED --> SUB
    SUB --> DIV
    GAINVAL --> DIV
    DIV --> PHOTON
```

### Kernel Thread Organization

```
Grid:    (32, 8) blocks = 256 blocks total
Threads: (32, 32) per block = 1024 threads per block
Coverage: 1024 × 256 pixels (one half-module)
```

```mermaid
block-beta
    columns 8

    block:grid["Half-Module 1024×256"]:8
        B1["Block<br/>0,0"] B2["Block<br/>1,0"] B3["..."] B4["Block<br/>31,0"]
        B5["Block<br/>0,1"] B6["Block<br/>1,1"] B7["..."] B8["Block<br/>31,1"]
        B9["..."] B10["..."] B11["..."] B12["..."]
        B13["Block<br/>0,7"] B14["Block<br/>1,7"] B15["..."] B16["Block<br/>31,7"]
    end
```

### Pedestal Accumulation Pipeline

```mermaid
flowchart TB
    subgraph Collect["Dark Frame Collection"]
        FRAMES[N Dark Frames<br/>per Gain Mode]
    end

    subgraph Accumulate["Per-Pixel Accumulation"]
        COUNT[count++]
        SUMX[sum_x += value]
        SUMXSQ[sum_x² += value²]
    end

    subgraph Finalize["Finalization"]
        MEAN["pedestal = sum_x / count"]
        VARIANCE["variance = (sum_x² - mean²) / count"]
        MASK["if count == 0: bad_pixel"]
    end

    subgraph Output["Output"]
        PEDFILE[pedestals.h5<br/>Per-gain pedestals]
        MASKFILE[mask.h5<br/>Bad pixel map]
    end

    FRAMES --> COUNT
    FRAMES --> SUMX
    FRAMES --> SUMXSQ
    COUNT --> MEAN
    SUMX --> MEAN
    SUMXSQ --> VARIANCE
    COUNT --> MASK
    MEAN --> PEDFILE
    VARIANCE --> PEDFILE
    MASK --> MASKFILE
```

---

## Memory Management

### CUDA Smart Pointers

```mermaid
classDiagram
    class shared_device_ptr~T~ {
        +T* ptr
        +size_t count
        +get() T*
        +operator[]
        +reset()
    }

    class raw_device_ptr~T~ {
        +T* ptr
        +get() T*
        +operator[]
    }

    class CudaStream {
        +cudaStream_t stream
        +synchronize()
    }

    class CudaEvent {
        +cudaEvent_t event
        +record(stream)
        +synchronize()
    }

    shared_device_ptr --> raw_device_ptr : wraps
    CudaStream --> CudaEvent : records
```

### Memory Allocation Helpers

| Function | Purpose |
|----------|---------|
| `make_cuda_malloc<T>(n)` | Allocate device memory |
| `make_cuda_pinned_malloc<T>(n)` | Allocate pinned host memory |
| `make_cuda_managed_malloc<T>(n)` | Allocate unified memory |
| `cuda_memcpy_htod()` | Host to device transfer |
| `cuda_memcpy_dtoh()` | Device to host transfer |

---

## Calibration Data Architecture

### CalibrationDataPath Structure

```mermaid
classDiagram
    class CalibrationDataPath {
        +path pedestal_path
        +optional~path~ mask_path
        +path gain_path
    }

    class PedestalData {
        +Array2D~float~[] data
        +ModuleMode mode
        +upload() void
        +get_gpu_ptrs(hmi) float**
    }

    class GainData {
        +Array2D~float~[] data
        +upload() void
        +get_gpu_ptrs(hmi) float**
    }

    CalibrationDataPath --> PedestalData : loads
    CalibrationDataPath --> GainData : loads
```

### Calibration Discovery

```mermaid
flowchart TB
    subgraph Env["Environment Variables"]
        GAINENV["JUNGFRAU_GAIN_MAPS"]
        CALLOG["JUNGFRAU_CALIBRATION_LOG"]
    end

    subgraph Lookup["Lookup Process"]
        EXP["Match Exposure Time"]
        TS["Match Timestamp<br/>(newest <= acquisition)"]
    end

    subgraph Files["Calibration Files"]
        PEDFILE["pedestals.h5<br/>3 gain modes × pixels"]
        MASKFILE["mask.h5<br/>Bad pixel bitmap"]
        GAINFILE["gain.bin<br/>3 gain modes × pixels"]
    end

    GAINENV --> GAINFILE
    CALLOG --> EXP
    EXP --> TS
    TS --> PEDFILE
    TS --> MASKFILE
```

---

## Detector Configurations

### Supported Detectors

| Detector | Modules | Half-Modules | Total Pixels |
|----------|---------|--------------|--------------|
| JF1M | 2 | 4 | 2 × 1024 × 512 |
| JF9M | 18 | 36 | 18 × 1024 × 512 |
| JF9M-SIM | 18 | 36 | 18 × 1024 × 512 |

### Module Layout (JF9M)

```mermaid
block-beta
    columns 3

    M1["Module 1"] M2["Module 2"] M3["Module 3"]
    M4["Module 4"] M5["Module 5"] M6["Module 6"]
    M7["Module 7"] M8["Module 8"] M9["Module 9"]
    M10["Module 10"] M11["Module 11"] M12["Module 12"]
    M13["Module 13"] M14["Module 14"] M15["Module 15"]
    M16["Module 16"] M17["Module 17"] M18["Module 18"]
```

Each module consists of 2 half-modules (1024×256 each).

---

## Build System

### Dependencies

```mermaid
flowchart TB
    subgraph CMake["CMakeLists.txt"]
        TARGET[morgul-cuda]
        LIB[libmorgul]
        TEST[morgul-tests]
    end

    subgraph FetchContent["FetchContent Dependencies"]
        FMT[fmt 11.2.0]
        JSON[nlohmann_json 3.12.0]
        ARGPARSE[argparse]
        DATE[date v3.0.3]
        EXPECTED[zeus_expected v1.2.0]
        GTEST[googletest]
        RWQ[readerwriterqueue v1.0.7]
    end

    subgraph System["System Libraries"]
        LZ4[LZ4]
        HDF5[HDF5]
        PNG[lodepng]
        ZMQ[cppzmq]
        BOOST[Boost]
    end

    subgraph Submodules["Git Submodules"]
        BITSHUFFLE[bitshuffle]
        SLS[slsDetectorPackage]
    end

    FetchContent --> LIB
    System --> LIB
    Submodules --> LIB
    LIB --> TARGET
    LIB --> TEST
```

### Build Requirements

- CMake 3.24+
- CUDA 12.2+
- C++20 / CUDA C++20
- System packages: LZ4, HDF5, ZeroMQ, Boost

---

## Python Toolset (Legacy)

### Module Overview

```mermaid
flowchart TB
    subgraph CLI["morgul.py CLI"]
        CALIBRATION["Calibration Commands"]
        UTILS["Utility Commands"]
    end

    subgraph Calibration
        PEDESTAL[morgul_pedestal.py<br/>Dark frame processing]
        MASK[morgul_mask.py<br/>Bad pixel detection]
        CORRECT[morgul_correct.py<br/>Offline correction]
        NXMX[morgul_nxmx.py<br/>NXmx conversion]
    end

    subgraph Utils
        GAINMAP[morgul_gainmap.py<br/>Gain map I/O]
        VIEW[view.py<br/>Napari viewer]
        WATCH[watcher/<br/>File monitoring]
    end

    CALIBRATION --> PEDESTAL
    CALIBRATION --> MASK
    CALIBRATION --> CORRECT
    CALIBRATION --> NXMX
    UTILS --> GAINMAP
    UTILS --> VIEW
    UTILS --> WATCH
```

---

## Data Flow Summary

### End-to-End Processing

```mermaid
flowchart LR
    subgraph Acquisition["Data Acquisition"]
        XRAY[X-Ray Source]
        SAMPLE[Sample]
        DETECTOR[Jungfrau Detector]
    end

    subgraph Transport["Data Transport"]
        UDP[UDP Packets]
        SLS[SLS Receivers]
    end

    subgraph Processing["CUDA Processing"]
        PARSE[Parse Headers]
        UPLOAD[Upload to GPU]
        CORRECT[Apply Corrections]
        COMPRESS[LZ4 Compress]
    end

    subgraph Output["Data Output"]
        ZMQ[ZMQ Stream]
        WRITER[File Writers]
        HDF5[HDF5 Files]
    end

    XRAY --> SAMPLE --> DETECTOR
    DETECTOR --> UDP --> SLS
    SLS --> PARSE --> UPLOAD --> CORRECT --> COMPRESS
    COMPRESS --> ZMQ --> WRITER --> HDF5
```

---

## Key Design Patterns

1. **GPU-First Processing**: All heavy computation on CUDA; CPU handles I/O only
2. **Smart Memory Management**: `shared_device_ptr<T>` ensures CUDA memory safety
3. **Stream-Based Parallelism**: CUDA streams enable async operations
4. **Lock-Free Communication**: `moodycamel::ReaderWriterQueue` for threads
5. **Result-Type Error Handling**: `zeus::expected<T, E>` for explicit errors
6. **Hot-Swappable Calibration**: Pedestals updated on-the-fly during acquisition
7. **Multi-Detector Support**: Configuration-driven for JF1M, JF9M variants

---

## File Locations Reference

### Core CUDA Files
- `cuda/main.cxx` - CLI entry point
- `cuda/live.cxx` - Live streaming system
- `cuda/kernels.cu` - CUDA computation kernels
- `cuda/calibration.cxx` - Calibration data management
- `cuda/cuda_common.hpp` - Memory utilities

### Configuration
- `cuda/constants.hpp` - Detector specifications
- `CMakeLists.txt` - Build configuration
- `pyproject.toml` - Python dependencies

### Python Tools
- `morgul/morgul.py` - CLI application
- `morgul/config.py` - Configuration
- `morgul/morgul_correct.py` - Offline correction
