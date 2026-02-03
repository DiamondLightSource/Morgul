# Morgul

High-performance processing pipeline for Jungfrau X-ray detectors at Diamond Light Source.

## Architecture

Two components:
- **cuda/** (primary): CUDA-based live correction system in C++20/CUDA
- **morgul/** (legacy): Python offline processing tools

## CUDA Component

### Commands
- `morgul-cuda live` - Real-time streaming from detector with GPU correction
- `morgul-cuda correct` - Offline correction of HDF5 files
- `morgul-cuda pedestal` - Generate pedestal calibration from dark frames

### Key Files
| File | Purpose |
|------|---------|
| `main.cxx` | CLI entry point, argument parsing |
| `live.cxx` | Live streaming - multi-threaded SLS receiver handling, ZMQ output |
| `kernels.cu` | CUDA kernels: image correction, pedestal accumulation/finalization |
| `calibration.cxx/hpp` | Load pedestals/gains from HDF5, upload to GPU |
| `cuda_common.hpp` | Smart pointers (`shared_device_ptr<T>`), CUDA utilities |
| `constants.hpp` | Detector configs (JF1M, JF9M), module layouts |
| `hdf5_tools.cxx/hpp` | HDF5 I/O abstractions |

### Processing Pipeline
1. Receive UDP packets via SLS receivers (one thread per port)
2. Parse SLS binary headers + DLS JSON metadata
3. Upload raw frames to GPU (pinned memory)
4. Apply correction kernel: `corrected = (raw - pedestal) / (gain * energy_keV)`
5. LZ4 compress and send via ZMQ to downstream writers

### Detector Data Format
- Raw pixel: 16-bit (14-bit ADC value + 2-bit gain mode)
- Gain modes: 0, 1, 2
- Half-module: 1024×256 pixels
- Full module: 1024×512 pixels (2 half-modules)

### Calibration
- Pedestals: per-pixel, per-gain mean from dark frames
- Gains: per-pixel, per-gain conversion factors
- Discovery via `JUNGFRAU_GAIN_MAPS` and `JUNGFRAU_CALIBRATION_LOG` env vars

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build
```

Requirements: CMake 3.24+, CUDA 12.2+, LZ4, HDF5, ZeroMQ, Boost

## Python Component (morgul/)

CLI tools via typer: `pedestal`, `mask`, `correct`, `nxmx`, `gainmap`, `view`

Used for offline calibration generation, NXmx conversion, and visualization.
