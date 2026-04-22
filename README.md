# CUDA Edge Detection at Scale

This repository contains the final independent project for the **CUDA at Scale for the Enterprise** course. 
The application processes a large volume of visual data (either images or hundreds of frames in high-res video files) using NVIDIA Performance Primitives (NPP) applying an edge detection filter. It satisfies the requirement of operating on "100s of small pieces of data or 10s of large pieces of data" because the target video typically includes hundreds, if not thousands, of frames processed efficiently on a GPU.

## Project Overview

The main functionality implements a Sobel operator for structural edge detection.
By leveraging memory mapping, batching with video frames, and hardware acceleration via `CUDA` and `NPP`, this allows for significant speedup vs single core CPU implementations.

### Algorithms and Kernels
The logic flow maps high-level image manipulation to hardware accelerated functions:

1. **RGB to Grayscale**: Using `nppiRGBToGray_8u_AC4C1R`, the loaded input is transformed to a single-channel 8-bit image to simplify the mathematical layout of gradients.
2. **Sobel Kernel Convolution**: A horizontal kernel `(-0.25, 0, 0.25 ...)` and a vertical kernel `(-0.25, -0.5, -0.25 ...)` estimate directional spatial gradients. `nppiFilter32f_8u16s_C1R` runs convolutions in parallel storing the signed intensity in intermediate `16s` bit representation.
3. **Absolute Mapping**: `nppiAbs_16s_C1R` recovers negative directional magnitudes. 
4. **Channel Recombination / Tinting**: `nppiOr_8u_C1R` applies the bitwise overlap of both gradient directions. Finally, `nppiMul_8u_C4RSfs` multiplies these combined edges with the original structural colors (giving the edges their original object colors).

## Dependencies

- **CUDA Toolkit** (tested on 11.8 / 12+)
- **NVIDIA Performance Primitives (NPP)** (installed alongside CUDA)
- **CUDA Samples Utility** (for headers like `<helper_cuda.h>`)
- **FreeImage** (Handles simple image read/writes) (`apt install libfreeimage-dev`)
- **OpenCV** (Used strictly for video I/O) (`apt install libopencv-dev`)

## Building the Project

Ensure you have your environment set up with `nvcc` reachable in your PATH. We use a standard command-line utility for parsing arguments.

1. Clone this repository locally.
2. Ensure dependency locations match `CUDA_PATH` in the `Makefile`. 
3. Run `make`.

```bash
cd cuda_edge_detection_scale
make clean && make
```
This builds the `run_filter` executable inside `/bin`.

## Running & CLI Arguments

The CLI takes `--input` and `--output` flags and automatically dictates processing mode via file extensions (e.g. `.mp4` routes to video stream batching while `.png` processes a single image).

```bash
# Process a large collection of frames (video)
./bin/run_filter --input data/sample.mp4 --output data/result.mp4

# Process an image 
./bin/run_filter --input data/sample.png --output data/sample_edges.png
```

## Lessons Learned

1. **Data Pitching:** Image structures generated on the system via FreeImage vs. OpenCV have different spatial layouts. Mapping pitch buffers securely from `cv::Mat` to `npp::ImageCPU` prevents image shearing.
2. **Buffer Overflows:** Filtering from an 8-bit unsign (`8u`) using gradient convolutions can yield negative answers. Safely staging with an intermediate 16-bit signed `16s` representation before converting back allowed correct magnitude derivation.

## Execution Proof

Terminal output proving successful CUDA allocation and video streaming performance processing ~300 individual frame computations contextually matches the required data limits set by the Assignment Rubric. (Please see execution screenshots attached below/alongside your Coursera portal submissions).
