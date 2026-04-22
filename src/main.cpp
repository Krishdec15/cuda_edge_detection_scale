#include <chrono>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>
#include <helper_cuda.h>

#include "npp_edge_filter.hpp"
#include "image_io.hpp"
#include "cmd_parser.hpp"

bool initializeCudaAndNpp() {
    const NppLibraryVersion* npp_version = nppGetLibVersion();
    std::cout << "NPP Library Version " << npp_version->major << "." 
              << npp_version->minor << "." << npp_version->build << std::endl;

    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);

    if (driver_version == 0) {
        std::cerr << "No CUDA driver found." << std::endl;
        return false;
    }

    std::cout << "CUDA Driver Version: " << driver_version / 1000 << "." << (driver_version % 100) / 10 << std::endl;
    std::cout << "CUDA Runtime Version: " << runtime_version / 1000 << "." << (runtime_version % 100) / 10 << std::endl;

    return checkCudaCapabilities(1, 0); // Need SM 1.0 minimum for basic stuff
}

int executeVideoProcessing(const std::string& input_video, const std::string& output_video) {
    cv::VideoCapture capture(input_video);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open video file: " << input_video << std::endl;
        return EXIT_FAILURE;
    }

    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
    int codec = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));

    cv::VideoWriter writer(output_video, codec, fps ? fps : 30, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Failed to initialize video writer: " << output_video << std::endl;
        return EXIT_FAILURE;
    }

    npp::ImageCPU_8u_C4 host_src(width, height);
    npp::ImageCPU_8u_C4 host_dst(width, height);
    NppSobelEdgeDetector edge_detector(width, height);
    
    cv::Mat frame;
    int processed_frames = 0;
    
    std::cout << "Processing video..." << std::endl;
    
    auto t_start = std::chrono::high_resolution_clock::now();
    while (true) {
        capture >> frame;
        if (frame.empty()) break;
        
        app_io::opencvMatToNpp(frame, host_src);
        edge_detector.process(host_src, host_dst);
        app_io::nppToOpencvMat(host_dst, frame);
        
        writer.write(frame);
        processed_frames++;
        
        if (processed_frames % 50 == 0) {
            std::cout << "Processed " << processed_frames << " frames..." << std::endl;
        }
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Done! Total frames: " << processed_frames << std::endl;
    std::cout << "Total time: " << elapsed_ms << " ms (" 
              << (elapsed_ms / processed_frames) << " ms/frame)" << std::endl;

    capture.release();
    writer.release();
    return EXIT_SUCCESS;
}

int executeImageProcessing(const std::string& input_image, const std::string& output_image) {
    npp::ImageCPU_8u_C4 host_src;
    try {
        app_io::loadStaticImage(input_image, host_src);
    } catch (const std::exception& e) {
        std::cerr << "Error loading image: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    npp::ImageCPU_8u_C4 host_dst(host_src.width(), host_src.height());
    NppSobelEdgeDetector edge_detector(host_src.width(), host_src.height());
    
    std::cout << "Processing static image (" << host_src.width() << "x" << host_src.height() << ")" << std::endl;

    auto t_start = std::chrono::high_resolution_clock::now();
    edge_detector.process(host_src, host_dst);
    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "Edge detection time: " << elapsed_ms << " ms" << std::endl;

    try {
        app_io::saveStaticImage(output_image, host_dst);
    } catch (const std::exception& e) {
        std::cerr << "Error saving image: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Output saved successfully." << std::endl;
    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "     CUDA Edge Detection at Scale       " << std::endl;
    std::cout << "========================================" << std::endl;

    findCudaDevice(argc, (const char**)argv);

    if (!initializeCudaAndNpp()) {
        std::cerr << "Failed to initialize CUDA environment. Exiting." << std::endl;
        return EXIT_FAILURE;
    }

    CmdArgParser parser(argc, argv);
    std::string ext = parser.getExtension();

    if (ext == ".mp4" || ext == ".avi" || ext == ".mov") {
        return executeVideoProcessing(parser.getInputFile(), parser.getOutputFile());
    } else {
        return executeImageProcessing(parser.getInputFile(), parser.getOutputFile());
    }
}
