#pragma once

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <helper_cuda.h>
#include <nppi.h>

#include <vector>

class SobelKernel {
public:
    explicit SobelKernel(const std::vector<Npp32f>& weights) {
        size_t bytes = weights.size() * sizeof(Npp32f);
        cudaMalloc(&device_kernel_ptr_, bytes);
        cudaMemcpy(device_kernel_ptr_, weights.data(), bytes, cudaMemcpyHostToDevice);
    }
    
    ~SobelKernel() { 
        if (device_kernel_ptr_) {
            cudaFree(device_kernel_ptr_); 
        }
    }
    
    // Disallow copy to avoid double free
    SobelKernel(const SobelKernel&) = delete;
    SobelKernel& operator=(const SobelKernel&) = delete;

    const Npp32f* getDevicePtr() const { return device_kernel_ptr_; }

private:
    Npp32f* device_kernel_ptr_ = nullptr;
};

class NppSobelEdgeDetector {
public:
    NppSobelEdgeDetector(int w, int h)
        : width_(w),
          height_(h),
          device_src_(w, h),
          device_gray_(w, h),
          device_edges_(w, h),
          device_output_broadcast_(w, h),
          kernel_x_({-0.25f, 0.0f, 0.25f, 
                     -0.50f, 0.0f, 0.50f, 
                     -0.25f, 0.0f, 0.25f}),
          kernel_y_({-0.25f, -0.50f, -0.25f, 
                      0.00f,  0.00f,  0.00f, 
                      0.25f,  0.50f,  0.25f}) {}

    ~NppSobelEdgeDetector() = default;

    void process(const npp::ImageCPU_8u_C4& host_input, npp::ImageCPU_8u_C4& host_output) const {
        const NppiSize roi_size{width_, height_};

        // Upload to Device
        device_src_.copyFrom(const_cast<Npp8u*>(host_input.data()), host_input.pitch());

        // Convert RGB to Grayscale
        NPP_CHECK_NPP(nppiRGBToGray_8u_AC4C1R(device_src_.data(), device_src_.pitch(),
                                              device_gray_.data(), device_gray_.pitch(), roi_size));

        // Apply Horizontal and Vertical Sobel Filters
        applyDirectionalFilter(device_gray_, device_edges_, kernel_x_);
        applyDirectionalFilter(device_gray_, device_gray_, kernel_y_);

        // Combine X and Y edges using bitwise OR
        NPP_CHECK_NPP(nppiOr_8u_C1R(device_edges_.data(), device_edges_.pitch(),
                                    device_gray_.data(), device_gray_.pitch(),
                                    device_gray_.data(), device_gray_.pitch(),
                                    roi_size));

        // Broadcast the 1-channel edges to a 4-channel image
        // Channel 0 (R)
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(device_gray_.data(), device_gray_.pitch(),
                                        device_output_broadcast_.data(), device_output_broadcast_.pitch(),
                                        roi_size));
        // Channel 1 (G)
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(device_gray_.data(), device_gray_.pitch(),
                                        device_output_broadcast_.data() + 1, device_output_broadcast_.pitch(),
                                        roi_size));
        // Channel 2 (B)
        NPP_CHECK_NPP(nppiCopy_8u_C1C4R(device_gray_.data(), device_gray_.pitch(),
                                        device_output_broadcast_.data() + 2, device_output_broadcast_.pitch(),
                                        roi_size));
        // Channel 3 (Alpha) - Set to 255
        NPP_CHECK_NPP(nppiSet_8u_C4CR(255, device_output_broadcast_.data() + 3,
                                      device_output_broadcast_.pitch(), roi_size));

        // Colorize edges by multiplying with the original image
        NPP_CHECK_NPP(nppiMul_8u_C4RSfs(device_output_broadcast_.data(), device_output_broadcast_.pitch(),
                                        device_src_.data(), device_src_.pitch(), 
                                        device_src_.data(), device_src_.pitch(), 
                                        roi_size, 8));

        // Download to Host
        device_src_.copyTo(host_output.data(), host_output.pitch());
    }

private:
    void applyDirectionalFilter(const npp::ImageNPP_8u_C1& dev_in, npp::ImageNPP_8u_C1& dev_out,
                                const SobelKernel& filter_kernel) const {
        npp::ImageNPP_16s_C1 device_temp_16s(width_, height_);
        NppiSize k_size = {3, 3};
        NppiPoint k_anchor = {1, 1};
        NppiSize roi = {width_, height_};

        // Filter: 8u to 16s
        NPP_CHECK_NPP(nppiFilter32f_8u16s_C1R(
            dev_in.data(), dev_in.pitch(),
            device_temp_16s.data(), device_temp_16s.pitch(),
            roi,
            filter_kernel.getDevicePtr(), k_size, k_anchor
        ));

        // Absolute value to keep positive edge magnitudes
        NPP_CHECK_NPP(nppiAbs_16s_C1R(device_temp_16s.data(), device_temp_16s.pitch(),
                                      device_temp_16s.data(), device_temp_16s.pitch(), roi));
        
        // Convert back: 16s to 8u
        NPP_CHECK_NPP(nppiConvert_16s8u_C1R(device_temp_16s.data(), device_temp_16s.pitch(),
                                            dev_out.data(), dev_out.pitch(), roi));
    }

    int width_;
    int height_;
    mutable npp::ImageNPP_8u_C4 device_src_;
    mutable npp::ImageNPP_8u_C1 device_gray_;
    mutable npp::ImageNPP_8u_C1 device_edges_;
    mutable npp::ImageNPP_8u_C4 device_output_broadcast_;
    const SobelKernel kernel_x_;
    const SobelKernel kernel_y_;
};
