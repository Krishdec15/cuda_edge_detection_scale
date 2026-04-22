#pragma once

#include <ImageIO.h>
#include <ImagesCPU.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace app_io
{

    // Read an image using FreeImage
    void loadStaticImage(const std::string &filepath, npp::ImageCPU_8u_C4 &host_image)
    {
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT fmt = FreeImage_GetFileType(filepath.c_str());
        if (fmt == FIF_UNKNOWN)
        {
            fmt = FreeImage_GetFIFFromFilename(filepath.c_str());
        }

        if (fmt == FIF_UNKNOWN)
        {
            throw std::runtime_error("Unsupported image format: " + filepath);
        }

        FIBITMAP *bitmap = nullptr;
        if (FreeImage_FIFSupportsReading(fmt))
        {
            bitmap = FreeImage_Load(fmt, filepath.c_str());
        }

        if (!bitmap)
        {
            throw std::runtime_error("Failed to load image: " + filepath);
        }

        // Force 32 bits per pixel / RGBA
        FIBITMAP *pBitmap32 = FreeImage_ConvertTo32Bits(bitmap);
        FreeImage_Unload(bitmap);

        npp::ImageCPU_8u_C4 new_host_image(FreeImage_GetWidth(pBitmap32), FreeImage_GetHeight(pBitmap32));

        unsigned int pitch = FreeImage_GetPitch(pBitmap32);
        const Npp8u *pSrcLine = FreeImage_GetBits(pBitmap32) + pitch * (FreeImage_GetHeight(pBitmap32) - 1);
        Npp8u *pDstLine = new_host_image.data();
        unsigned int dstPitch = new_host_image.pitch();

        for (size_t y = 0; y < new_host_image.height(); ++y)
        {
            memcpy(pDstLine, pSrcLine, new_host_image.width() * 4);
            pSrcLine -= pitch; // FreeImage is upside down
            pDstLine += dstPitch;
        }

        FreeImage_Unload(pBitmap32);
        host_image.swap(new_host_image);
    }

    // Write an image using FreeImage
    void saveStaticImage(const std::string &filepath, const npp::ImageCPU_8u_C4 &host_image)
    {
        FIBITMAP *resultBitmap = FreeImage_Allocate(host_image.width(), host_image.height(), 32);
        if (!resultBitmap)
        {
            throw std::runtime_error("Failed to allocate FreeImage bitmap for: " + filepath);
        }

        unsigned int dstPitch = FreeImage_GetPitch(resultBitmap);
        Npp8u *pDstLine = FreeImage_GetBits(resultBitmap) + dstPitch * (host_image.height() - 1);
        const Npp8u *pSrcLine = host_image.data();
        unsigned int srcPitch = host_image.pitch();

        for (size_t y = 0; y < host_image.height(); ++y)
        {
            memcpy(pDstLine, pSrcLine, host_image.width() * 4);
            pSrcLine += srcPitch;
            pDstLine -= dstPitch; // FreeImage is upside down
        }

        if (!FreeImage_Save(FIF_PNG, resultBitmap, filepath.c_str(), 0))
        {
            FreeImage_Unload(resultBitmap);
            throw std::runtime_error("Failed to save image to: " + filepath);
        }

        FreeImage_Unload(resultBitmap);
    }

    // Convert OpenCV Mat to NPP ImageCPU
    void opencvMatToNpp(const cv::Mat &cv_frame, npp::ImageCPU_8u_C4 &host_image)
    {
        cv::Mat rgbaFrame;
        if (cv_frame.channels() == 3)
        {
            cv::cvtColor(cv_frame, rgbaFrame, cv::COLOR_BGR2RGBA);
        }
        else if (cv_frame.channels() == 1)
        {
            cv::cvtColor(cv_frame, rgbaFrame, cv::COLOR_GRAY2RGBA);
        }
        else
        {
            rgbaFrame = cv_frame;
        }

        const int bytes_per_row = rgbaFrame.cols * rgbaFrame.elemSize();
        for (int y = 0; y < rgbaFrame.rows; ++y)
        {
            std::memcpy(host_image.data() + y * host_image.pitch(),
                        rgbaFrame.data + y * rgbaFrame.step,
                        bytes_per_row);
        }
    }

    // Convert NPP ImageCPU back to OpenCV Mat
    void nppToOpencvMat(const npp::ImageCPU_8u_C4 &host_image, cv::Mat &cv_frame)
    {
        cv::Mat rgbaFrame(host_image.height(), host_image.width(), CV_8UC4);
        const int bytes_per_row = host_image.width() * 4;

        for (int y = 0; y < host_image.height(); ++y)
        {
            std::memcpy(rgbaFrame.data + y * rgbaFrame.step,
                        host_image.data() + y * host_image.pitch(),
                        bytes_per_row);
        }
        cv::cvtColor(rgbaFrame, cv_frame, cv::COLOR_RGBA2BGR);
    }

} // namespace app_io
