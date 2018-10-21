#pragma once

#include <opencv2/core/mat.hpp>
#include "image.h"
#include "OpenCLEnvironment.h"

class ImageProcessor {
private:
    OpenCLEnvironment env;

    cv::Mat convolve(const Image &image, const std::vector<float> &kernel, int kernelHeight);

    cv::Mat mirror(const Image &image, cl_kernel mirrorKernel);

public:
    explicit ImageProcessor(OpenCLEnvironment &env) : env(env) {}

    cv::Mat createSpectrumMat(const Image &image);

    cv::Mat boxBlur(const Image &image, unsigned int size);

    cv::Mat edges(const Image &image);

    cv::Mat grayscale(const Image &image);

    cv::Mat mirrorHorizontal(const Image &image);

    cv::Mat mirrorVertical(const Image &image);

    cv::Mat rotate90(const Image &image);
};