#pragma once

#include <opencv2/core/mat.hpp>
#include "image.h"
#include "OpenCLEnvironment.h"

class ImageProcessor {
private:
    OpenCLEnvironment env;

public:
    ImageProcessor(OpenCLEnvironment& env);

    cv::Mat createSpectrumMat(const Image &image);
};