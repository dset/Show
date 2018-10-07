#pragma once

#include <OpenCL/opencl.h>
#include <opencv2/core/mat.hpp>
#include "image.h"

class ImageProcessor {
private:
    cl_context clContext;
    cl_program clProgram;
    cl_command_queue clQueue;
    cl_kernel fourierRowsAndTransposeKernel;
public:
    ImageProcessor();
    cv::Mat toSpectrum(Image image);
};