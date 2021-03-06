#pragma once

#include <OpenCL/opencl.h>

class OpenCLEnvironment {
public:
    cl_context clContext;
    cl_program clProgram;
    cl_command_queue clQueue;

    cl_kernel byteImageToComplexKernel;
    cl_kernel fourierColsAndTransposeKernel;
    cl_kernel complexImageToLogMagnitudeKernel;
    cl_kernel convolveKernel;
    cl_kernel grayscaleKernel;
    cl_kernel mirrorHorizontalKernel;
    cl_kernel mirrorVerticalKernel;
    cl_kernel rotate90Kernel;

    OpenCLEnvironment();
};