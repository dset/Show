#include "OpenCLEnvironment.h"
#include <fstream>
#include <sstream>
#include <iostream>

OpenCLEnvironment::OpenCLEnvironment() {
    std::ifstream infk("../kernels/program.cl");
    std::stringstream stringbuffer;
    stringbuffer << infk.rdbuf();
    std::string kernelString = stringbuffer.str();
    const char *kernelSource = kernelString.c_str();

    cl_device_id devices[5];
    cl_uint numDevices = 0;
    clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 5, devices, &numDevices);

    clContext = clCreateContext(nullptr, 1, &devices[1], nullptr, nullptr, nullptr);
    clQueue = clCreateCommandQueue(clContext, devices[1], 0, nullptr);
    clProgram = clCreateProgramWithSource(clContext, 1, &kernelSource, nullptr, nullptr);

    cl_int buildError = clBuildProgram(clProgram, 0, nullptr, nullptr, nullptr, nullptr);
    if (buildError == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize = 0;
        clGetProgramBuildInfo(clProgram, devices[1], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(clProgram, devices[1], CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cout << log << std::endl;
    }

    byteImageToComplexKernel = clCreateKernel(clProgram, "byteImageToComplex", nullptr);
    fourierColsAndTransposeKernel = clCreateKernel(clProgram, "fourierColsAndTranspose", nullptr);
    complexImageToLogMagnitudeKernel = clCreateKernel(clProgram, "complexImageToLogMagnitude", nullptr);
    convolveKernel = clCreateKernel(clProgram, "convolve", nullptr);
}