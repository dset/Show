#include "ImageProcessor.h"

ImageProcessor::ImageProcessor(OpenCLEnvironment& env): env(env) {}

cv::Mat ImageProcessor::createSpectrumMat(const Image &image) {
    size_t imageSize = image.height * image.width * image.channels;
    size_t byteImageSize = imageSize * sizeof(uint8_t);
    size_t complexImageSize = imageSize * sizeof(cl_float2);
    size_t floatImageSize = imageSize * sizeof(float);

    size_t dimensions[] = {image.height, image.width, image.channels};
    size_t transposedDimensions[] = {image.width, image.height, image.channels};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, byteImageSize, nullptr, nullptr);
    cl_mem cBuffer1 = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, complexImageSize, nullptr, nullptr);
    cl_mem cBuffer2 = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, complexImageSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, floatImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);

    clSetKernelArg(env.byteImageToComplexKernel, 0, sizeof(inputBuffer), &inputBuffer);
    clSetKernelArg(env.byteImageToComplexKernel, 1, sizeof(cBuffer1), &cBuffer1);
    clEnqueueNDRangeKernel(env.clQueue, env.byteImageToComplexKernel, 3, nullptr, dimensions, nullptr, 0, nullptr, nullptr);

    clSetKernelArg(env.fourierRowsAndTransposeKernel, 0, sizeof(cBuffer1), &cBuffer1);
    clSetKernelArg(env.fourierRowsAndTransposeKernel, 1, sizeof(cBuffer2), &cBuffer2);
    clEnqueueNDRangeKernel(env.clQueue, env.fourierRowsAndTransposeKernel, 3, nullptr, dimensions, nullptr, 0, nullptr, nullptr);

    clSetKernelArg(env.fourierRowsAndTransposeKernel, 0, sizeof(cBuffer2), &cBuffer2);
    clSetKernelArg(env.fourierRowsAndTransposeKernel, 1, sizeof(cBuffer1), &cBuffer1);
    clEnqueueNDRangeKernel(env.clQueue, env.fourierRowsAndTransposeKernel, 3, nullptr, transposedDimensions, nullptr, 0, nullptr, nullptr);

    clSetKernelArg(env.complexImageToLogMagnitudeKernel, 0, sizeof(cBuffer1), &cBuffer1);
    clSetKernelArg(env.complexImageToLogMagnitudeKernel, 1, sizeof(outputBuffer), &outputBuffer);
    clEnqueueNDRangeKernel(env.clQueue, env.complexImageToLogMagnitudeKernel, 3, nullptr, dimensions, nullptr, 0, nullptr, nullptr);

    cv::Mat res(image.height, image.width, CV_32FC(image.channels));
    clEnqueueReadBuffer(env.clQueue, outputBuffer, CL_FALSE, 0, floatImageSize, res.data, 0, nullptr, nullptr);

    clFinish(env.clQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(cBuffer1);
    clReleaseMemObject(cBuffer2);
    clReleaseMemObject(outputBuffer);

    cv::normalize(res, res, 0.0f, 1.0f, cv::NORM_MINMAX);

    return res;
}