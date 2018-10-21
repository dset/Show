#include "ImageProcessor.h"

void callKernel(cl_command_queue& queue, cl_kernel& kernel, const std::vector<size_t>& dimensions, cl_uint) {
    clEnqueueNDRangeKernel(queue, kernel, static_cast<cl_uint>(dimensions.size()),
            nullptr, dimensions.data(), nullptr, 0, nullptr, nullptr);
}

template <typename T, typename... Args>
void callKernel(cl_command_queue& queue, cl_kernel& kernel, const std::vector<size_t>& dimensions, cl_uint narg, T& arg1, Args&... args) {
    clSetKernelArg(kernel, narg, sizeof(arg1), &arg1);
    callKernel(queue, kernel, dimensions, narg + 1, args...);
}

cv::Mat ImageProcessor::createSpectrumMat(const Image &image) {
    size_t imageSize = image.height * image.width * image.channels;
    size_t byteImageSize = imageSize * sizeof(uint8_t);
    size_t complexImageSize = imageSize * sizeof(cl_float2);
    size_t floatImageSize = imageSize * sizeof(float);

    std::vector<size_t> dims{image.height, image.width, image.channels};
    std::vector<size_t> trans_dims{image.width, image.height, image.channels};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, byteImageSize, nullptr, nullptr);
    cl_mem cBuffer1 = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, complexImageSize, nullptr, nullptr);
    cl_mem cBuffer2 = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, complexImageSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, floatImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);

    callKernel(env.clQueue, env.byteImageToComplexKernel, dims, 0, inputBuffer, cBuffer1);
    callKernel(env.clQueue, env.fourierColsAndTransposeKernel, dims, 0, cBuffer1, cBuffer2);
    callKernel(env.clQueue, env.fourierColsAndTransposeKernel, trans_dims, 0, cBuffer2, cBuffer1);
    callKernel(env.clQueue, env.complexImageToLogMagnitudeKernel, dims, 0, cBuffer1, outputBuffer);

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

cv::Mat ImageProcessor::boxBlur(const Image &image, unsigned int size) {
    std::vector<float> kernel;
    kernel.reserve(size * size);
    for (int i = 0; i < size * size; i++) {
        kernel.push_back(1.0f / (size * size));
    }
    return convolve(image, kernel, size);
}

cv::Mat ImageProcessor::edges(const Image &image) {
    std::vector<float> kernel = {-1.0f, -1.0f, -1.0f, -1.0f, 8.0f, -1.0f, -1.0f, -1.0f, -1.0f};
    return convolve(image, kernel, 3);
}

cv::Mat ImageProcessor::convolve(const Image &image, const std::vector<float>& kernel, int kernelHeight) {
    size_t imageSize = image.height * image.width * image.channels;
    size_t byteImageSize = imageSize * sizeof(uint8_t);

    size_t kernelSize = kernel.size() * sizeof(float);
    cl_int kHeight = kernelHeight;
    cl_int kWidth = (int) kernel.size() / kernelHeight;

    std::vector<size_t> dims = {image.height, image.width, image.channels};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_ONLY, byteImageSize, nullptr, nullptr);
    cl_mem kernelBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_ONLY, kernelSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, byteImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(env.clQueue, kernelBuffer, CL_FALSE, 0, kernelSize, kernel.data(), 0, nullptr, nullptr);

    callKernel(env.clQueue, env.convolveKernel, dims, 0, inputBuffer, kernelBuffer, kHeight, kWidth, outputBuffer);

    cv::Mat res(image.height, image.width, CV_8UC(image.channels));
    clEnqueueReadBuffer(env.clQueue, outputBuffer, CL_FALSE, 0, byteImageSize, res.data, 0, nullptr, nullptr);

    clFinish(env.clQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(kernelBuffer);
    clReleaseMemObject(outputBuffer);

    return res;
}

cv::Mat ImageProcessor::grayscale(const Image &image) {
    size_t byteImageSize = image.height * image.width * image.channels * sizeof(uint8_t);
    size_t outImageSize = image.height * image.width * sizeof(uint8_t);

    std::vector<size_t> dims = {image.height, image.width};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_ONLY, byteImageSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, outImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);

    callKernel(env.clQueue, env.grayscaleKernel, dims, 0, inputBuffer, image.channels, outputBuffer);

    cv::Mat res(image.height, image.width, CV_8UC1);
    clEnqueueReadBuffer(env.clQueue, outputBuffer, CL_FALSE, 0, outImageSize, res.data, 0, nullptr, nullptr);

    clFinish(env.clQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);

    return res;
}

cv::Mat ImageProcessor::mirror(const Image &image, cl_kernel mirrorKernel) {
    size_t byteImageSize = image.height * image.width * image.channels * sizeof(uint8_t);

    std::vector<size_t> dims = {image.height, image.width, image.channels};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_ONLY, byteImageSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, byteImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);

    callKernel(env.clQueue, mirrorKernel, dims, 0, inputBuffer, outputBuffer);

    cv::Mat res(image.height, image.width, CV_8UC(image.channels));
    clEnqueueReadBuffer(env.clQueue, outputBuffer, CL_FALSE, 0, byteImageSize, res.data, 0, nullptr, nullptr);

    clFinish(env.clQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);

    return res;
}

cv::Mat ImageProcessor::mirrorHorizontal(const Image &image) {
    return mirror(image, env.mirrorHorizontalKernel);
}

cv::Mat ImageProcessor::mirrorVertical(const Image &image) {
    return mirror(image, env.mirrorVerticalKernel);
}

cv::Mat ImageProcessor::rotate90(const Image &image) {
    size_t byteImageSize = image.height * image.width * image.channels * sizeof(uint8_t);

    std::vector<size_t> dims = {image.height, image.width, image.channels};

    cl_mem inputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_ONLY, byteImageSize, nullptr, nullptr);
    cl_mem outputBuffer = clCreateBuffer(env.clContext, CL_MEM_READ_WRITE, byteImageSize, nullptr, nullptr);

    clEnqueueWriteBuffer(env.clQueue, inputBuffer, CL_FALSE, 0, byteImageSize, image.data.data(), 0, nullptr, nullptr);

    callKernel(env.clQueue, env.rotate90Kernel, dims, 0, inputBuffer, outputBuffer);

    cv::Mat res(image.width, image.height, CV_8UC(image.channels));
    clEnqueueReadBuffer(env.clQueue, outputBuffer, CL_FALSE, 0, byteImageSize, res.data, 0, nullptr, nullptr);

    clFinish(env.clQueue);

    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);

    return res;
}