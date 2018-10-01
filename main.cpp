#include <opencv2/opencv.hpp>
#include <OpenCL/opencl.h>
#include "bmp.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Give file name as first argument" << std::endl;
        return 1;
    }

    std::ifstream inf(argv[1], std::ios::binary);
    if (!inf) {
        std::cerr << "Unable to open file " << argv[1] << std::endl;
        return 1;
    }

    Image image = bmp::read(inf).toGreyscale();

    const unsigned int height = image.getHeight();
    const unsigned int width = image.getWidth();
    std::vector<cl_float2> input{};
    input.resize(height * width);
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            float real = image.at(row, col, 0);
            input[row * width + col] = {real, 0.0f};
        }
    }

    std::vector<cl_float2> output{};
    output.resize(input.size());
    size_t dataSize = (height * width) * sizeof(cl_float2);

    std::ifstream infk("../kernels/program.cl");
    std::stringstream stringbuffer;
    stringbuffer << infk.rdbuf();
    std::string kernelString = stringbuffer.str();
    const char *kernelSource = kernelString.c_str();

    cl_device_id devices[5];
    cl_uint numDevices = 0;
    clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 5, devices, &numDevices);

    cl_context context = clCreateContext(nullptr, 1, &devices[1], nullptr, nullptr, nullptr);
    cl_command_queue queue = clCreateCommandQueue(context, devices[1], 0, nullptr);

    cl_int programError = 0;
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &programError);
    std::cout << "Program error is " << programError << " and success is " << CL_SUCCESS << std::endl;

    cl_int buildError = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    std::cout << "Build error is " << buildError << std::endl;
    if (buildError == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, devices[1], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize, '\0');
        clGetProgramBuildInfo(program, devices[1], CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        std::cout << log << std::endl;
    }

    cl_int createKernelError = 0;
    cl_kernel rowsKernel = clCreateKernel(program, "ft2rows", &createKernelError);
    cl_kernel colsKernel = clCreateKernel(program, "ft2cols", &createKernelError);
    std::cout << "Create kernel error is " << createKernelError << std::endl;

    cl_int createBufferError = 0;
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, nullptr, &createBufferError);
    std::cout << "Create buffer error is " << createBufferError << std::endl;
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, nullptr, &createBufferError);
    std::cout << "Create buffer error is " << createBufferError << std::endl;

    cl_int enqueueError = clEnqueueWriteBuffer(queue, inputBuffer, CL_FALSE, 0, dataSize, input.data(), 0, nullptr, nullptr);
    std::cout << "Enqueue error is " << enqueueError << std::endl;

    cl_int setArgsError = clSetKernelArg(rowsKernel, 0, sizeof(inputBuffer), &inputBuffer);
    std::cout << "Set args error is " << setArgsError << std::endl;
    setArgsError = clSetKernelArg(rowsKernel, 1, sizeof(outputBuffer), &outputBuffer);
    std::cout << "Set args error is " << setArgsError << std::endl;

    size_t dimensions[] = {height, width, 0};
    cl_int enqueueKernelError = clEnqueueNDRangeKernel(queue, rowsKernel, 2, nullptr, dimensions, nullptr, 0, nullptr, nullptr);
    std::cout << "Enqueue kernel error is " << enqueueKernelError << std::endl;

    clSetKernelArg(colsKernel, 0, sizeof(outputBuffer), &outputBuffer);
    clSetKernelArg(colsKernel, 1, sizeof(inputBuffer), &inputBuffer);
    clEnqueueNDRangeKernel(queue, colsKernel, 2, nullptr, dimensions, nullptr, 0, nullptr, nullptr);

    cl_int enqueueReadError = clEnqueueReadBuffer(queue, inputBuffer, CL_FALSE, 0, dataSize, output.data(), 0, nullptr, nullptr);
    std::cout << "Enqueue read error is " << enqueueReadError << std::endl;

    cl_int finishError = clFinish(queue);
    std::cout << "Finish error is " << finishError << std::endl;

    std::vector<float> outputImage;
    outputImage.resize(output.size());
    float max = 0.0f;
    for (int i = 0; i < output.size(); i++) {
        outputImage[i] = std::log(std::sqrt(output[i].x * output[i].x + output[i].y * output[i].y));
        max = std::max(max, outputImage[i]);
    }

    for (auto& op : outputImage) {
        op = op / max;
    }

    cv::Mat outputMat(height, width, CV_32FC1, outputImage.data());
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", outputMat);
    cv::waitKey(0);

    for (int i = 0; i < numDevices; i++) {
        char name[128];
        size_t num = 0;
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, &num);
        std::cout << name << std::endl;
    }

    return 0;
}