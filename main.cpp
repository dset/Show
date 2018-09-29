#include <opencv2/opencv.hpp>
#include "image.h"
#include "bmp.h"
#include <OpenCL/opencl.h>

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

    Image image = bmp::read(inf);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image.toMat());
    cv::waitKey(0);

    cv::Mat gray = image.toGreyscale().toMat();
    cv::imshow("Display Image", gray);
    cv::waitKey(0);

    //cv::Mat identity = image.identity();
    //cv::imshow("Display Image", identity);
    //cv::waitKey(0);

    cl_device_id devices[5];
    cl_uint numDevices = 0;
    clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 5, devices, &numDevices);

    for (int i = 0; i < numDevices; i++) {
        char name[128];
        size_t num = 0;
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, &num);
        std::cout << name << std::endl;
    }

    return 0;
}