#include <opencv2/opencv.hpp>
#include "bmp.h"
#include "ImageProcessor.h"

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

    //Image image(2, 2, 3, {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
    Image image = bmp::read(inf);
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);

    OpenCLEnvironment env;
    ImageProcessor processor(env);

    int key = -1;
    while (key != 'q') {
        if (key == 's') {
            cv::imshow("Display window", processor.createSpectrumMat(image));
        } else if (key == 'b') {
            cv::imshow("Display window", processor.boxBlur(image, 11));
        } else {
            cv::imshow("Display window", image.createMat());
        }

        key = cv::waitKey(0);
    }

    return 0;
}