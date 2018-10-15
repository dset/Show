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

    //Image image(2, 2, 3, {10, 10, 10, 2, 2, 2, 3, 3, 3, 4, 4, 4});
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
        } else if (key == 'e') {
            cv::imshow("Display window", processor.edges(image));
        } else if (key == 'g') {
            cv::imshow("Display window", processor.grayscale(image));
        } else {
            cv::imshow("Display window", image.createMat());
        }

        key = cv::waitKey(0);
    }

    return 0;
}