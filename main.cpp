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

    Image image = bmp::read(inf);
    std::string name = "Display window";
    cv::namedWindow(name, cv::WINDOW_AUTOSIZE);

    OpenCLEnvironment env;
    ImageProcessor processor(env);

    int key = -1;
    while (key != 'q') {
        if (key == 's') {
            cv::imshow(name, processor.createSpectrumMat(image));
        } else if (key == 'b') {
            cv::imshow(name, processor.boxBlur(image, 11));
        } else if (key == 'e') {
            cv::imshow(name, processor.edges(image));
        } else if (key == 'g') {
            cv::imshow(name, processor.grayscale(image));
        } else if (key == 'h') {
            cv::imshow(name, processor.mirrorHorizontal(image));
        } else if (key == 'v') {
            cv::imshow(name, processor.mirrorVertical(image));
        } else if (key == 'r') {
            cv::imshow(name, processor.rotate90(image));
        } else {
            cv::imshow(name, image.createMat());
        }

        key = cv::waitKey(0);
    }

    return 0;
}