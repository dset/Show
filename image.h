#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

class Image {
private:
    unsigned int height;
    unsigned int width;
    unsigned int channels;
    std::vector<uint8_t> data;

public:
    Image(unsigned int height, unsigned int width, unsigned int channels,
          std::vector<uint8_t> data);

    unsigned int getHeight();

    unsigned int getWidth();

    uint8_t at(unsigned int row, unsigned int col, unsigned int channel);

    Image toGreyscale();

    cv::Mat toMat();

    cv::Mat ft2();

    cv::Mat identity();
};