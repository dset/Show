#include <utility>

#include "image.h"
#include "math.h"

Image::Image(unsigned int height, unsigned int width, unsigned int channels,
             std::vector<uint8_t> data)
        : height(height), width(width), channels(channels), data(std::move(data)) {}

unsigned int Image::getHeight() const {
    return height;
}

unsigned int Image::getWidth() const {
    return width;
}

uint8_t Image::at(unsigned int row, unsigned int col, unsigned int channel) const {
    return data[show::math::index(width, channels, row, col, channel)];
}

cv::Mat Image::createMat() const {
    return cv::Mat(height, width, CV_8UC(channels), (void *) data.data());
}