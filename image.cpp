#include "image.h"
#include "math.h"

Image::Image(unsigned int height, unsigned int width, unsigned int channels,
             std::vector<uint8_t> data)
        : height(height), width(width), channels(channels), data(std::move(data)) {}

cv::Mat Image::toMat() {
    return cv::Mat(height, width, CV_8UC(channels), data.data());
}

uint8_t Image::at(unsigned int row, unsigned int col, unsigned int channel) {
    return data[show::math::index(width, channels, row, col, channel)];
}

Image Image::toGreyscale() {
    std::vector<uint8_t> res;
    res.resize(height * width);

    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            uint8_t pixel = 0;
            for (unsigned int channel = 0; channel < channels; channel++) {
                pixel += at(row, col, channel) / channels;
            }
            res[show::math::index(width, 1, row, col, 0)] = pixel;
        }
    }

    return Image(height, width, 1, res);
}

cv::Mat Image::ft2() {
    if (channels > 1) {
        throw std::invalid_argument("FT on multi-channel images is not supported");
    }

    std::vector<std::complex<double>> x;
    x.resize(height * width);
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            x[show::math::index(width, 1, row, col, 0)] = at(row, col, 0);
        }
    }

    std::vector<std::complex<double>> freq = show::math::ft2(x, height, width);
    std::vector<double> magnitudes;
    magnitudes.resize(freq.size());
    double max = magnitudes[0];
    double min = magnitudes[0];
    for (int i = 0; i < magnitudes.size(); i++) {
        magnitudes[i] = std::log(std::abs(freq[i]));
        max = std::max(max, magnitudes[i]);
        min = std::min(min, magnitudes[i]);
    }

    for (double &magnitude : magnitudes) {
        magnitude = (magnitude - min) / (max - min);
    }

    return cv::Mat(height, width, CV_64FC1, magnitudes.data());
}

cv::Mat Image::identity() {
    std::vector<std::complex<double>> x;
    x.resize(height * width);
    for (unsigned int row = 0; row < height; row++) {
        for (unsigned int col = 0; col < width; col++) {
            x[show::math::index(width, 1, row, col, 0)] = at(row, col, 0);
        }
    }

    std::vector<std::complex<double>> freq = show::math::ft2(x, height, width);
    std::vector<std::complex<double>> time = show::math::ft2(freq, height, width);
    std::vector<std::complex<double>> freq2 = show::math::ft2(time, height, width);
    std::vector<std::complex<double>> time2 = show::math::ft2(freq2, height, width);

    std::vector<double> real;
    real.resize(height * width);

    for (int i = 0; i < time2.size(); i++) {
        real[i] = time2[i].real();
        std::cout << real[i] << ", ";
        if ((i + 1) % 500 == 0) {
            std::cout << std::endl;
        }
    }

    return cv::Mat(height, width, CV_64FC1, real.data());
}