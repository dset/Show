#include <iostream>
#include "math.h"

namespace show {
    namespace math {
        std::vector<std::complex<double>>
        ft2(std::vector<std::complex<double>> x, unsigned int height,
            unsigned int width) {
            const double pi = std::acos(-1);

            std::vector<std::complex<double>> intermediate;
            intermediate.resize(height * width);

            for (unsigned int row = 0; row < height; row++) {
                std::cout << "Doing row " << row << std::endl;

                for (unsigned int k = 0; k < width; k++) {
                    std::complex<double> sum(0.0, 0.0);
                    for (unsigned int n = 0; n < width; n++) {
                        std::complex<double> xn = x[index(width, 1, row, n, 0)];
                        double angle = 2 * pi * k * n / width;
                        sum += xn * std::complex<double>(std::cos(angle), -std::sin(angle));
                    }

                    intermediate[index(width, 1, row, k, 0)] = sum;
                }
            }

            std::vector<std::complex<double>> freq;
            freq.resize(height * width);

            for (unsigned int col = 0; col < width; col++) {
                std::cout << "Doing col " << col << std::endl;

                for (unsigned int k = 0; k < height; k++) {
                    std::complex<double> sum(0.0, 0.0);
                    for (unsigned int n = 0; n < height; n++) {
                        std::complex<double> xn = intermediate[index(width, 1, n, col, 0)];
                        double angle = 2 * pi * k * n / height;
                        sum += xn * std::complex<double>(std::cos(angle), -std::sin(angle));
                    }

                    freq[index(width, 1, k, col, 0)] = sum;
                }
            }

            return freq;
        }

        unsigned int
        index(unsigned int width, unsigned int channels, unsigned int row, unsigned int col,
              unsigned int channel) {
            return row * width * channels + col * channels + channel;
        }
    }
}