#pragma once

#include <vector>
#include <complex>

namespace show {
    namespace math {
        std::vector<std::complex<double>>
        ft2(std::vector<std::complex<double>> x, unsigned int height, unsigned int width);

        unsigned int
        index(unsigned int width, unsigned int channels, unsigned int row, unsigned int col,
              unsigned int channel);
    }
}