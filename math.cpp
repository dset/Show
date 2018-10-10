#include "math.h"

namespace show {
    namespace math {
        unsigned int
        index(unsigned int width, unsigned int channels, unsigned int row, unsigned int col,
              unsigned int channel) {
            return row * width * channels + col * channels + channel;
        }
    }
}