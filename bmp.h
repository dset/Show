#pragma once

#include <fstream>
#include "image.h"

namespace bmp {
    Image read(std::ifstream &in);
}