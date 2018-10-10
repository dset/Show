#include "bmp.h"
#include "math.h"

namespace bmp {
    struct __attribute__((packed)) BmpHeader {
        uint16_t signature;
        uint32_t size;
        uint16_t res1;
        uint16_t res2;
        uint32_t offset;
    };

    struct __attribute__((packed)) DibHeader {
        uint32_t size;
        int32_t width;
        int32_t height;
        uint16_t numColorPlanes;
        uint16_t numBitsPerPixel;
        uint32_t compression;
        uint32_t imageSize;
        int32_t horResolution;
        int32_t verResolution;
        uint32_t numColors;
        uint32_t numImportantColors;
    };

    Image read(std::ifstream& in) {
        BmpHeader header{};
        DibHeader dibHeader{};
        in.read((char *) &header, sizeof(header));
        in.read((char *) &dibHeader, sizeof(dibHeader));

        if (dibHeader.size != 40) {
            throw std::invalid_argument("Unsupported DIB type.");
        }

        int skip = header.offset - header.size - dibHeader.size;
        in.ignore(skip);

        unsigned int channels = dibHeader.numBitsPerPixel / 8u;
        int rowSize = ((dibHeader.width * dibHeader.numBitsPerPixel + 31) / 32) * 4;
        int rowSkip = rowSize - (dibHeader.width * dibHeader.numBitsPerPixel / 8);

        std::vector<uint8_t> data;
        data.resize(dibHeader.height * dibHeader.width * channels);

        uint8_t pixel;
        for (unsigned int row = 0; row < dibHeader.height; row++) {
            for (unsigned int col = 0; col < dibHeader.width; col++) {
                for (unsigned int channel = 0; channel < channels; channel++) {
                    in.read((char *) &pixel, 1);
                    unsigned int index = show::math::index(dibHeader.width, channels,
                                                      dibHeader.height - 1 - row, col, channel);
                    data[index] = pixel;
                }
            }

            in.ignore(rowSkip);
        }

        return Image(dibHeader.height, dibHeader.width, channels, std::move(data));
    }
}