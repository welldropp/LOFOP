// Image construction and file loading for the LOFOP C++ SDK.
//
// Decoding is not LOFOP's business: production callers almost always already
// hold pixels (a decoded video frame, a cv::Mat, a camera buffer) and should
// use Image::from_pixels, which copies nothing but the bytes. For the cases
// that do need a file, binary PPM/PGM is supported with no dependencies --
// enough for tests, tooling, and pipelines that dump raw frames -- and
// building with LOFOP_WITH_STB_IMAGE adds every format stb_image handles.

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "lofop/lofop.hpp"

#ifdef LOFOP_WITH_STB_IMAGE
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

namespace lofop {
namespace {

// Read the next whitespace-delimited token of a NetPBM header, skipping the
// '#' comments the format allows between any two tokens.
bool next_header_token(std::istream& stream, std::string& token) {
    token.clear();
    int ch = stream.get();
    while (stream) {
        if (std::isspace(ch) != 0) {
            ch = stream.get();
            continue;
        }
        if (ch == '#') {
            while (stream && ch != '\n') {
                ch = stream.get();
            }
            continue;
        }
        break;
    }
    while (stream && std::isspace(ch) == 0) {
        token.push_back(static_cast<char>(ch));
        ch = stream.get();
    }
    return !token.empty();
}

Image load_netpbm(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw Error("Cannot open image file: " + path);
    }
    std::string magic;
    if (!next_header_token(stream, magic) || (magic != "P6" && magic != "P5")) {
        throw Error(
            "Unsupported image format: " + path +
            " (this build reads binary PPM/PGM; rebuild with LOFOP_WITH_STB_IMAGE "
            "for JPEG/PNG, or decode it yourself and use Image::from_pixels)");
    }
    const int channels = (magic == "P6") ? 3 : 1;
    std::string width_token;
    std::string height_token;
    std::string maxval_token;
    if (!next_header_token(stream, width_token) || !next_header_token(stream, height_token) ||
        !next_header_token(stream, maxval_token)) {
        throw Error("Truncated image header: " + path);
    }
    int width = 0;
    int height = 0;
    int maxval = 0;
    try {
        width = std::stoi(width_token);
        height = std::stoi(height_token);
        maxval = std::stoi(maxval_token);
    } catch (const std::exception&) {
        throw Error("Malformed image header: " + path);
    }
    if (width <= 0 || height <= 0) {
        throw Error("Image has non-positive dimensions: " + path);
    }
    if (maxval != 255) {
        throw Error("Only 8-bit images are supported (maxval 255): " + path);
    }
    // Exactly one whitespace character separates the header from the payload.
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channels;
    std::vector<uint8_t> pixels(count);
    stream.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(count));
    if (static_cast<std::size_t>(stream.gcount()) != count) {
        throw Error("Truncated image data: " + path);
    }
    return Image(width, height, channels, std::move(pixels));
}

}  // namespace

Image::Image(int width, int height, int channels, std::vector<uint8_t> pixels)
    : width_(width), height_(height), channels_(channels), pixels_(std::move(pixels)) {
    if (width_ <= 0 || height_ <= 0) {
        throw Error("Image needs positive dimensions");
    }
    if (channels_ != 1 && channels_ != 3) {
        throw Error("Image supports 1 or 3 channels");
    }
    const std::size_t expected =
        static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * channels_;
    if (pixels_.size() != expected) {
        std::ostringstream message;
        message << "Pixel buffer does not match the declared image size (expected " << expected
                << ", got " << pixels_.size() << ")";
        throw Error(message.str());
    }
}

Image Image::from_pixels(const uint8_t* pixels, int width, int height, int channels) {
    if (pixels == nullptr) {
        throw Error("Image::from_pixels received a null buffer");
    }
    if (width <= 0 || height <= 0) {
        throw Error("Image needs positive dimensions");
    }
    if (channels != 1 && channels != 3) {
        throw Error("Image supports 1 or 3 channels");
    }
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * channels;
    return Image(width, height, channels, std::vector<uint8_t>(pixels, pixels + count));
}

Image Image::load(const std::string& path) {
#ifdef LOFOP_WITH_STB_IMAGE
    int width = 0;
    int height = 0;
    int source_channels = 0;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &source_channels, 3);
    if (data != nullptr) {
        const std::size_t count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;
        std::vector<uint8_t> pixels(data, data + count);
        stbi_image_free(data);
        return Image(width, height, 3, std::move(pixels));
    }
    // stb could not decode it; the NetPBM reader gives a clearer diagnostic.
#endif
    return load_netpbm(path);
}

}  // namespace lofop
