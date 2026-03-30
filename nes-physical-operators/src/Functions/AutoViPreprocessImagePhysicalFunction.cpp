/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <Functions/AutoViPreprocessImagePhysicalFunction.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <png.h>
#include <scope_guard.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <PhysicalFunctionRegistry.hpp>
#include <val.hpp>

namespace NES
{

AutoViPreprocessImagePhysicalFunction::AutoViPreprocessImagePhysicalFunction(PhysicalFunction childPhysicalFunction)
    : childPhysicalFunction(std::move(childPhysicalFunction))
{
}

namespace
{
constexpr uint32_t modelExtent = 64;
constexpr uint32_t modelChannels = 3;
constexpr uint64_t outputTensorBytes = uint64_t{1} * modelChannels * modelExtent * modelExtent * sizeof(float);

struct DecodedRgbImage
{
    png_uint_32 width;
    png_uint_32 height;
    std::vector<uint8_t> pixels;
};

DecodedRgbImage decodePngToRgb(std::span<const std::byte> pngBytes)
{
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, pngBytes.data(), pngBytes.size()))
    {
        throw FormattingError("AUTOVI_PREPROCESS_IMAGE could not read PNG header: {}", image.message);
    }
    SCOPE_EXIT
    {
        png_image_free(&image);
    };

    image.format = PNG_FORMAT_RGB;
    std::vector<uint8_t> pixels(PNG_IMAGE_SIZE(image));
    if (!png_image_finish_read(&image, nullptr, pixels.data(), 0, nullptr))
    {
        throw FormattingError("AUTOVI_PREPROCESS_IMAGE could not decode PNG payload: {}", image.message);
    }

    return {.width = image.width, .height = image.height, .pixels = std::move(pixels)};
}

float sampleRgbChannel(const DecodedRgbImage& image, const float srcX, const float srcY, const size_t channel)
{
    const auto maxX = static_cast<int>(image.width) - 1;
    const auto maxY = static_cast<int>(image.height) - 1;
    const auto x0 = std::clamp(static_cast<int>(std::floor(srcX)), 0, maxX);
    const auto y0 = std::clamp(static_cast<int>(std::floor(srcY)), 0, maxY);
    const auto x1 = std::clamp(x0 + 1, 0, maxX);
    const auto y1 = std::clamp(y0 + 1, 0, maxY);

    const auto wx = srcX - static_cast<float>(x0);
    const auto wy = srcY - static_cast<float>(y0);

    const auto pixel = [&](const int x, const int y)
    {
        const auto offset = (static_cast<size_t>(y) * image.width + static_cast<size_t>(x)) * modelChannels + channel;
        return static_cast<float>(image.pixels[offset]);
    };

    const auto top = pixel(x0, y0) * (1.0F - wx) + pixel(x1, y0) * wx;
    const auto bottom = pixel(x0, y1) * (1.0F - wx) + pixel(x1, y1) * wx;
    return top * (1.0F - wy) + bottom * wy;
}

void preprocessPngToAutoViTensor(int8_t* inputPtr, uint64_t inputSize, int8_t* outputPtr)
{
    const auto image = decodePngToRgb(std::span<const std::byte>(reinterpret_cast<const std::byte*>(inputPtr), inputSize));

    for (uint32_t y = 0; y < modelExtent; ++y)
    {
        const auto srcY = ((static_cast<float>(y) + 0.5F) * static_cast<float>(image.height) / static_cast<float>(modelExtent)) - 0.5F;
        for (uint32_t x = 0; x < modelExtent; ++x)
        {
            const auto srcX = ((static_cast<float>(x) + 0.5F) * static_cast<float>(image.width) / static_cast<float>(modelExtent)) - 0.5F;
            const auto baseIndex = static_cast<size_t>(y) * modelExtent + x;

            for (size_t channel = 0; channel < modelChannels; ++channel)
            {
                const auto sampled = sampleRgbChannel(image, srcX, srcY, channel);
                const auto value = sampled / 255.0F;
                const auto tensorIndex = channel * modelExtent * modelExtent + baseIndex;
                std::memcpy(outputPtr + tensorIndex * sizeof(float), &value, sizeof(value));
            }
        }
    }
}
}

VarVal AutoViPreprocessImagePhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto inputValue = childPhysicalFunction.execute(record, arena).getRawValueAs<VariableSizedData>();
    auto output = arena.allocateVariableSizedData(nautilus::val<uint64_t>(outputTensorBytes));
    nautilus::invoke(preprocessPngToAutoViTensor, inputValue.getContent(), inputValue.getSize(), output.getContent());
    return VariableSizedData(output.getContent(), nautilus::val<uint64_t>(outputTensorBytes));
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterAUTOVI_PREPROCESS_IMAGEPhysicalFunction(PhysicalFunctionRegistryArguments args)
{
    PRECONDITION(args.childFunctions.size() == 1, "AUTOVI_PREPROCESS_IMAGE must have exactly one child function");
    return AutoViPreprocessImagePhysicalFunction(args.childFunctions[0]);
}
}
