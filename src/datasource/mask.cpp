#include <nano/datasource/mask.h>

using namespace nano;

bool nano::optional(const mask_cmap_t& mask, tensor_size_t samples)
{
    const auto bytes = samples / 8;
    for (tensor_size_t byte = 0; byte < bytes; ++byte)
    {
        if (mask(byte) != 0xFF)
        {
            return true;
        }
    }
    for (tensor_size_t sample = 8 * bytes; sample < samples; ++sample)
    {
        if (!getbit(mask, sample))
        {
            return true;
        }
    }
    return false;
}
