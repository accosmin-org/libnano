#pragma once

#include <memory>

namespace nano
{
    template <typename tbase, typename tderived, std::enable_if_t<std::is_base_of_v<tbase, tderived>, bool> = true>
    class clonable_t
    {
    public:

        virtual std::unique_ptr<tbase> clone() const
        {
           return std::make_unique<tderived>(*this);
        }
    };
}
