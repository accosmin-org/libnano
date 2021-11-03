#pragma once

#include <nano/generator/elemwise_base.h>

namespace nano
{
    struct input_sclass_t { static constexpr auto input_rank = 1U; };
    struct input_mclass_t { static constexpr auto input_rank = 2U; };
    struct input_scalar_t { static constexpr auto input_rank = 4U; };
    struct input_struct_t { static constexpr auto input_rank = 4U; };

    class NANO_PUBLIC elemwise_input_sclass_t : public input_sclass_t, public base_elemwise_generator_t
    {
    public:

        using base_elemwise_generator_t::base_elemwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC elemwise_input_mclass_t : public input_mclass_t, public base_elemwise_generator_t
    {
    public:

        using base_elemwise_generator_t::base_elemwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC elemwise_input_scalar_t : public input_scalar_t, public base_elemwise_generator_t
    {
    public:

        using base_elemwise_generator_t::base_elemwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC elemwise_input_struct_t : public input_struct_t, public base_elemwise_generator_t
    {
    public:

        using base_elemwise_generator_t::base_elemwise_generator_t;

        feature_mapping_t do_fit() override;
    };
}
