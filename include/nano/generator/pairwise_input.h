#pragma once

#include <nano/generator/pairwise_base.h>

namespace nano
{
    struct input1_sclass_t { static constexpr auto input_rank1 = 1U; };
    struct input1_mclass_t { static constexpr auto input_rank1 = 2U; };
    struct input1_scalar_t { static constexpr auto input_rank1 = 4U; };
    struct input1_struct_t { static constexpr auto input_rank1 = 4U; };

    struct input2_sclass_t { static constexpr auto input_rank2 = 1U; };
    struct input2_mclass_t { static constexpr auto input_rank2 = 2U; };
    struct input2_scalar_t { static constexpr auto input_rank2 = 4U; };
    struct input2_struct_t { static constexpr auto input_rank2 = 4U; };

    class NANO_PUBLIC pairwise_input_sclass_sclass_t : public input1_sclass_t, public input2_sclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_sclass_mclass_t : public input1_sclass_t, public input2_mclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_sclass_scalar_t : public input1_sclass_t, public input2_scalar_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_sclass_struct_t : public input1_sclass_t, public input2_struct_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_mclass_sclass_t : public input1_mclass_t, public input2_sclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_mclass_mclass_t : public input1_mclass_t, public input2_mclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_mclass_scalar_t : public input1_mclass_t, public input2_scalar_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_mclass_struct_t : public input1_mclass_t, public input2_struct_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_scalar_sclass_t : public input1_scalar_t, public input2_sclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_scalar_mclass_t : public input1_scalar_t, public input2_mclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_scalar_scalar_t : public input1_scalar_t, public input2_scalar_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_scalar_struct_t : public input1_scalar_t, public input2_struct_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_struct_sclass_t : public input1_struct_t, public input2_sclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_struct_mclass_t : public input1_struct_t, public input2_mclass_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_struct_scalar_t : public input1_struct_t, public input2_scalar_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };

    class NANO_PUBLIC pairwise_input_struct_struct_t : public input1_struct_t, public input2_struct_t, public base_pairwise_generator_t
    {
    public:

        using base_pairwise_generator_t::base_pairwise_generator_t;

        feature_mapping_t do_fit() override;
    };
}
