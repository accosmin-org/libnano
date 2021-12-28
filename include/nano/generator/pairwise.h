#pragma once

#include <nano/generator/pairwise_base.h>
#include <nano/generator/pairwise_input.h>

namespace nano
{
    ///
    /// \brief interface for pair-wise feature generators.
    ///
    ///     new features are generated as a function of:
    ///         * original feature1,
    ///         * component index of the original feature1,
    ///         * original feature2,
    ///         * component index of the original feature2.
    ///
    template
    <
        typename tcomputer,
        std::enable_if_t<std::is_base_of_v<base_pairwise_generator_t, tcomputer>, bool> = true
    >
    class NANO_PUBLIC pairwise_generator_t : public tcomputer
    {
    public:

        static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

        using tcomputer::tcomputer;

        ///
        /// \brief @see generator_t
        ///
        void select(
            [[maybe_unused]] indices_cmap_t samples,
            [[maybe_unused]] tensor_size_t ifeature,
            [[maybe_unused]] scalar_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::scalar)
            {
                this->template iterate<tcomputer::input_rank1, tcomputer::input_rank2>(
                    samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature), [&] (auto it)
                {
                    if (this->should_drop(ifeature))
                    {
                        storage.full(NaN);
                    }
                    else
                    {
                        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                storage(index) = op(values1, values2);
                            }
                            else
                            {
                                storage(index) = NaN;
                            }
                        }
                    }
                });
            }
        }

        ///
        /// \brief @see generator_t
        ///
        void select(
            [[maybe_unused]] indices_cmap_t samples,
            [[maybe_unused]] tensor_size_t ifeature,
            [[maybe_unused]] sclass_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::sclass)
            {
                this->template iterate<tcomputer::input_rank1, tcomputer::input_rank2>(
                    samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature), [&] (auto it)
                {
                    if (this->should_drop(ifeature))
                    {
                        storage.full(-1);
                    }
                    else
                    {
                        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                storage(index) = op(values1, values2);
                            }
                            else
                            {
                                storage(index) = -1;
                            }
                        }
                    }
                });
            }
        }

        ///
        /// \brief @see generator_t
        ///
        void select(
            [[maybe_unused]] indices_cmap_t samples,
            [[maybe_unused]] tensor_size_t ifeature,
            [[maybe_unused]] mclass_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::mclass)
            {
                this->template iterate<tcomputer::input_rank1, tcomputer::input_rank2>(
                    samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature), [&] (auto it)
                {
                    if (this->should_drop(ifeature))
                    {
                        storage.full(-1);
                    }
                    else
                    {
                        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                op(values1, values2, storage.vector(index));
                            }
                            else
                            {
                                storage.vector(index).setConstant(-1);
                            }
                        }
                    }
                });
            }
        }

        ///
        /// \brief @see generator_t
        ///
        void select(
            [[maybe_unused]] indices_cmap_t samples,
            [[maybe_unused]] tensor_size_t ifeature,
            [[maybe_unused]] struct_map_t storage) const override
        {
            if constexpr (tcomputer::generated_type == generator_type::structured)
            {
                this->template iterate<tcomputer::input_rank1, tcomputer::input_rank2>(
                    samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature), [&] (auto it)
                {
                    if (this->should_drop(ifeature))
                    {
                        storage.full(NaN);
                    }
                    else
                    {
                        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                op(values1, values2, storage.vector(index));
                            }
                            else
                            {
                                storage.tensor(index).full(NaN);
                            }
                        }
                    }
                });
            }
        }

        ///
        /// \brief @see generator_t
        ///
        void flatten(
            [[maybe_unused]] indices_cmap_t samples,
            [[maybe_unused]] tensor2d_map_t storage,
            [[maybe_unused]] tensor_size_t column) const override
        {
            for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ ifeature)
            {
                this->template iterate<tcomputer::input_rank1, tcomputer::input_rank2>(
                    samples, ifeature, this->mapped_original1(ifeature), this->mapped_original2(ifeature), [&] (auto it)
                {
                    const auto should_drop = this->should_drop(ifeature);
                    const auto [op, colsize] = this->process(ifeature);

                    if (should_drop)
                    {
                        this->flatten_dropped(storage, column, colsize);
                    }
                    else
                    {
                        for (; it; ++ it)
                        {
                            if (const auto [index, given1, values1, given2, values2] = *it; given1 && given2)
                            {
                                if constexpr (tcomputer::generated_type == generator_type::scalar)
                                {
                                    storage(index, column) = op(values1, values2);
                                }
                                else
                                {
                                    auto segment = storage.vector(index).segment(column, colsize);
                                    if constexpr (tcomputer::generated_type == generator_type::sclass)
                                    {
                                        segment.setConstant(-1.0);
                                        segment(op(values1, values2)) = +1.0;
                                    }
                                    else if constexpr (tcomputer::generated_type == generator_type::mclass)
                                    {
                                        op(values1, values2, segment);
                                        segment.array() = 2.0 * segment.array() - 1.0;
                                    }
                                    else
                                    {
                                        op(values1, values2, segment);
                                    }
                                }
                            }
                            else
                            {
                                if constexpr (tcomputer::generated_type == generator_type::scalar)
                                {
                                    storage(index, column) = NaN;
                                }
                                else
                                {
                                    auto segment = storage.array(index).segment(column, colsize);
                                    segment.setConstant(NaN);
                                }
                            }
                        }
                    }
                    column += colsize;
                });
            }
        }
    };
}