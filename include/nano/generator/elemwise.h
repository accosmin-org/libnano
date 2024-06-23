#pragma once

#include <nano/generator/elemwise_base.h>
#include <nano/generator/elemwise_input.h>

namespace nano
{
///
/// \brief interface for element-wise feature generators.
///
template <class tcomputer, std::enable_if_t<std::is_base_of_v<base_elemwise_generator_t, tcomputer>, bool> = true>
class NANO_PUBLIC elemwise_generator_t : public tcomputer
{
public:
    using tcomputer::tcomputer;

    ///
    /// \brief @see clonable_t
    ///
    rgenerator_t clone() const override { return std::make_unique<elemwise_generator_t<tcomputer>>(*this); }

    ///
    /// \brief @see generator_t
    ///
    void do_select([[maybe_unused]] indices_cmap_t samples, [[maybe_unused]] const tensor_size_t ifeature,
                   [[maybe_unused]] scalar_map_t storage) const override
    {
        if constexpr (tcomputer::generated_type == generator_type::scalar)
        {
            this->template iterate<tcomputer::input_rank>(samples, ifeature, this->mapped_original(ifeature),
                                                          [this, ifeature, &storage](auto it)
                                                          { this->select_scalar(ifeature, storage, it); });
        }
    }

    ///
    /// \brief @see generator_t
    ///
    void do_select([[maybe_unused]] indices_cmap_t samples, [[maybe_unused]] const tensor_size_t ifeature,
                   [[maybe_unused]] sclass_map_t storage) const override
    {
        if constexpr (tcomputer::generated_type == generator_type::sclass)
        {
            this->template iterate<tcomputer::input_rank>(samples, ifeature, this->mapped_original(ifeature),
                                                          [this, ifeature, &storage](auto it)
                                                          { this->select_sclass(ifeature, storage, it); });
        }
    }

    ///
    /// \brief @see generator_t
    ///
    void do_select([[maybe_unused]] indices_cmap_t samples, [[maybe_unused]] const tensor_size_t ifeature,
                   [[maybe_unused]] mclass_map_t storage) const override
    {
        if constexpr (tcomputer::generated_type == generator_type::mclass)
        {
            this->template iterate<tcomputer::input_rank>(samples, ifeature, this->mapped_original(ifeature),
                                                          [this, ifeature, &storage](auto it)
                                                          { this->select_mclass(ifeature, storage, it); });
        }
    }

    ///
    /// \brief @see generator_t
    ///
    void do_select([[maybe_unused]] indices_cmap_t samples, [[maybe_unused]] const tensor_size_t ifeature,
                   [[maybe_unused]] struct_map_t storage) const override
    {
        if constexpr (tcomputer::generated_type == generator_type::structured)
        {
            this->template iterate<tcomputer::input_rank>(samples, ifeature, this->mapped_original(ifeature),
                                                          [this, ifeature, &storage](auto it)
                                                          { this->select_struct(ifeature, storage, it); });
        }
    }

    ///
    /// \brief @see generator_t
    ///
    void flatten([[maybe_unused]] indices_cmap_t samples, [[maybe_unused]] tensor2d_map_t storage,
                 [[maybe_unused]] tensor_size_t column) const override
    {
        for (tensor_size_t ifeature = 0, features = this->features(); ifeature < features; ++ifeature)
        {
            this->template iterate<tcomputer::input_rank>(samples, ifeature, this->mapped_original(ifeature),
                                                          [this, ifeature, &column, &storage](auto it)
                                                          {
                                                              const auto should_drop   = this->should_drop(ifeature);
                                                              const auto [op, colsize] = this->process(ifeature);

                                                              if (should_drop)
                                                              {
                                                                  this->flatten_dropped(storage, column, colsize);
                                                              }
                                                              else
                                                              {
                                                                  this->flatten(storage, column, op, colsize, it);
                                                              }
                                                              column += colsize;
                                                          });
        }
    }

private:
    template <class titerator>
    void select_scalar(const tensor_size_t ifeature, const scalar_map_t& storage, titerator it) const
    {
        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
        for (; it; ++it)
        {
            if (const auto [index, given, values] = *it; given)
            {
                storage(index) = op(values);
            }
            else
            {
                storage(index) = this->NaN;
            }
        }
    }

    template <class titerator>
    void select_sclass(const tensor_size_t ifeature, const sclass_map_t& storage, titerator it) const
    {
        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
        for (; it; ++it)
        {
            if (const auto [index, given, values] = *it; given)
            {
                storage(index) = op(values);
            }
            else
            {
                storage(index) = -1;
            }
        }
    }

    template <class titerator>
    void select_mclass(const tensor_size_t ifeature, const mclass_map_t& storage, titerator it) const
    {
        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
        for (; it; ++it)
        {
            if (const auto [index, given, values] = *it; given)
            {
                op(values, storage.vector(index));
            }
            else
            {
                storage.vector(index).setConstant(-1);
            }
        }
    }

    template <class titerator>
    void select_struct(const tensor_size_t ifeature, const struct_map_t& storage, titerator it) const
    {
        [[maybe_unused]] const auto [op, colsize] = this->process(ifeature);
        for (; it; ++it)
        {
            if (const auto [index, given, values] = *it; given)
            {
                op(values, storage.vector(index));
            }
            else
            {
                storage.tensor(index).full(this->NaN);
            }
        }
    }

    template <class toperator, class titerator>
    void flatten(const tensor2d_map_t& storage, const tensor_size_t column, const toperator& op,
                 [[maybe_unused]] const tensor_size_t colsize, titerator it) const
    {
        for (; it; ++it)
        {
            if (const auto [index, given, values] = *it; given)
            {
                if constexpr (tcomputer::generated_type == generator_type::scalar)
                {
                    storage(index, column) = op(values);
                }
                else
                {
                    auto segment = storage.vector(index).segment(column, colsize);
                    if constexpr (tcomputer::generated_type == generator_type::sclass)
                    { // NOLINT(bugprone-branch-clone)
                        segment.setConstant(-1.0);
                        const auto class_index = op(values);
                        if (class_index < segment.size())
                        {
                            segment(class_index) = +1.0;
                        }
                    }
                    else if constexpr (tcomputer::generated_type == generator_type::mclass)
                    {
                        op(values, segment);
                        segment.array() = 2.0 * segment.array() - 1.0;
                    }
                    else
                    {
                        op(values, segment);
                    }
                }
            }
            else
            {
                if constexpr (tcomputer::generated_type == generator_type::scalar)
                {
                    storage(index, column) = this->NaN;
                }
                else
                {
                    auto segment = storage.array(index).segment(column, colsize);
                    segment.setConstant(this->NaN);
                }
            }
        }
    }
};
} // namespace nano
