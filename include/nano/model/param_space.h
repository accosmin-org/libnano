#pragma once

#include <nano/arch.h>
#include <nano/tensor.h>

namespace nano
{
    class param_space_t;
    using param_spaces_t = std::vector<param_space_t>;

    ///
    /// \brief represent the space of values that a (hyper-)parameter can have with
    ///     support for mapping (scaling) these values so that the fitting the surrogate
    ///     model is feasible.
    ///
    class NANO_PUBLIC param_space_t
    {
    public:

        enum class type
        {
            log10,          ///<
            linear,         ///<
        };

        ///
        /// \brief constructor
        ///
        param_space_t(type, tensor1d_t grid_values);

        ///
        /// \brief
        ///
        scalar_t to_surrogate(scalar_t value) const;

        ///
        /// \brief
        ///
        scalar_t from_surrogate(scalar_t value) const;

        ///
        /// \brief
        ///
        scalar_t closest_grid_value_from_surrogate(scalar_t value) const;

    private:

        static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

        // attributes
        type            m_type{type::linear};   ///<
        tensor1d_t      m_grid_values;          ///<
        scalar_t        m_min{NaN}, m_max{NaN}; ///<
    };
}
