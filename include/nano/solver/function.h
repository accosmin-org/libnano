#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief wrapper over function_t to keep track of the number of function value and gradient calls.
    ///
    class solver_function_t final : public function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        explicit solver_function_t(const function_t& function) :
            function_t(function),
            m_function(function)
        {
        }

        ///
        /// \brief @see function_t
        ///
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr, vgrad_config_t config = vgrad_config_t{}) const override
        {
            const auto calls = (config.m_summand < 0) ? 1.0 : (1.0 / static_cast<tensor_size_t>(summands()));

            m_fcalls += calls;
            m_gcalls += (gx != nullptr) ? calls : 0.0;
            return m_function.vgrad(x, gx);
        }

        ///
        /// \brief number of function evaluation calls.
        ///
        auto fcalls() const { return static_cast<tensor_size_t>(m_fcalls + 0.5); }

        ///
        /// \brief number of function gradient calls.
        ///
        auto gcalls() const { return static_cast<tensor_size_t>(m_gcalls + 0.5); }

    private:

        // attributes
        const function_t&       m_function;             ///<
        mutable scalar_t        m_fcalls{0.0};          ///< #function value evaluations
        mutable scalar_t        m_gcalls{0.0};          ///< #function gradient evaluations
    };
}
