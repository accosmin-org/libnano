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
        scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const override
        {
            m_fcalls += 1;
            m_gcalls += (gx != nullptr) ? 1 : 0;
            return m_function.vgrad(x, gx);
        }

        ///
        /// \brief number of function evaluation calls
        ///
        auto fcalls() const { return m_fcalls; }

        ///
        /// \brief number of function gradient calls
        ///
        auto gcalls() const { return m_gcalls; }

    private:

        // attributes
        const function_t&       m_function;             ///<
        mutable tensor_size_t   m_fcalls{0};            ///< #function value evaluations
        mutable tensor_size_t   m_gcalls{0};            ///< #function gradient evaluations
    };
}
