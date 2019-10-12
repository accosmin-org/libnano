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
        /// \brief compute function value (and gradient if provided)
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
        size_t fcalls() const { return m_fcalls; }

        ///
        /// \brief number of function gradient calls
        ///
        size_t gcalls() const { return m_gcalls; }

    private:

        // attributes
        const function_t&   m_function;         ///<
        mutable size_t      m_fcalls{0};        ///< #function value evaluations
        mutable size_t      m_gcalls{0};        ///< #function gradient evaluations
    };
}
