#pragma once

#include <nano/arch.h>
#include <nano/eigen.h>
#include <nano/string.h>

namespace nano
{
    ///
    /// \brief generic multi-dimensional optimization problem.
    ///
    class NANO_PUBLIC function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        function_t(string_t name, tensor_size_t size);

        ///
        /// \brief enable copying
        ///
        function_t(const function_t&) = default;
        function_t& operator=(const function_t&) = default;

        ///
        /// \brief enable moving
        ///
        function_t(function_t&&) noexcept = default;
        function_t& operator=(function_t&&) noexcept = default;

        ///
        /// \brief destructor
        ///
        virtual ~function_t() = default;

        ///
        /// \brief function name to identify it in tests and benchmarks
        ///
        string_t name() const;

        ///
        /// \brief returns the number of dimensions.
        ///
        tensor_size_t size() const { return m_size; }

        ///
        /// \brief compute the gradient accuracy (given vs. central finite difference approximation).
        ///
        scalar_t grad_accuracy(const vector_t& x) const;

        ///
        /// \brief check if the function is convex along the [x1, x2] line.
        ///
        bool is_convex(const vector_t& x1, const vector_t& x2, int steps) const;

        ///
        /// \brief returns whether the function is convex.
        ///
        bool convex() const { return m_convex; }

        ///
        /// \brief returns whether the function is smooth.
        ///
        /// NB: if not, then only sub-gradients are available.
        ///
        bool smooth() const { return m_smooth; }

        ///
        /// \brief returns the number of summands the function can be decomposed into.
        ///
        /// NB: if not applicable, then the number of summands is one.
        /// NB: if many summands, then stochastic optimization methods may be appropriate (e.g. ML problems).
        ///
        tensor_size_t summands() const { return m_summands; }

        ///
        /// \brief evaluate the function's value at the given point
        ///     (and its gradient or sub-gradient if not smooth).
        ///
        virtual scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

        ///
        /// \brief evaluate the function's given summand's value at the given point
        ///     (and its gradient or sub-gradient if not smooth).
        ///
        virtual scalar_t vgrad(const vector_t& x, tensor_size_t summand, vector_t* gx = nullptr) const;

    protected:

        void convex(bool);
        void smooth(bool);
        void summands(tensor_size_t);

    private:

        // attributes
        string_t        m_name;             ///<
        tensor_size_t   m_size{0};          ///< #free dimensions to optimize for
        bool            m_convex{false};    ///< whether the function is convex
        bool            m_smooth{false};    ///< whether the function is smooth (otherwise subgradients should be used)
        tensor_size_t   m_summands{1};      ///< number of summands (if stochastic optimization methods are appropriate)
    };
}
