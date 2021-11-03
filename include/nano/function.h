#pragma once

#include <regex>
#include <memory>
#include <nano/arch.h>
#include <nano/eigen.h>
#include <nano/string.h>

namespace nano
{
    class function_t;
    using ref_function_t = std::reference_wrapper<const function_t>;
    using rfunction_t = std::unique_ptr<function_t>;
    using rfunctions_t = std::vector<rfunction_t>;

    ///
    /// \brief
    ///
    enum class convexity
    {
        yes,
        no,
        unknown,
    };

    ///
    /// \brief construct test functions having the number of dimensions within the given range.
    ///
    NANO_PUBLIC rfunctions_t get_functions(
        tensor_size_t min_dims, tensor_size_t max_dims, convexity,
        const std::regex& name_regex = std::regex(".+"));

    ///
    /// \brief generic multi-dimensional optimization problem.
    ///
    class NANO_PUBLIC function_t
    {
    public:

        ///
        /// \brief constructor
        ///
        function_t(string_t name, tensor_size_t size, convexity);

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
        /// \brief number of dimensions
        ///
        tensor_size_t size() const { return m_size; }

        ///
        /// \brief compute the gradient accuracy (given vs. central finite difference approximation)
        ///
        scalar_t grad_accuracy(const vector_t& x) const;

        ///
        /// \brief check if the function is convex along the [x1, x2] line
        ///
        bool is_convex(const vector_t& x1, const vector_t& x2, int steps) const;

        ///
        /// \brief returns convexity state (if known)
        ///
        convexity convex() const { return m_convexity; }

        ///
        /// \brief evaluate the function's value at the give point (and its gradient if provided).
        ///
        virtual scalar_t vgrad(const vector_t& x, vector_t* gx = nullptr) const = 0;

    private:

        // attributes
        string_t        m_name;                             ///<
        tensor_size_t   m_size{0};                          ///< #dimensions
        convexity       m_convexity{convexity::unknown};    ///<
    };
}
