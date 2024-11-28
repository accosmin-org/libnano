#pragma once

#include <nano/factory.h>
#include <nano/function/constraint.h>
#include <nano/string.h>

namespace nano
{
class function_t;
using rfunction_t = std::unique_ptr<function_t>;

enum class convexity : uint8_t
{
    ignore,
    yes,
    no
};

enum class smoothness : uint8_t
{
    ignore,
    yes,
    no
};

enum class constrained : uint8_t
{
    ignore,
    yest,
    no
};

///
/// \brief generic multi-dimensional function typically used as the objective of a numerical optimization problem.
///
/// optionally a set of equality and inequality constraints can be added
/// following the generic constrained optimization problems:
///
///     argmin      f(x),           - the objective function
///     such that   h_j(x) = 0,     - the equality constraints
///                 g_i(x) <= 0.    - the inequality constraints
///
/// NB: the (sub-)gradient of the function must be implemented.
/// NB: the functions can be convex or non-convex and smooth or non-smooth.
///
class NANO_PUBLIC function_t : public typed_t, public clonable_t<function_t>
{
public:
    ///
    /// \brief constructor
    ///
    function_t(string_t id, tensor_size_t size);

    ///
    /// \brief returns the available implementations for benchmarking numerical optimization methods.
    ///
    static factory_t<function_t>& all();

    ///
    /// \brief function name to identify it in tests and benchmarks.
    ///
    string_t name(bool with_size = true) const;

    ///
    /// \brief returns the number of dimensions.
    ///
    tensor_size_t size() const { return m_size; }

    ///
    /// \brief returns whether the function is convex.
    ///
    bool convex() const { return m_convexity == convexity::yes; }

    ///
    /// \brief returns whether the function is smooth.
    ///
    /// NB: mathematically a smooth function is of C^inf class (gradients exist and are ontinuous for any order),
    /// but here it implies that the function is of C^1 class (differentiable and with continuous gradients)
    /// as required by the line-search methods. If this is not the case, either only sub-gradients are available
    /// of the gradients are not continuous.
    ///
    ///
    bool smooth() const { return m_smoothness == smoothness::yes; }

    ///
    /// \brief returns the strong convexity coefficient.
    ///
    /// NB: if not convex, then the coefficient is zero.
    /// NB: it can be used to speed-up some numerical optimization algorithms if greater than zero.
    ///
    scalar_t strong_convexity() const { return m_strong_convexity; }

    ///
    /// \brief returns true if the given point satisfies all the stored constraints.
    ///
    bool valid(const vector_t& x) const;

    ///
    /// \brief register a constraint.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain(constraint_t&&);

    ///
    /// \brief register a bound constraint for all dimensions:
    ///     min <= x[i] <= max.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_range(scalar_t min, scalar_t max);

    ///
    /// \brief register a bound constraint for the given dimension:
    ///     min <= x[dimmension] <= max.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_range(scalar_t min, scalar_t max, tensor_size_t dimension);

    ///
    /// \brief register a bound constraint for all dimensions:
    ///     min[i] <= x[i] <= max[i].
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_range(const vector_t& min, const vector_t& max);

    ///
    /// \brief register a one-sided inequality contraint for all dimensions:
    ///     x[i] <= max.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_less(scalar_t max);

    ///
    /// \brief register a one-sided inequality contraint for the given dimension:
    ///     x[dimension] <= max.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_less(scalar_t max, tensor_size_t dimension);

    ///
    /// \brief register a one-sided inequality contraint for all dimensions:
    ///     x[i] <= max[i].
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_less(const vector_t& max);

    ///
    /// \brief register a one-sided inequality contraint for all dimensions:
    ///     x[i] >= min.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_greater(scalar_t min);

    ///
    /// \brief register a one-sided inequality contraint for the given dimension:
    ///     x[dimension] >= min.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_greater(scalar_t min, tensor_size_t dimension);

    ///
    /// \brief register a one-sided inequality contraint for all dimensions:
    ///     x[i] >= min[i].
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    bool constrain_greater(const vector_t& min);

    ///
    /// \brief register a linear equality constraint: A * x = b.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    template <class tmatrixA, class tvectorb,
              std::enable_if_t<is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>, bool> = true, ///<
              std::enable_if_t<is_eigen_v<tvectorb> || is_tensor_v<tvectorb>, bool> = true> ///<
    bool constrain_equality(const tmatrixA& A, const tvectorb& b)
    {
        if constexpr (is_tensor_v<tmatrixA>)
        {
            static_assert(tmatrixA::rank() == 2U);
        }
        if constexpr (is_tensor_v<tvectorb>)
        {
            static_assert(tvectorb::rank() == 1U);
        }
        return constrain_equalities(A, b);
    }

    ///
    /// \brief register a linear equality constraint: a.dot(x) = b.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    template <class tvectora, std::enable_if_t<is_eigen_v<tvectora> || is_tensor_v<tvectora>, bool> = true>
    bool constrain_equality(const tvectora& a, const scalar_t b)
    {
        if constexpr (is_eigen_v<tvectora>)
        {
            assert(a.cols() == 1);
        }
        else
        {
            static_assert(tvectora::rank() == 1U);
        }
        return constrain_equalities(a.transpose(), vector_t::constant(1, b));
    }

    ///
    /// \brief register a linear inequality constraint: A * x <= b.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    template <class tmatrixA, class tvectorb,
              std::enable_if_t<is_eigen_v<tmatrixA> || is_tensor_v<tmatrixA>, bool> = true, ///<
              std::enable_if_t<is_eigen_v<tvectorb> || is_tensor_v<tvectorb>, bool> = true> ///<
    bool constrain_inequality(const tmatrixA& A, const tvectorb& b)
    {
        if constexpr (is_tensor_v<tmatrixA>)
        {
            static_assert(tmatrixA::rank() == 2U);
        }
        if constexpr (is_tensor_v<tvectorb>)
        {
            static_assert(tvectorb::rank() == 1U);
        }
        return constrain_inequalities(A, b);
    }

    ///
    /// \brief register a linear inequality constraint: a.dot(x) <= b.
    ///
    /// NB: returns false if the constraint is neither valid nor compatible with the objective function.
    ///
    template <class tvectora, std::enable_if_t<is_eigen_v<tvectora> || is_tensor_v<tvectora>, bool> = true>
    bool constrain_inequality(const tvectora& a, const scalar_t b)
    {
        if constexpr (is_eigen_v<tvectora>)
        {
            assert(a.cols() == 1);
        }
        else
        {
            static_assert(tvectora::rank() == 1U);
        }
        return constrain_inequalities(a.transpose(), vector_t::constant(1, b));
    }

    ///
    /// \brief returns the set of registered constraints.
    ///
    const constraints_t& constraints() const;

    ///
    /// \brief returns the number of equality constraints.
    ///
    tensor_size_t n_equalities() const;

    ///
    /// \brief returns the number of inequality constraints.
    ///
    tensor_size_t n_inequalities() const;

    ///
    /// \brief evaluate the function's value at the given point
    ///     (and optionally its gradient or sub-gradient if not smooth).
    ///
    scalar_t vgrad(vector_cmap_t x, vector_map_t gx = vector_map_t{}) const;

    ///
    /// \brief returns the number of function evaluation calls registered so far.
    ///
    tensor_size_t fcalls() const;

    ///
    /// \brief returns the number of function gradient calls registered so far.
    ///
    tensor_size_t gcalls() const;

    ///
    /// \brief clear collected statistics (e.g. function calls).
    ///
    void clear_statistics() const;

    ///
    /// \brief construct test functions having:
    ///     - the number of dimensions within the given range,
    ///     - the given number of summands and
    ///     - the given requirements in terms of smoothness and convexity.
    ///
    struct config_t
    {
        tensor_size_t m_min_dims{2};                    ///<
        tensor_size_t m_max_dims{8};                    ///<
        convexity     m_convexity{convexity::ignore};   ///<
        smoothness    m_smoothness{smoothness::ignore}; ///<
        tensor_size_t m_summands{1000};                 ///<
    };

    static rfunctions_t make(const config_t&, const std::regex& id_regex = std::regex(".+"));

    ///
    /// \brief construct a test function with the given number of free dimensions and summands (if applicable).
    ///
    virtual rfunction_t make(tensor_size_t dims, tensor_size_t summands) const;

protected:
    void convex(convexity);
    void smooth(smoothness);
    void strong_convexity(scalar_t);

    virtual scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const = 0;

private:
    template <class tmatrixA, class tvectorb>
    bool constrain_equalities(const tmatrixA& A, const tvectorb& b)
    {
        bool ok = true;
        for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
        {
            ok = (A.rows() == b.size()) && constrain(constraint::linear_equality_t{A.row(i), -b(i)});
        }
        return ok;
    }

    template <class tmatrixA, class tvectorb>
    bool constrain_inequalities(const tmatrixA& A, const tvectorb& b)
    {
        bool ok = true;
        for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
        {
            ok = (A.rows() == b.size()) && constrain(constraint::linear_inequality_t{A.row(i), -b(i)});
        }
        return ok;
    }

    // attributes
    tensor_size_t         m_size{0};                    ///< #free dimensions to optimize for
    convexity             m_convexity{convexity::no};   ///< whether the function is convex
    smoothness            m_smoothness{smoothness::no}; ///< whether the function is smooth
    scalar_t              m_strong_convexity{0};        ///< strong-convexity coefficient
    constraints_t         m_constraints;                ///< optional equality and inequality constraints
    mutable tensor_size_t m_fcalls{0};                  ///< number of function value evaluations
    mutable tensor_size_t m_gcalls{0};                  ///< number of function gradient evaluations
};
} // namespace nano
