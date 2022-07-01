#include <nano/core/numeric.h>
#include <nano/function.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] static auto check_gradient(const function_t& function, int trials = 100, scalar_t epsilon_factor = 5.0)
{
    for (auto trial = 0; trial < trials; ++trial)
    {
        const vector_t x = vector_t::Random(function.size());
        UTEST_CHECK_LESS(function.grad_accuracy(x), epsilon_factor * epsilon2<scalar_t>());
    }
}

[[maybe_unused]] static auto check_convexity(const function_t& function, int trials = 100)
{
    for (auto trial = 0; trial < trials && function.convex(); ++trial)
    {
        const vector_t x0 = vector_t::Random(function.size());
        const vector_t x1 = vector_t::Random(function.size());
        UTEST_CHECK(function.is_convex(x0, x1, 20));
    }
}

class sum_function_t final : public function_t
{
public:
    explicit sum_function_t(tensor_size_t size)
        : function_t("sum", size)
    {
        convex(true);
        smooth(true);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = vector_t::Ones(x.size());
        }

        return x.sum();
    }
};

class cauchy_function_t final : public function_t
{
public:
    explicit cauchy_function_t(tensor_size_t size)
        : function_t("cauchy", size)
    {
        convex(false);
        smooth(true);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const override
    {
        if (gx != nullptr)
        {
            gx->noalias() = 2.0 * x / (0.36 + x.dot(x));
        }

        return std::log(0.36 + x.dot(x));
    }
};

class sumabsm1_function_t final : public function_t
{
public:
    explicit sumabsm1_function_t(tensor_size_t size)
        : function_t("sumabsm1", size)
    {
        convex(true);
        smooth(false);
    }

    scalar_t vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const override
    {
        if (gx != nullptr)
        {
            gx->array() = x.array().sign();
        }

        return x.array().abs().sum() - 1.0;
    }
};
