#include <nano/core/numeric.h>
#include <nano/function/util.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] static vector_t make_random_x0(const function_t& function, const scalar_t scale = 1.0)
{
    const auto tensor = make_random_tensor<scalar_t>(make_dims(function.size()), -scale, +scale);
    return tensor.vector();
}

[[maybe_unused]] static std::vector<vector_t> make_random_x0s(const function_t& function, const scalar_t scale = 1.0)
{
    const auto dims = make_dims(function.size());

    std::vector<vector_t> vectors;
    vectors.emplace_back(make_random_x0(function, scale));

    const auto append = [&](const auto... tscalars)
    {
        const auto tensor = make_tensor<scalar_t>(dims, tscalars...);
        vectors.emplace_back(tensor.vector());
    };

    // bug: OSGA solver fails here
    if (function.name() == "Exponential[4D]")
    {
        append(0.6524579797097991, 0.3700950306557635, -0.1710712483548298, -0.1947479784347773);
    }

    // bug: OSGA solver fails here
    if (function.name() == "Kinks[4D]")
    {
        append(0.2545247188178488, -0.6632348569872683, -0.6260742327486718, 0.5950229544941097);
    }

    // bug: BFGS+backtrack solver fails here
    if (function.name() == "Qing[4D]")
    {
        append(0.4958921077711123, 0.9830812076917252, -0.5192140013563706, 0.0439401045948384);
        append(0.4708653254587751, -0.4933506021493146, 0.6892169862294326, -0.0050907711577802);
    }

    // bug: CGD+lemarechal solver fails here
    if (function.name() == "Rosenbrock[4D]")
    {
        append(-0.5950864762215742, 0.6160556733668063, 0.2815843360435921, 0.2692838673927147);
    }

    // bug: cgdescent line-search fails here
    if (function.name() == "Dixon-Price[2D]")
    {
        append(0.439934771063, -0.788200738134);
    }

    // bug: cgdescent line-search fails here
    if (function.name() == "Exponential[1D]")
    {
        append(0.817256233948);
    }

    // bug: LBFGS fails here (constrained objective2)
    if (function.name() == "objective2[2D]")
    {
        append(-2.278061902088, 3.148238354814);
    }

    return vectors;
}

[[maybe_unused]] static auto check_gradient(const function_t& function, const int trials = 100,
                                            const scalar_t epsilon = 1e-8)
{
    for (auto trial = 0; trial < trials; ++trial)
    {
        const auto x = make_random_x0(function);
        UTEST_CHECK_LESS(grad_accuracy(function, x, epsilon), epsilon);
    }
}

[[maybe_unused]] static auto check_convexity(const function_t& function, int trials = 100)
{
    for (auto trial = 0; trial < trials && function.convex(); ++trial)
    {
        const auto x0 = make_random_x0(function);
        const auto x1 = make_random_x0(function);
        UTEST_CHECK(is_convex(function, x0, x1, 20));
    }
}
