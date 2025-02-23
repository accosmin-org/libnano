#include <function/program/cvx48b.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx48b_t::linear_program_cvx48b_t(const tensor_size_t dims, const scalar_t lambda)
    : linear_program_t("cvx48b", vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("cvx48b::lambda", -1e+10, LE, -1.0, LT, 0.0));

    parameter("cvx48b::lambda") = lambda;

    const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
    const auto b = urand<scalar_t>(-1.0, +1.0);
    const auto c = lambda * a;

    reset(c);
    optimum(lambda * b / (1.0 + c.lpNorm<2>()));

    critical(lambda <= 0.0);
    critical((a * variable()) <= b);
}

rfunction_t linear_program_cvx48b_t::clone() const
{
    return std::make_unique<linear_program_cvx48b_t>(*this);
}

string_t linear_program_cvx48b_t::do_name() const
{
    const auto lambda = parameter("cvx48b::lambda").value<scalar_t>();

    return scat(type_id(), "[lambda=", lambda, "]");
}

rfunction_t linear_program_cvx48b_t::make(const tensor_size_t dims) const
{
    const auto lambda = parameter("cvx48b::lambda").value<scalar_t>();

    return std::make_unique<linear_program_cvx48b_t>(dims, lambda);
}
