#include <function/lp/cvx48b.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx48b_t::linear_program_cvx48b_t(const tensor_size_t dims, const uint64_t seed, const scalar_t lambda)
    : linear_program_t("cvx48b", vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::cvx48b::lambda", -1e+10, LE, lambda, LT, 0.0));

    const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0, seed);
    const auto b = urand<scalar_t>(-1.0, +1.0);
    const auto c = lambda * a;

    this->c() = c;
    optimum(lambda * b);

    critical(lambda <= 0.0);
    critical((a * variable()) <= b);
}

rfunction_t linear_program_cvx48b_t::clone() const
{
    return std::make_unique<linear_program_cvx48b_t>(*this);
}

string_t linear_program_cvx48b_t::do_name() const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto lambda = parameter("function::cvx48b::lambda").value<scalar_t>();

    return scat(type_id(), "[lambda=", lambda, ",seed=", seed, "]");
}

rfunction_t linear_program_cvx48b_t::make(const tensor_size_t dims) const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto lambda = parameter("function::cvx48b::lambda").value<scalar_t>();

    return std::make_unique<linear_program_cvx48b_t>(dims, seed, lambda);
}
