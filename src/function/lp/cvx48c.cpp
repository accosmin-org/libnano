#include <function/lp/cvx48c.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx48c_t::linear_program_cvx48c_t(const tensor_size_t dims, const uint64_t seed)
    : linear_program_t("cvx48c", vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));

    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);
    const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);
    const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0, seed);

    this->c() = c;
    optimum(l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign());

    critical(l <= variable());
    critical(variable() <= u);
}

rfunction_t linear_program_cvx48c_t::clone() const
{
    return std::make_unique<linear_program_cvx48c_t>(*this);
}

string_t linear_program_cvx48c_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[seed=", seed, "]");
}

rfunction_t linear_program_cvx48c_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return std::make_unique<linear_program_cvx48c_t>(dims, seed);
}
