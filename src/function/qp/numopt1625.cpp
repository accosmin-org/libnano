#include <function/qp/numopt1625.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>

using namespace nano;

quadratic_program_numopt1625_t::quadratic_program_numopt1625_t(const tensor_size_t dims, const uint64_t seed)
    : quadratic_program_t("numopt1625", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));

    auto rng   = make_rng(seed);
    auto udist = make_udist<scalar_t>(-1.0, +1.0);

    const auto x0 = make_full_vector<scalar_t>(dims, [&]() { return udist(rng); });
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;
    const auto l  = make_full_vector<scalar_t>(dims, [&]() { return udist(rng); });
    const auto u  = l.array() + 0.1;

    this->Q() = Q;
    this->c() = c;
    optimum(x0.array().max(l.array()).min(u.array()));

    critical(l <= variable());
    critical(variable() <= u);
}

rfunction_t quadratic_program_numopt1625_t::clone() const
{
    return std::make_unique<quadratic_program_numopt1625_t>(*this);
}

string_t quadratic_program_numopt1625_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[seed=", seed, "]");
}

rfunction_t quadratic_program_numopt1625_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return std::make_unique<quadratic_program_numopt1625_t>(dims, seed);
}
