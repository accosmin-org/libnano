#include <function/lp/cvx410.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx410_t::linear_program_cvx410_t(const tensor_size_t dims, const uint64_t seed)
    : linear_program_t("cvx410", vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));

    const auto D = make_random_matrix<scalar_t>(dims, dims, -1.0, +1.0, seed);
    const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);

    // the solution is feasible
    const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0, seed);
    const auto b = A * x;

    this->c() = c;
    optimum(x);

    critical(variable() >= 0.0);
    critical((A * variable()) == b);
}

rfunction_t linear_program_cvx410_t::clone() const
{
    return std::make_unique<linear_program_cvx410_t>(*this);
}

string_t linear_program_cvx410_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[seed=", seed, "]");
}

rfunction_t linear_program_cvx410_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return std::make_unique<linear_program_cvx410_t>(dims, seed);
}
