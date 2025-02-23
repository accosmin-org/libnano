#include <function/program/cvx410.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx410_t::linear_program_cvx410_t(const tensor_size_t dims)
    : linear_program_t("cvx410", vector_t::zero(dims))
{
    const auto D = make_random_matrix<scalar_t>(dims, dims);
    const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
    const auto c = make_random_vector<scalar_t>(dims);

    // the solution is feasible
    const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0);
    const auto b = A * x;

    reset(c);
    optimum(x);

    critical(variable() >= 0.0);
    critical((A * variable()) == b);
}

rfunction_t linear_program_cvx410_t::clone() const
{
    return std::make_unique<linear_program_cvx410_t>(*this);
}

rfunction_t linear_program_cvx410_t::make(const tensor_size_t dims) const
{
    return std::make_unique<linear_program_cvx410_t>(dims);
}
