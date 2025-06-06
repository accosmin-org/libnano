#include <function/lp/cvx49.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx49_t::linear_program_cvx49_t(const tensor_size_t dims, const uint64_t seed)
    : linear_program_t("cvx49", vector_t::zero(dims))
{
    parameter("function::seed") = seed;

    const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0, seed);
    const auto A = matrix_t::identity(dims, dims);
    const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0, seed);

    this->c() = c;
    optimum(b);

    critical((A * variable()) <= b);
}

rfunction_t linear_program_cvx49_t::clone() const
{
    return std::make_unique<linear_program_cvx49_t>(*this);
}

string_t linear_program_cvx49_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[seed=", seed, "]");
}

rfunction_t linear_program_cvx49_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return std::make_unique<linear_program_cvx49_t>(dims, seed);
}
