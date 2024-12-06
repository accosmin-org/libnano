#include <Eigen/Dense>
#include <function/program/cvx49.h>
#include <nano/core/scat.h>
#include <nano/function/numeric.h>

using namespace nano;

using Type = Eigen::CwiseNullaryOp<              ///<
    Eigen::internal::scalar_constant_op<double>, ///<
    Eigen::Matrix<double, -1, 1>>;               ///<

static_assert(is_eigen_v<Type>);
static_assert(!is_tensor_v<Type>);
static_assert(is_vector_v<Type>);

linear_program_cvx49_t::linear_program_cvx49_t(const tensor_size_t dims)
    : linear_program_t("cvx49", dims)
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
    const auto A = matrix_t::identity(dims, dims);
    const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

    reset(c);
    optimum(b);

    //(A * variable()) <= b;
}

rfunction_t linear_program_cvx49_t::clone() const
{
    return std::make_unique<linear_program_cvx49_t>(*this);
}

rfunction_t linear_program_cvx49_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx49_t>(dims);
}