#include <Eigen/Dense>
#include <function/program/numopt162.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

quadratic_program_numopt162_t::quadratic_program_numopt162_t(const tensor_size_t dims, const scalar_t neqs_dims_ratio)
    : quadratic_program_t(scat("numopt162[neqs=", neqs_dims_ratio, "]"), matrix_t{matrix_t::zero(dims, dims)},
                          vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("numopt162::neqs_ratio", 0.0, LT, 0.5, LE, 1.0));

    parameter("numopt162::neqs_ratio") = neqs_dims_ratio;

    const auto neqs = static_cast<tensor_size_t>(neqs_dims_ratio * static_cast<scalar_t>(dims));

    const auto x0 = make_random_vector<scalar_t>(dims);
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;

    auto L = make_random_matrix<scalar_t>(neqs, neqs);
    auto U = make_random_matrix<scalar_t>(neqs, dims);

    L.matrix().triangularView<Eigen::Upper>().setZero();
    U.matrix().triangularView<Eigen::Lower>().setZero();

    L.diagonal().array() = 1.0;
    U.diagonal().array() = 1.0;

    const auto A     = L * U;
    const auto b     = make_random_vector<scalar_t>(neqs);
    const auto invAA = (A * A.transpose()).inverse();
    const auto xbest = x0 + A.transpose() * invAA * (b - A * x0);

    reset(Q, c);
    optimum(xbest);

    critical((A * variable()) == b);
}

rfunction_t quadratic_program_numopt162_t::clone() const
{
    return std::make_unique<quadratic_program_numopt162_t>(*this);
}

rfunction_t quadratic_program_numopt162_t::make(const tensor_size_t dims) const
{
    const auto neqs_dims_ratio = parameter("numopt162::neqs_ratio").value<scalar_t>();

    return std::make_unique<quadratic_program_numopt162_t>(dims, neqs_dims_ratio);
}
