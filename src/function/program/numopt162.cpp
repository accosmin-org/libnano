#include <Eigen/Dense>
#include <function/program/numopt162.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
auto make_neqs(const tensor_size_t dims, const scalar_t neqs)
{
    return std::max(tensor_size_t{1}, static_cast<tensor_size_t>(neqs * static_cast<scalar_t>(dims)));
}
} // namespace

quadratic_program_numopt162_t::quadratic_program_numopt162_t(const tensor_size_t dims, const scalar_t neqs)
    : quadratic_program_t("numopt162", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("numopt162::neqs", 0.0, LT, 0.5, LE, 1.0));

    parameter("numopt162::neqs") = neqs;

    const auto x0    = make_random_vector<scalar_t>(dims);
    const auto Q     = matrix_t::identity(dims, dims);
    const auto c     = -x0;
    const auto Arows = make_neqs(dims, neqs);

    auto L = make_random_matrix<scalar_t>(Arows, Arows);
    auto U = make_random_matrix<scalar_t>(Arows, dims);

    L.matrix().triangularView<Eigen::Upper>().setZero();
    U.matrix().triangularView<Eigen::Lower>().setZero();
    L.diagonal().array() = 1.0;
    U.diagonal().array() = 1.0;

    const auto A     = L * U;
    const auto b     = make_random_vector<scalar_t>(Arows);
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

string_t quadratic_program_numopt162_t::do_name() const
{
    const auto neqs = parameter("numopt162::neqs").value<scalar_t>();

    return scat(type_id(), "[neqs=", neqs, "]");
}

rfunction_t quadratic_program_numopt162_t::make(const tensor_size_t dims) const
{
    const auto neqs = parameter("numopt162::neqs").value<scalar_t>();

    return std::make_unique<quadratic_program_numopt162_t>(dims, neqs);
}
