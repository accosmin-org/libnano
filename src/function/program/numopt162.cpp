#include <Eigen/Dense>
#include <function/program/numopt162.h>
#include <nano/core/scat.h>

using namespace nano;

quadratic_program_numopt162_t::quadratic_program_numopt162_t(const tensor_size_t dims, const tensor_size_t neqs)
    : quadratic_program_numopt162_t(scat("numopt162[neqs=", neqs, "]"), dims)
{
    assert(neqs >= 1);
    assert(neqs <= dims);

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

    (A * (*this)) == b;

    this->xbest(xbest);
}

rfunction_t quadratic_program_numopt162_t::clone() const
{
    return std::make_unique<quadratic_program_numopt162_t>(*this);
}

rfunction_t quadratic_program_numopt162_t::make(const tensor_size_t                  dims,
                                                [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<quadratic_program_numopt162_t>(dims);
}
