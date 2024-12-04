#include <Eigen/Dense>
#include <function/program/numopt1625.h>
#include <nano/core/scat.h>

using namespace nano;

quadratic_program_numopt1625_t::quadratic_program_numopt1625_t(const tensor_size_t dims)
    : quadratic_program_numopt1625_t("numopt1625", dims)
{
    const auto x0 = make_random_vector<scalar_t>(dims);
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;
    const auto l  = make_random_vector<scalar_t>(dims);
    const auto u  = l.array() + 0.1;

    reset(Q, c);

    (*this) >= l;
    (*this) <= u;

    this->xbest(x0.array().max(l.array()).min(u.array()));
}

rfunction_t quadratic_program_numopt1625_t::clone() const
{
    return std::make_unique<quadratic_program_numopt1625_t>(*this);
}

rfunction_t quadratic_program_numopt1625_t::make(const tensor_size_t                  dims,
                                                 [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<quadratic_program_numopt1625_t>(dims);
}
