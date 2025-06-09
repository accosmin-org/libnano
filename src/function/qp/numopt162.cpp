#include <Eigen/Dense>
#include <function/qp/numopt162.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

quadratic_program_numopt162_t::quadratic_program_numopt162_t(const tensor_size_t dims, const uint64_t seed,
                                                             const scalar_t neqs)
    : quadratic_program_t("numopt162", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::numopt162::neqs", 0.0, LT, neqs, LE, 1.0));

    auto rng   = make_rng(seed);
    auto udist = make_udist<scalar_t>(-1.0, +1.0);

    const auto n = dims;
    const auto p = std::max(tensor_size_t{1}, static_cast<tensor_size_t>(neqs * static_cast<scalar_t>(n)));

    const auto x0 = make_full_vector<scalar_t>(n, [&]() { return udist(rng); });
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;

    auto L = make_full_matrix<scalar_t>(p, p, [&]() { return udist(rng); });
    auto U = make_full_matrix<scalar_t>(p, n, [&]() { return udist(rng); });

    L.matrix().triangularView<Eigen::Upper>().setZero();
    U.matrix().triangularView<Eigen::Lower>().setZero();
    L.diagonal().array() = 1.0;
    U.diagonal().array() = 1.0;

    const auto A     = L * U;
    const auto b     = make_full_vector<scalar_t>(p, [&]() { return udist(rng); });
    const auto invAA = (A * A.transpose()).inverse();
    const auto xbest = x0 + A.transpose() * invAA * (b - A * x0);

    this->Q() = Q;
    this->c() = c;
    optimum(xbest);

    critical((A * variable()) == b);
}

rfunction_t quadratic_program_numopt162_t::clone() const
{
    return std::make_unique<quadratic_program_numopt162_t>(*this);
}

string_t quadratic_program_numopt162_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();
    const auto neqs = parameter("function::numopt162::neqs").value<scalar_t>();

    return scat(type_id(), "[neqs=", neqs, ",seed=", seed, "]");
}

rfunction_t quadratic_program_numopt162_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();
    const auto neqs = parameter("function::numopt162::neqs").value<scalar_t>();

    return std::make_unique<quadratic_program_numopt162_t>(dims, seed, neqs);
}
