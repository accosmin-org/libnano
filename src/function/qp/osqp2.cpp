#include <function/qp/osqp2.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>
#include <nano/function/util.h>

using namespace nano;

quadratic_program_osqp2_t::quadratic_program_osqp2_t(const tensor_size_t dims, const uint64_t seed, const scalar_t neqs,
                                                     const scalar_t alpha)
    : quadratic_program_t("osqp2", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::osqp2::neqs", 0.0, LT, neqs, LT, 1.0));
    register_parameter(parameter_t::make_scalar("function::osqp2::alpha", 0.0, LT, alpha, LE, 100.0));

    const auto n = dims;
    const auto p = std::clamp(static_cast<tensor_size_t>(neqs * static_cast<scalar_t>(n)), tensor_size_t{1}, n - 1);

    auto rng   = make_rng(seed);
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};

    const auto q = make_full_vector<scalar_t>(n, [&]() { return gdist(rng); });
    const auto x = make_full_vector<scalar_t>(n, [&]() { return gdist(rng); });

    const auto M = make_full_matrix<scalar_t>(n, n, [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });
    const auto A = make_full_matrix<scalar_t>(p, n, [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });

    this->Q() = M * M.transpose() + alpha * matrix_t::identity(n, n);
    this->c() = q;

    auto AA = A;
    auto bb = vector_t{A * x};
    reduce(AA, bb);

    critical((AA * variable()) == bb);
}

rfunction_t quadratic_program_osqp2_t::clone() const
{
    return std::make_unique<quadratic_program_osqp2_t>(*this);
}

string_t quadratic_program_osqp2_t::do_name() const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto neqs  = parameter("function::osqp2::neqs").value<scalar_t>();
    const auto alpha = parameter("function::osqp2::alpha").value<scalar_t>();

    return scat(type_id(), "[neqs=", neqs, ",alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t quadratic_program_osqp2_t::make(const tensor_size_t dims) const
{
    const auto seed  = parameter("function::seed").value<uint64_t>();
    const auto neqs  = parameter("function::osqp2::neqs").value<scalar_t>();
    const auto alpha = parameter("function::osqp2::alpha").value<scalar_t>();

    return std::make_unique<quadratic_program_osqp2_t>(dims, seed, neqs, alpha);
}
