#include <function/qp/osqp4.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

quadratic_program_osqp4_t::quadratic_program_osqp4_t(const tensor_size_t dims, const uint64_t seed,
                                                     const scalar_t factors, const scalar_t gamma)
    : quadratic_program_t("osqp4", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    parameter("function::seed") = seed;
    register_parameter(parameter_t::make_scalar("function::osqp4::factors", 0.0, LT, factors, LT, 1.0));
    register_parameter(parameter_t::make_scalar("function::osqp4::gamma", 0.0, LT, gamma, LE, 1e+6));

    const auto n = dims;
    const auto k = std::clamp(static_cast<tensor_size_t>(factors * static_cast<scalar_t>(n)), tensor_size_t{1}, n - 1);

    auto rng   = make_rng(seed);
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};
    auto ddist = std::uniform_real_distribution<scalar_t>{0.0, std::sqrt(static_cast<scalar_t>(k))};

    auto miu = vector_t{n};

    std::generate(miu.begin(), miu.end(), [&]() { return gdist(rng); });

    auto F = matrix_t{n, k};
    auto Q = matrix_t{n, n};
    auto d = vector_t{n};

    std::generate(F.begin(), F.end(), [&]() { return (sdist(rng) < 0.50) ? gdist(rng) : 0.0; });
    std::generate(d.begin(), d.end(), [&]() { return ddist(rng); });

    Q = F * F.transpose();
    Q.matrix().diagonal() += d.vector();

    this->Q() = Q;
    this->c() = -miu / (2.0 * gamma);

    critical((vector_t::constant(dims, 1.0) * variable()) == 1.0);
    critical(variable() >= 0.0);
}

rfunction_t quadratic_program_osqp4_t::clone() const
{
    return std::make_unique<quadratic_program_osqp4_t>(*this);
}

string_t quadratic_program_osqp4_t::do_name() const
{
    const auto seed    = parameter("function::seed").value<uint64_t>();
    const auto factors = parameter("function::osqp4::factors").value<scalar_t>();
    const auto gamma   = parameter("function::osqp4::gamma").value<scalar_t>();

    return scat(type_id(), "[factors=", factors, ",gamma=", gamma, ",seed=", seed, "]");
}

rfunction_t quadratic_program_osqp4_t::make(const tensor_size_t dims) const
{
    const auto seed    = parameter("function::seed").value<uint64_t>();
    const auto factors = parameter("function::osqp4::factors").value<scalar_t>();
    const auto gamma   = parameter("function::osqp4::gamma").value<scalar_t>();

    return std::make_unique<quadratic_program_osqp4_t>(dims, seed, factors, gamma);
}
