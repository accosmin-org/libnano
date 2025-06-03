#include <function/program/portfolio.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
auto make_k(const tensor_size_t dims, const scalar_t factors)
{
    return std::max(tensor_size_t{1}, static_cast<tensor_size_t>(factors * static_cast<scalar_t>(dims)));
}
} // namespace

quadratic_program_portfolio_t::quadratic_program_portfolio_t(const tensor_size_t dims, const scalar_t factors,
                                                             const scalar_t gamma)
    : quadratic_program_t("portfolio", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("portfolio::factors", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("portfolio::gamma", 0.0, LT, 1.0, LE, 1e+6));

    parameter("portfolio::factors") = factors;
    parameter("portfolio::gamma") = gamma;

    const auto k = make_k(dims, factors);

    auto rng   = make_rng();
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};
    auto ddist = std::uniform_real_distribution<scalar_t>{0.0, std::sqrt(static_cast<scalar_t>(k))};

    auto miu = vector_t{dims};

    std::generate(miu.begin(), miu.end(), [&]() { return gdist(rng); });

    auto F = matrix_t{dims, k};
    auto Q = matrix_t{dims, dims};
    auto d = vector_t{dims};

    std::generate(F.begin(), F.end(), [&]() { return (sdist(rng) < 0.50) ? gdist(rng) : 0.0; });
    std::generate(d.begin(), d.end(), [&]() { return ddist(rng); });

    Q = F * F.transpose();
    Q.matrix().diagonal() += d.vector();

    this->Q() = Q;
    this->c() = -miu / (2.0 * gamma);

    critical((vector_t::constant(dims, 1.0) * variable()) == 1.0);
    critical(variable() >= 0.0);
}

rfunction_t quadratic_program_portfolio_t::clone() const
{
    return std::make_unique<quadratic_program_portfolio_t>(*this);
}

string_t quadratic_program_portfolio_t::do_name() const
{
    const auto factors = parameter("portfolio::factors").value<scalar_t>();
    const auto gamma   = parameter("portfolio::gamma").value<scalar_t>();

    return scat(type_id(), "[factors=", factors, ",gamma=", gamma, "]");
}

rfunction_t quadratic_program_portfolio_t::make(const tensor_size_t dims) const
{
    const auto factors = parameter("portfolio::factors").value<scalar_t>();
    const auto gamma   = parameter("portfolio::gamma").value<scalar_t>();

    return std::make_unique<quadratic_program_portfolio_t>(dims, factors, gamma);
}
