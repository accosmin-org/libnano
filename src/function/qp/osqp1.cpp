#include <function/qp/osqp1.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

quadratic_program_osqp1_t::quadratic_program_osqp1_t(const tensor_size_t dims, const uint64_t seed,
                                                     const scalar_t nineqs, const scalar_t alpha)
    : quadratic_program_t("osqp1", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));
    register_parameter(parameter_t::make_scalar("function::osqp1::nineqs", 1.0, LE, nineqs, LE, 100.0));
    register_parameter(parameter_t::make_scalar("function::osqp1::alpha", 0.0, LT, alpha, LE, 100.0));

    const auto n = dims;
    const auto m = std::max(tensor_size_t{1}, static_cast<tensor_size_t>(nineqs * static_cast<scalar_t>(n)));

    auto rng   = make_rng(seed);
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto udist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};

    auto q = vector_t{n};
    auto l = vector_t{m};
    auto u = vector_t{m};

    std::generate(q.begin(), q.end(), [&]() { return gdist(rng); });
    std::generate(l.begin(), l.end(), [&]() { return -udist(rng); });
    std::generate(u.begin(), u.end(), [&]() { return udist(rng); });

    auto M = matrix_t{n, n};
    auto A = matrix_t{m, n};

    std::generate(M.begin(), M.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });
    std::generate(A.begin(), A.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });

    this->Q() = M * M.transpose() + alpha * matrix_t::identity(n, n);
    this->c() = q;

    critical((A * variable()) >= l);
    critical((A * variable()) <= u);
}

rfunction_t quadratic_program_osqp1_t::clone() const
{
    return std::make_unique<quadratic_program_osqp1_t>(*this);
}

string_t quadratic_program_osqp1_t::do_name() const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto nineqs = parameter("function::osqp1::nineqs").value<scalar_t>();
    const auto alpha  = parameter("function::osqp1::alpha").value<scalar_t>();

    return scat(type_id(), "[nineqs=", nineqs, ",alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t quadratic_program_osqp1_t::make(const tensor_size_t dims) const
{
    const auto seed   = parameter("function::seed").value<uint64_t>();
    const auto nineqs = parameter("function::osqp1::nineqs").value<scalar_t>();
    const auto alpha  = parameter("function::osqp1::alpha").value<scalar_t>();

    return std::make_unique<quadratic_program_osqp1_t>(dims, seed, nineqs, alpha);
}
