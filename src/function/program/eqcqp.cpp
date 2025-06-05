#include <function/program/eqcqp.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

equality_constrained_quadratic_program_t::equality_constrained_quadratic_program_t(const tensor_size_t dims)
    : quadratic_program_t("eqcqp", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("eqcqp::neqs", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("eqcqp::alpha", 0.0, LT, 1e-2, LE, 100.0));

    generate(reference == nullptr ? *this : *reference);
}

rfunction_t equality_constrained_quadratic_program_t::clone() const
{
    return std::make_unique<equality_constrained_quadratic_program_t>(*this);
}

string_t equality_constrained_quadratic_program_t::do_name() const
{
    const auto neqs  = parameter("eqcqp::neqs").value<scalar_t>();
    const auto alpha = parameter("eqcqp::alpha").value<scalar_t>();
    const auto seed  = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[neqs=", neqs, ",alpha=", alpha, ",seed=", seed, "]");
}

rfunction_t equality_constrained_quadratic_program_t::make(const tensor_size_t dims) const
{
    return std::make_unique<equality_constrained_quadratic_program_t>(dims, *this);
}

void equality_constrained_quadratic_program_t::generate(const function_t& reference)
{
    const auto pratio = parameter("eqcqp::pratio").value<scalar_t>();
    const auto alpha  = parameter("eqcqp::alpha").value<scalar_t>();
    const auto seed   = parameter("function::seed").value<uint64_t>();

    const auto n = this->dims();
    const auto p = std::clamp(static_cast<tensor_size_t>(pratio * static_cast<scalar_t>(n)), tensor_size_t{1}, n - 1);

    auto rng   = make_rng(seed);
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};

    auto q = vector_t{n};
    auto x = vector_t{n};

    std::generate(q.begin(), q.end(), [&]() { return gdist(rng); });
    std::generate(x.begin(), x.end(), [&]() { return gdist(rng); });

    auto M = matrix_t{n, n};
    auto A = matrix_t{p, n};

    std::generate(M.begin(), M.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });
    std::generate(A.begin(), A.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });

    this->Q() = M * M.transpose() + alpha * matrix_t::identity(n, n);
    this->c() = q;

    critical((A * variable()) == (A * x));
}
