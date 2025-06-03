#include <function/program/eqcqp.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
auto make_neqs(const tensor_size_t dims, const scalar_t neqs)
{
    return std::clamp(tensor_size_t{1}, static_cast<tensor_size_t>(neqs * static_cast<scalar_t>(dims)), dims - 1);
}
} // namespace

quadratic_program_eqcqp_t::quadratic_program_eqcqp_t(const tensor_size_t dims, const scalar_t neqs,
                                                     const scalar_t alpha)
    : quadratic_program_t("eqcqp", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("eqcqp::neqs", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("eqcqp::alpha", 0.0, LT, 1e-2, LE, 100.0));

    parameter("eqcqp::neqs")  = neqs;
    parameter("eqcqp::alpha") = alpha;

    const auto nneqs = make_neqs(dims, neqs);

    auto rng   = make_rng();
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};

    auto q = vector_t{dims};
    auto b = vector_t{nneqs};

    std::generate(q.begin(), q.end(), [&]() { return gdist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return gdist(rng); });

    auto M = matrix_t{dims, dims};
    auto A = matrix_t{nneqs, dims};

    std::generate(M.begin(), M.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });
    std::generate(A.begin(), A.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });

    this->Q() = M * M.transpose() + alpha * matrix_t::identity(dims, dims);
    this->c() = q;

    critical((A * variable()) == b);
}

rfunction_t quadratic_program_eqcqp_t::clone() const
{
    return std::make_unique<quadratic_program_eqcqp_t>(*this);
}

string_t quadratic_program_eqcqp_t::do_name() const
{
    const auto neqs  = parameter("eqcqp::neqs").value<scalar_t>();
    const auto alpha = parameter("eqcqp::alpha").value<scalar_t>();

    return scat(type_id(), "[neqs=", neqs, ",alpha=", alpha, "]");
}

rfunction_t quadratic_program_eqcqp_t::make(const tensor_size_t dims) const
{
    const auto neqs  = parameter("eqcqp::neqs").value<scalar_t>();
    const auto alpha = parameter("eqcqp::alpha").value<scalar_t>();

    return std::make_unique<quadratic_program_eqcqp_t>(dims, neqs, alpha);
}
