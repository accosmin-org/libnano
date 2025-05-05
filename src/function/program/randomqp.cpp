#include <function/program/randomqp.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

namespace
{
auto make_ineqs(const tensor_size_t dims, const scalar_t ineqs)
{
    return std::max(tensor_size_t{1}, static_cast<tensor_size_t>(ineqs * static_cast<scalar_t>(dims)));
}
} // namespace

quadratic_program_randomqp_t::quadratic_program_randomqp_t(const tensor_size_t dims, const scalar_t ineqs,
                                                           const scalar_t alpha)
    : quadratic_program_t("randomqp", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    register_parameter(parameter_t::make_scalar("randomqp::nineqs", 1.0, LE, 10.0, LE, 100.0));
    register_parameter(parameter_t::make_scalar("randomqp::alpha", 0.0, LT, 1e-2, LE, 100.0));

    parameter("randomqp::nineqs") = ineqs;
    parameter("randomqp::alpha")  = alpha;

    const auto nineqs = make_ineqs(dims, ineqs);

    auto rng   = make_rng();
    auto gdist = std::normal_distribution<scalar_t>{0.0, 1.0};
    auto udist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};
    auto sdist = std::uniform_real_distribution<scalar_t>{0.0, 1.0};

    auto q = vector_t{dims};
    auto l = vector_t{nineqs};
    auto u = vector_t{nineqs};

    std::generate(q.begin(), q.end(), [&]() { return gdist(rng); });
    std::generate(l.begin(), l.end(), [&]() { return -udist(rng); });
    std::generate(u.begin(), u.end(), [&]() { return udist(rng); });

    auto M = matrix_t{dims, dims};
    auto A = matrix_t{nineqs, dims};

    std::generate(M.begin(), M.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });
    std::generate(A.begin(), A.end(), [&]() { return (sdist(rng) < 0.15) ? gdist(rng) : 0.0; });

    this->Q() = M * M.transpose() + alpha * matrix_t::identity(dims, dims);
    this->c() = q;

    critical((A * variable()) >= l);
    critical((A * variable()) <= u);
}

rfunction_t quadratic_program_randomqp_t::clone() const
{
    return std::make_unique<quadratic_program_randomqp_t>(*this);
}

string_t quadratic_program_randomqp_t::do_name() const
{
    const auto ineqs = parameter("randomqp::nineqs").value<scalar_t>();
    const auto alpha = parameter("randomqp::alpha").value<scalar_t>();

    return scat(type_id(), "[ineqs=", ineqs, ",alpha=", alpha, "]");
}

rfunction_t quadratic_program_randomqp_t::make(const tensor_size_t dims) const
{
    const auto ineqs = parameter("randomqp::nineqs").value<scalar_t>();
    const auto alpha = parameter("randomqp::alpha").value<scalar_t>();

    return std::make_unique<quadratic_program_randomqp_t>(dims, ineqs, alpha);
}
