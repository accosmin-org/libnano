#include <function/nonlinear/kinks.h>
#include <nano/core/stats.h>
#include <nano/tensor/tensor.h>

using namespace nano;

function_kinks_t::function_kinks_t(const tensor_size_t dims, const uint64_t seed)
    : function_t("kinks", dims)
    , m_kinks(make_random_matrix<scalar_t>(std::max(tensor_size_t{1}, static_cast<tensor_size_t>(std::sqrt(dims))),
                                           dims, -1.0, +1.0, seed))
{
    register_parameter(parameter_t::make_integer("function::seed", 0, LE, seed, LE, 10000));

    convex(convexity::yes);
    smooth(smoothness::no);

    auto kinks = vector_t{m_kinks.rows()};
    for (tensor_size_t i = 0; i < m_kinks.cols(); ++i)
    {
        kinks.vector() = m_kinks.matrix().col(i);

        const auto opt = median(std::begin(kinks), std::end(kinks));
        m_offset += (kinks.array() - opt).abs().sum();
    }
}

rfunction_t function_kinks_t::clone() const
{
    return std::make_unique<function_kinks_t>(*this);
}

scalar_t function_kinks_t::do_eval(eval_t eval) const
{
    const auto xv = x.vector();
    const auto km = m_kinks.matrix();

    if (gx.size() == x.size())
    {
        gx.full(0);
        for (tensor_size_t i = 0; i < km.rows(); ++i)
        {
            gx.array() += (xv.transpose().array() - km.row(i).array()).sign();
        }
    }

    scalar_t fx = -m_offset;
    for (tensor_size_t i = 0; i < km.rows(); ++i)
    {
        fx += (xv.transpose().array() - km.row(i).array()).abs().sum();
    }
    return fx;
}

string_t function_kinks_t::do_name() const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return scat(type_id(), "[seed=", seed, "]");
}

rfunction_t function_kinks_t::make(const tensor_size_t dims) const
{
    const auto seed = parameter("function::seed").value<uint64_t>();

    return std::make_unique<function_kinks_t>(dims, seed);
}
