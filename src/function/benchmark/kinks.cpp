#include <nano/core/stats.h>
#include <nano/function/benchmark/kinks.h>
#include <nano/tensor/tensor.h>

using namespace nano;

function_kinks_t::function_kinks_t(tensor_size_t dims)
    : function_t("kinks", dims)
    , m_kinks(make_random_matrix<scalar_t>(std::max(tensor_size_t(1), static_cast<tensor_size_t>(std::sqrt(dims))),
                                           dims, -1.0, +1.0, seed_t{42U}))
{
    convex(convexity::yes);
    smooth(smoothness::no);

    vector_t kinks(m_kinks.rows());
    for (tensor_size_t i = 0; i < m_kinks.cols(); ++i)
    {
        kinks = m_kinks.col(i);

        const auto opt = median(begin(kinks), end(kinks));
        m_offset += (kinks.array() - opt).abs().sum();
    }
}

rfunction_t function_kinks_t::clone() const
{
    return std::make_unique<function_kinks_t>(*this);
}

scalar_t function_kinks_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        gx->setZero();
        for (tensor_size_t i = 0; i < m_kinks.rows(); ++i)
        {
            gx->array() += (x.transpose().array() - m_kinks.row(i).array()).sign();
        }
    }

    scalar_t fx = 0;
    for (tensor_size_t i = 0; i < m_kinks.rows(); ++i)
    {
        fx += (x.transpose().array() - m_kinks.row(i).array()).abs().sum();
    }
    return fx - m_offset;
}

rfunction_t function_kinks_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_kinks_t>(dims);
}
