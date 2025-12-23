#pragma once

#include <function/ml/dataset.h>
#include <nano/core/strutil.h>
#include <nano/critical.h>
#include <nano/function.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/lasso.h>

namespace nano
{
///
/// \brief synthetic linear machine learning model
///     where the predictions are an affine transformation of the inputs.
///
/// NB: the targets can be configured to be correlated only to some inputs (features) modulo a fixed constant.
/// NB: simulates either univariate regression or classification problems.
///
template <class tloss>
class NANO_PUBLIC linear_model_t : public function_t
{
public:
    linear_model_t(const char* const suffix, const tensor_size_t dims, const uint64_t seed, const scalar_t sratio,
                   const tensor_size_t modulo, const lasso_type type, const scalar_t alpha1, const scalar_t alpha2)
        : function_t(scat(tloss::basename, (type == lasso_type::constrained) ? '#' : '+', suffix),
                     make_size(dims, type))
        , m_dataset(make_samples(dims, sratio), make_outputs(dims), make_inputs(dims), seed, modulo, tloss::regression)
    {
        function_t::convex(tloss::convex ? convexity::yes : convexity::no);
        function_t::strong_convexity(alpha2);

        if (type == lasso_type::constrained)
        {
            constrain_lasso();
            function_t::smooth(tloss::smooth ? smoothness::yes : smoothness::no);
        }
        else
        {
            function_t::smooth((alpha1 == 0.0 && tloss::smooth) ? smoothness::yes : smoothness::no);
        }
    }

    scalar_t do_enet_eval(eval_t eval, const lasso_type type, const scalar_t alpha1, const scalar_t alpha2) const
    {
        if (type == lasso_type::constrained)
        {
            const auto n = size() / 2;
            const auto x = eval.m_x.segment(0, n);
            const auto z = eval.m_x.segment(n, n);

            auto fx = m_dataset.do_eval<tloss>(this->make_lasso_eval(eval));

            if (eval.has_grad())
            {
                eval.m_gx.segment(0, n).array() += alpha2 * x.array();
                eval.m_gx.segment(n, n).array() = alpha1;
            }

            if (eval.has_hess())
            {
                update_lasso_hess(eval);
                eval.m_hx.block(0, 0, n, n).diagonal().array() += alpha2;
            }

            fx += alpha1 * z.sum() + 0.5 * (std::sqrt(alpha2) * x).squaredNorm();
            return fx;
        }
        else
        {
            const auto n = size();
            const auto x = eval.m_x.segment(0, n);

            auto fx = m_dataset.do_eval<tloss>(eval);

            if (eval.has_grad())
            {
                eval.m_gx.array() += alpha1 * x.array().sign() + alpha2 * x.array();
            }

            if (eval.has_hess())
            {
                eval.m_hx.diagonal().array() += alpha2;
            }

            fx += alpha1 * x.lpNorm<1>() + 0.5 * (std::sqrt(alpha2) * x).squaredNorm();
            return fx;
        }
    }

private:
    inline void constrain_lasso()
    {
        // min  f(x, z)
        // s.t. +x <= z
        //      -x <= z
        const auto n = size() / 2;

        auto A              = matrix_t{2 * n, 2 * n};
        A.block(0, 0, n, n) = matrix_t::identity(n, n);
        A.block(0, n, n, n) = -matrix_t::identity(n, n);
        A.block(n, 0, n, n) = -matrix_t::identity(n, n);
        A.block(n, n, n, n) = -matrix_t::identity(n, n);

        critical(A * variable() <= vector_t::zero(2 * n));
    }

    static eval_t make_lasso_eval(eval_t eval)
    {
        auto       x  = eval.m_x;
        auto       gx = eval.m_gx;
        auto       hx = eval.m_hx;
        const auto n  = x.size() / 2;

        return {
            .m_x  = x.slice(0, n),
            .m_gx = eval.has_grad() ? gx.slice(0, n) : gx.tensor(),
            .m_hx = eval.has_hess() ? hx.reshape(hx.size()).slice(0, n * n).reshape(n, n).tensor() : hx.tensor(),
        };
    }

    static void update_lasso_hess(eval_t eval)
    {
        auto       x  = eval.m_x;
        auto       hx = eval.m_hx;
        const auto n  = x.size() / 2;

        hx.block(n, n, n, n) = hx.reshape(4 * n * n).slice(0, n * n).reshape(n, n).matrix();
        hx.block(0, 0, n, n) = hx.block(n, n, n, n);

        hx.block(0, n, n, n) = matrix_t::zero(n, n);
        hx.block(n, 0, n, n) = matrix_t::zero(n, n);
        hx.block(n, n, n, n) = matrix_t::zero(n, n);
    }

    static tensor_size_t make_size(const tensor_size_t dims, const lasso_type type = lasso_type::unconstrained)
    {
        switch (type)
        {
        case lasso_type::unconstrained:
            // solve for (x,)
            return std::max(dims, tensor_size_t{2});

        default:
            // solve for (x, z)
            return std::max(2 * dims, tensor_size_t{2});
        }
    }

    static tensor_size_t make_inputs(const tensor_size_t dims) { return std::max(dims, tensor_size_t{2}); }

    static tensor_size_t make_outputs([[maybe_unused]] const tensor_size_t dims) { return tensor_size_t{1}; }

    static tensor_size_t make_samples(const tensor_size_t dims, const scalar_t sratio)
    {
        return static_cast<tensor_size_t>(std::max(sratio * static_cast<scalar_t>(dims), 10.0));
    }

    // attributes
    linear_dataset_t m_dataset; ///<
};
} // namespace nano
