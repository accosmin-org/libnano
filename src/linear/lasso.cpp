#include <nano/linear/function.h>
#include <nano/linear/lasso.h>
#include <nano/linear/util.h>

using namespace nano;

lasso_t::lasso_t()
    : linear_t("lasso")
{
}

rlinear_t lasso_t::clone() const
{
    return std::make_unique<lasso_t>(*this);
}

param_spaces_t lasso_t::make_param_spaces() const
{
    return {linear::make_param_space("l1reg")};
}

linear::function_t lasso_t::make_function(const flatten_iterator_t& iterator, const loss_t& loss,
                                          tensor1d_cmap_t params) const
{
    assert(params.size() == 1);

    const auto l1reg = params(0);
    const auto l2reg = 0.0;

    return {iterator, loss, l1reg, l2reg};
}
