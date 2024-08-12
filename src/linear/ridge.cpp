#include <nano/linear/function.h>
#include <nano/linear/ridge.h>
#include <nano/linear/util.h>

using namespace nano;

ridge_t::ridge_t()
    : linear_t("ridge")
{
}

rlinear_t ridge_t::clone() const
{
    return std::make_unique<ridge_t>(*this);
}

param_spaces_t ridge_t::make_param_spaces() const
{
    return {linear::make_param_space("l2reg")};
}

linear::function_t ridge_t::make_function(const flatten_iterator_t& iterator, const loss_t& loss,
                                          tensor1d_cmap_t params) const
{
    assert(params.size() == 1);

    const auto l1reg = 0.0;
    const auto l2reg = params(0);

    return {iterator, loss, l1reg, l2reg};
}
