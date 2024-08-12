#include <nano/linear/elastic_net.h>
#include <nano/linear/function.h>
#include <nano/linear/util.h>

using namespace nano;

elastic_net_t::elastic_net_t()
    : linear_t("elastic_net")
{
}

rlinear_t elastic_net_t::clone() const
{
    return std::make_unique<elastic_net_t>(*this);
}

param_spaces_t elastic_net_t::make_param_spaces() const
{
    return {linear::make_param_space("l1reg"), linear::make_param_space("l2reg")};
}

linear::function_t elastic_net_t::make_function(const flatten_iterator_t& iterator, const loss_t& loss,
                                                tensor1d_cmap_t params) const
{
    assert(params.size() == 2);

    const auto l1reg = params(0);
    const auto l2reg = params(1);

    return {iterator, loss, l1reg, l2reg};
}
