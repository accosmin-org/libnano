#include <nano/linear/function.h>
#include <nano/linear/ordinary.h>
#include <nano/linear/util.h>

using namespace nano;

ordinary_t::ordinary_t()
    : linear_t("ordinary")
{
}

rlinear_t ordinary_t::clone() const
{
    return std::make_unique<ordinary_t>(*this);
}

param_spaces_t ordinary_t::make_param_spaces() const
{
    return {};
}

linear::function_t ordinary_t::make_function(const flatten_iterator_t& iterator, const loss_t& loss,
                                             [[maybe_unused]] tensor1d_cmap_t params) const
{
    assert(params.size() == 0);

    const auto l1reg = 0.0;
    const auto l2reg = 0.0;

    return {iterator, loss, l1reg, l2reg};
}
