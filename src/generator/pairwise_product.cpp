#include <nano/generator/pairwise_product.h>

using namespace nano;

feature_t pairwise_product_t::feature(const tensor_size_t ifeature) const
{
    return make_scalar_feature(ifeature, "product");
}

template class nano::pairwise_generator_t<pairwise_product_t>;
