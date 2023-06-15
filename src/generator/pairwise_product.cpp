#include <nano/generator/pairwise_product.h>

using namespace nano;

pairwise_product_t::pairwise_product_t()
    : pairwise_input_scalar_scalar_t("product")
{
}

pairwise_product_t::pairwise_product_t(indices_t original_features)
    : pairwise_input_scalar_scalar_t("product", std::move(original_features))
{
}

pairwise_product_t::pairwise_product_t(indices_t original_features1, indices_t original_features2)
    : pairwise_input_scalar_scalar_t("product", std::move(original_features1), std::move(original_features2))
{
}

feature_t pairwise_product_t::feature(const tensor_size_t ifeature) const
{
    return make_scalar_feature(ifeature, "product");
}

template class nano::pairwise_generator_t<pairwise_product_t>;
