#include <nano/generator/elemwise_identity.h>

using namespace nano;

feature_t sclass_identity_t::feature(const tensor_size_t ifeature) const
{
    return datasource().feature(mapped_original(ifeature));
}

feature_t mclass_identity_t::feature(const tensor_size_t ifeature) const
{
    return datasource().feature(mapped_original(ifeature));
}

feature_t scalar_identity_t::feature(const tensor_size_t ifeature) const
{
    return datasource().feature(mapped_original(ifeature));
}

feature_t struct_identity_t::feature(const tensor_size_t ifeature) const
{
    return datasource().feature(mapped_original(ifeature));
}

template class nano::elemwise_generator_t<sclass_identity_t>;
template class nano::elemwise_generator_t<mclass_identity_t>;
template class nano::elemwise_generator_t<scalar_identity_t>;
template class nano::elemwise_generator_t<struct_identity_t>;
