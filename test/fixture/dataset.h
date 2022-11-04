#include "fixture/generator.h"
#include <nano/generator/elemwise_identity.h>

using namespace nano;

static auto make_dataset(const datasource_t& datasource)
{
    auto dataset = dataset_t{datasource};
    add_generator<elemwise_generator_t<sclass_identity_t>>(dataset);
    add_generator<elemwise_generator_t<mclass_identity_t>>(dataset);
    add_generator<elemwise_generator_t<scalar_identity_t>>(dataset);
    add_generator<elemwise_generator_t<struct_identity_t>>(dataset);
    return dataset;
}
