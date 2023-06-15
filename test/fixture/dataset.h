#pragma once

#include "fixture/generator.h"
#include <nano/generator/elemwise_identity.h>

using namespace nano;

static auto make_dataset(const datasource_t& datasource)
{
    auto dataset = dataset_t{datasource};
    add_generator<sclass_identity_generator_t>(dataset);
    add_generator<mclass_identity_generator_t>(dataset);
    add_generator<scalar_identity_generator_t>(dataset);
    add_generator<struct_identity_generator_t>(dataset);
    return dataset;
}
