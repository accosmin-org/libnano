#include <nano/generator/elemwise_input.h>
#include <nano/generator/select.h>

using namespace nano;

feature_mapping_t elemwise_input_sclass_t::do_fit()
{
    return select_sclass(datasource(), original_features());
}

feature_mapping_t elemwise_input_mclass_t::do_fit()
{
    return select_mclass(datasource(), original_features());
}

feature_mapping_t elemwise_input_scalar_t::do_fit()
{
    return select_scalar(datasource(), original_features());
}

feature_mapping_t elemwise_input_struct_t::do_fit()
{
    return select_struct(datasource(), original_features());
}
