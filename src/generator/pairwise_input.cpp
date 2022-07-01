#include <nano/generator/pairwise_input.h>
#include <nano/generator/select.h>

using namespace nano;

feature_mapping_t pairwise_input_sclass_sclass_t::do_fit()
{
    return make_pairwise(select_sclass(dataset(), original_features1()),
                         select_sclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_sclass_mclass_t::do_fit()
{
    return make_pairwise(select_sclass(dataset(), original_features1()),
                         select_mclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_sclass_scalar_t::do_fit()
{
    return make_pairwise(select_sclass(dataset(), original_features1()),
                         select_scalar(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_sclass_struct_t::do_fit()
{
    return make_pairwise(select_sclass(dataset(), original_features1()),
                         select_struct(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_mclass_sclass_t::do_fit()
{
    return make_pairwise(select_mclass(dataset(), original_features1()),
                         select_sclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_mclass_mclass_t::do_fit()
{
    return make_pairwise(select_mclass(dataset(), original_features1()),
                         select_mclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_mclass_scalar_t::do_fit()
{
    return make_pairwise(select_mclass(dataset(), original_features1()),
                         select_scalar(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_mclass_struct_t::do_fit()
{
    return make_pairwise(select_mclass(dataset(), original_features1()),
                         select_struct(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_scalar_sclass_t::do_fit()
{
    return make_pairwise(select_scalar(dataset(), original_features1()),
                         select_sclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_scalar_mclass_t::do_fit()
{
    return make_pairwise(select_scalar(dataset(), original_features1()),
                         select_mclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_scalar_scalar_t::do_fit()
{
    return make_pairwise(select_scalar(dataset(), original_features1()),
                         select_scalar(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_scalar_struct_t::do_fit()
{
    return make_pairwise(select_scalar(dataset(), original_features1()),
                         select_struct(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_struct_sclass_t::do_fit()
{
    return make_pairwise(select_struct(dataset(), original_features1()),
                         select_sclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_struct_mclass_t::do_fit()
{
    return make_pairwise(select_struct(dataset(), original_features1()),
                         select_mclass(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_struct_scalar_t::do_fit()
{
    return make_pairwise(select_struct(dataset(), original_features1()),
                         select_scalar(dataset(), original_features2()));
}

feature_mapping_t pairwise_input_struct_struct_t::do_fit()
{
    return make_pairwise(select_struct(dataset(), original_features1()),
                         select_struct(dataset(), original_features2()));
}
