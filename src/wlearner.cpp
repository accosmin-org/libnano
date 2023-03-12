#include <mutex>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/criterion.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/hinge.h>
#include <nano/wlearner/stump.h>
#include <nano/wlearner/table.h>

using namespace nano;
using namespace nano::wlearner;

wlearner_t::wlearner_t(string_t id)
    : clonable_t(std::move(id))
{
    register_parameter(parameter_t::make_enum("wlearner::criterion", criterion_type::aicc));
}

scalar_t wlearner_t::fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    assert(samples.min() >= 0);
    assert(samples.max() < dataset.samples());
    assert(gradients.dims() == cat_dims(dataset.samples(), dataset.target_dims()));

    const auto score = do_fit(dataset, samples, gradients);
    if (score != wlearner_t::no_fit_score())
    {
        learner_t::fit_dataset(dataset);
    }
    return score;
}

tensor4d_t wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples) const
{
    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    outputs.zero();
    predict(dataset, samples, outputs);

    return outputs;
}

void wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples, tensor4d_map_t outputs) const
{
    learner_t::critical_compatible(dataset);

    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    do_predict(dataset, samples, outputs);
}

cluster_t wlearner_t::split(const dataset_t& dataset, const indices_t& samples) const
{
    learner_t::critical_compatible(dataset);

    return do_split(dataset, samples);
}

bool wlearner_t::try_merge(const rwlearner_t&)
{
    return false;
}

factory_t<wlearner_t>& wlearner_t::all()
{
    static auto manager = factory_t<wlearner_t>{};
    const auto  op      = []()
    {
        manager.add<affine_wlearner_t>("affine mapping (scalar features): h(x|feature) = weight * x[feature] + bias");

        manager.add<stump_wlearner_t>(
            "decision stump (scalar features): h(x|feature,threshold) = high, if x[feature] >= threshold, else low");

        manager.add<hinge_wlearner_t>(
            "hinge (scalar features): h(x|feature,threshold,sign) = beta * {sign * (x[feature] - threshold)}+");

        manager.add<dense_table_wlearner_t>("dense look-up-table (categorical features)");
        manager.add<kbest_table_wlearner_t>("k-best look-up-table (categorical features)");
        manager.add<ksplit_table_wlearner_t>("k-split look-up-table (categorical features)");
        manager.add<dstep_table_wlearner_t>("discrete step look-up-table (categorical features)");

        manager.add<dtree_wlearner_t>(
            "decision tree (any features): recursively split samples using decision stumps and look-up tables");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
