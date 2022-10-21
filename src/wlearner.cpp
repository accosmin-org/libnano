#include <mutex>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/dstep.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/hinge.h>
#include <nano/wlearner/stump.h>
#include <nano/wlearner/table.h>

using namespace nano;

wlearner_t::wlearner_t(string_t id)
    : clonable_t(std::move(id))
{
}

tensor4d_t wlearner_t::predict(const dataset_t& dataset, const indices_cmap_t& samples) const
{
    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    outputs.zero();
    predict(dataset, samples, outputs);

    return outputs;
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
        manager.add<table_wlearner_t>("look-up-table (categorical features): h(x|feature) = table[x[feature]]");
        manager.add<dtree_wlearner_t>(
            "decision tree (any features): recursively split samples using decision stumps and look-up tables");
        manager.add<dstep_wlearner_t>(
            "single-output look-up-table (categorical features): h(x|feature,fv) = pred * (x[feature] == fv)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
