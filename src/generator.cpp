#include <mutex>
#include <nano/generator/elemwise_gradient.h>
#include <nano/generator/elemwise_identity.h>
#include <nano/generator/pairwise_product.h>

using namespace nano;

generator_t::generator_t(string_t id)
    : clonable_t(std::move(id))
{
}

void generator_t::fit(const datasource_t& datasource)
{
    m_datasource = &datasource;
}

void generator_t::allocate(tensor_size_t features)
{
    m_feature_infos.resize(features);
    m_feature_infos.zero();

    m_feature_rands.resize(static_cast<size_t>(features));
    std::generate(std::begin(m_feature_rands), std::end(m_feature_rands), []() { return make_rng(); });
}

void generator_t::undrop()
{
    m_feature_infos.array() = 0x00;
}

void generator_t::drop(tensor_size_t feature)
{
    m_feature_infos(feature) = 0x01;
}

void generator_t::unshuffle()
{
    m_feature_infos.array() = 0x00;
}

void generator_t::shuffle(tensor_size_t feature)
{
    m_feature_infos(feature) = 0x02;
}

bool generator_t::should_drop(tensor_size_t feature) const
{
    return m_feature_infos(feature) == 0x01;
}

bool generator_t::should_shuffle(tensor_size_t feature) const
{
    return m_feature_infos(feature) == 0x02;
}

indices_t generator_t::shuffled(indices_cmap_t samples, tensor_size_t feature) const
{
    auto rng = m_feature_rands[static_cast<size_t>(feature)];

    indices_t shuffled_samples = samples;
    std::shuffle(std::begin(shuffled_samples), std::end(shuffled_samples), rng);
    return shuffled_samples;
}

void generator_t::flatten_dropped(tensor2d_map_t storage, tensor_size_t column, tensor_size_t colsize)
{
    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

    const auto samples = storage.size<0>();

    if (colsize == 1)
    {
        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            storage(sample, column) = NaN;
        }
    }
    else
    {
        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            auto segment = storage.vector(sample).segment(column, colsize);
            segment.setConstant(NaN);
        }
    }
}

const datasource_t& generator_t::datasource() const
{
    critical(m_datasource == nullptr, "generator: cannot access the dataset before fitting!");

    return *m_datasource;
}

factory_t<generator_t>& generator_t::all()
{
    static auto manager = factory_t<generator_t>{};
    const auto  op      = []()
    {
        manager.add<elemwise_generator_t<elemwise_gradient_t>>(
            "gradient-like features (e.g. edge orientation & magnitude) from structured features (e.g. images)");

        manager.add<elemwise_generator_t<sclass_identity_t>>(
            "identity transformation, forward the single-label features");

        manager.add<elemwise_generator_t<mclass_identity_t>>(
            "identity transformation, forward the multi-label features");

        manager.add<elemwise_generator_t<scalar_identity_t>>("identity transformation, forward the scalar features");
        manager.add<elemwise_generator_t<struct_identity_t>>(
            "identity transformation, forward the structured features (e.g. images)");

        manager.add<pairwise_generator_t<pairwise_product_t>>("product of scalar features to generate quadratic terms");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
