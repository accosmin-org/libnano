#include <nano/generator.h>
#include <nano/core/tpool.h>
#include <nano/dataset/iterator.h>

using namespace nano;

static void handle_sclass(tensor_size_t ifeature, const feature_t& feature)
{
    critical(
        feature.type() != feature_type::sclass,
        "generator_t: unhandled single-label feature <", ifeature, ":", feature, ">!");
}

static void handle_mclass(tensor_size_t ifeature, const feature_t& feature)
{
    critical(
        feature.type() != feature_type::mclass,
        "generator_t: unhandled multi-label feature <", ifeature, ":", feature, ">!");
}

static void handle_scalar(tensor_size_t ifeature, const feature_t& feature)
{
    critical(
        feature.type() == feature_type::sclass ||
        feature.type() == feature_type::mclass ||
        size(feature.dims()) != 1,
        "generator_t: unhandled scalar feature <", ifeature, ":", feature, ">!");
}

static void handle_struct(tensor_size_t ifeature, const feature_t& feature)
{
    critical(
        feature.type() == feature_type::sclass ||
        feature.type() == feature_type::mclass ||
        size(feature.dims()) == 1,
        "generator_t: unhandled structured feature <", ifeature, ":", feature, ">!");
}

template <typename tscalar, size_t trank, typename... tindices>
static auto resize_and_map(tensor_mem_t<tscalar, trank>& buffer, tindices... dims)
{
    if (buffer.size() < ::nano::size(make_dims(dims...)))
    {
        buffer.resize(dims...);
    }
    return map_tensor(buffer.data(), dims...);
}

template <typename toperator>
static void loop(execution_type execution, indices_cmap_t samples, tensor_size_t batch, const toperator& op)
{
    switch (execution)
    {
    case execution_type::par:
        ::nano::loopr(samples.size(), batch, op);
        break;

    default:
        for (tensor_size_t begin = 0, size = samples.size(); begin < size; begin += batch)
        {
            op(begin, std::min(begin + batch, size), 0U);
        }
        break;
    }
}

template <typename toperator>
static void loop(execution_type execution, indices_cmap_t ifeatures, const toperator& op)
{
    switch (execution)
    {
    case execution_type::par:
        ::nano::loopi(ifeatures.size(), [&] (tensor_size_t index, size_t tnum)
        {
            op(ifeatures(index), tnum);
        });
        break;

    default:
        for (tensor_size_t ifeature : ifeatures)
        {
            op(ifeature, 0U);
        }
        break;
    }
}

dataset_generator_t::dataset_generator_t(const dataset_t& dataset) :
    m_dataset(dataset)
{
}

void dataset_generator_t::update()
{
    tensor_size_t total_columns = 0, features = 0, generators = 0;
    for (const auto& generator : m_generators)
    {
        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ ifeature, ++ features)
        {
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass:  total_columns += feature.classes() - 1; break;
            case feature_type::mclass:  total_columns += feature.classes(); break;
            default:                    total_columns += size(feature.dims()); break;
            }
        }
        ++ generators;
    }

    m_column_mapping.resize(total_columns, 3);
    m_feature_mapping.resize(features, 5);
    m_generator_mapping.resize(generators, 1);

    tensor_size_t offset_features = 0, offset_columns = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        const auto old_offset_columns = offset_columns;

        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ ifeature, ++ offset_features)
        {
            m_feature_mapping(offset_features, 0) = index;
            m_feature_mapping(offset_features, 1) = ifeature;

            tensor_size_t dim1 = 1, dim2 = 1, dim3 = 1, columns = 0;
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass:
                columns = feature.classes() - 1;
                break;
            case feature_type::mclass:
                dim1 = feature.classes();
                columns = feature.classes();
                break;
            default:
                dim1 = feature.dims()[0];
                dim2 = feature.dims()[1];
                dim3 = feature.dims()[2];
                columns = size(feature.dims());
                break;
            }
            m_feature_mapping(offset_features, 2) = dim1;
            m_feature_mapping(offset_features, 3) = dim2;
            m_feature_mapping(offset_features, 4) = dim3;

            for (tensor_size_t icolumn = 0; icolumn < columns; ++ icolumn, ++ offset_columns)
            {
                m_column_mapping(offset_columns, 0) = index;
                m_column_mapping(offset_columns, 1) = icolumn;
                m_column_mapping(offset_columns, 2) = offset_features;
            }
        }

        m_generator_mapping(index ++, 0) = offset_columns - old_offset_columns;
    }

    update_stats();
}

void dataset_generator_t::update_stats()
{
    std::vector<tensor_size_t> sclasss, mclasss, scalars, structs;
    for (tensor_size_t i = 0, size = features(); i < size; ++ i)
    {
        switch (const auto& feature = this->feature(i); feature.type())
        {
        case feature_type::sclass:
            sclasss.push_back(i);
            break;

        case feature_type::mclass:
            mclasss.push_back(i);
            break;

        default:
            (::nano::size(feature.dims()) > 1 ? structs : scalars).push_back(i);
            break;
        }
    }

    m_select_stats.m_sclass_features = map_tensor(sclasss.data(), make_dims(static_cast<tensor_size_t>(sclasss.size())));
    m_select_stats.m_mclass_features = map_tensor(mclasss.data(), make_dims(static_cast<tensor_size_t>(mclasss.size())));
    m_select_stats.m_scalar_features = map_tensor(scalars.data(), make_dims(static_cast<tensor_size_t>(scalars.size())));
    m_select_stats.m_struct_features = map_tensor(structs.data(), make_dims(static_cast<tensor_size_t>(structs.size())));
}

tensor_size_t dataset_generator_t::features() const
{
    return m_feature_mapping.size<0>();
}

feature_t dataset_generator_t::feature(tensor_size_t feature) const
{
    return byfeature(feature)->feature(m_feature_mapping(feature, 1));
}

tensor_size_t dataset_generator_t::columns() const
{
    return m_column_mapping.size<0>();
}

tensor_size_t dataset_generator_t::column2feature(tensor_size_t column) const
{
    return m_column_mapping(column, 2);
}

sclass_cmap_t dataset_generator_t::select(indices_cmap_t samples, tensor_size_t feature, sclass_mem_t& buffer) const
{
    check(samples);
    handle_sclass(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size());
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

mclass_cmap_t dataset_generator_t::select(indices_cmap_t samples, tensor_size_t feature, mclass_mem_t& buffer) const
{
    check(samples);
    handle_mclass(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size(), m_feature_mapping(feature, 2));
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

scalar_cmap_t dataset_generator_t::select(indices_cmap_t samples, tensor_size_t feature, scalar_mem_t& buffer) const
{
    check(samples);
    handle_scalar(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size());
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

struct_cmap_t dataset_generator_t::select(indices_cmap_t samples, tensor_size_t feature, struct_mem_t& buffer) const
{
    check(samples);
    handle_struct(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size(),
        m_feature_mapping(feature, 2), m_feature_mapping(feature, 3), m_feature_mapping(feature, 4));
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

tensor2d_map_t dataset_generator_t::flatten(indices_cmap_t samples, tensor2d_t& buffer) const
{
    check(samples);

    const auto storage = resize_and_map(buffer, samples.size(), columns());

    tensor_size_t offset = 0, index = 0;
    for (const auto& generator : m_generators)
    {
        generator->flatten(samples, storage, offset);
        offset += m_generator_mapping(index ++, 0);
    }
    return storage;
}

feature_t dataset_generator_t::target() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return feature_t{};

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            return feature;
        });
    }
}

tensor3d_dims_t dataset_generator_t::target_dims() const
{
    switch (m_dataset.type())
    {
    case task_type::unsupervised:
        return make_dims(0, 0, 0);

    default:
        return m_dataset.visit_target([] (const feature_t& feature, const auto&, const auto&)
        {
            switch (feature.type())
            {
            case feature_type::sclass:  return make_dims(feature.classes(), 1, 1); // NOLINT(bugprone-branch-clone)
            case feature_type::mclass:  return make_dims(feature.classes(), 1, 1);
            default:                    return feature.dims();
            }
        });
    }
}

tensor4d_map_t dataset_generator_t::targets(indices_cmap_t samples, tensor4d_t& buffer) const
{
    check(samples);

    if (m_dataset.type() == task_type::unsupervised)
    {
        critical0("dataset_generator_t: targets are not available for unsupervised datasets!");
    }

    return m_dataset.visit_target([&] (const feature_t& feature, const auto& data, const auto& mask)
    {
        return loop_samples(data, mask, samples,
        [&] (auto it)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (; it; ++ it)
            {
                if (const auto [index, given, label] = *it; given)
                {
                    storage.array(index).setConstant(-1.0);
                    storage.array(index)(static_cast<tensor_size_t>(label)) = +1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_map_t{storage};
        },
        [&] (auto it)
        {
            const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
            for (; it; ++ it)
            {
                if (const auto [index, given, hits] = *it; given)
                {
                    storage.array(index) = hits.array().template cast<scalar_t>() * 2.0 - 1.0;
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_map_t{storage};
        },
        [&] (auto it)
        {
            const auto [dim1, dim2, dim3] = feature.dims();
            const auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            for (; it; ++ it)
            {
                if (const auto [index, given, values] = *it; given)
                {
                    storage.array(index) = values.array().template cast<scalar_t>();
                }
                else
                {
                    storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                }
            }
            return tensor4d_map_t{storage};
        });
    });
}

tensor1d_t dataset_generator_t::sample_weights(indices_cmap_t samples, const targets_stats_t& targets_stats) const
{
    check(samples);

    if (m_dataset.type() == task_type::unsupervised)
    {
        tensor1d_t weights(samples.size());
        weights.full(1.0);
        return weights;
    }
    else
    {
        return m_dataset.visit_target([&] (const feature_t& feature, const auto& data, const auto& mask)
        {
            return loop_samples(data, mask, samples,
            [&] (auto it)
            {
                const auto* pstats = std::get_if<sclass_stats_t>(&targets_stats);
                critical(
                    pstats == nullptr ||
                    pstats->classes() != feature.classes(),
                    "dataset_generator_t: mis-matching single-label targets statistics, expecting ",
                    feature.classes(), " classes, got ",
                    pstats == nullptr ? tensor_size_t(0) : pstats->classes(), " instead!");

                return pstats->sample_weights(feature, it);
            },
            [&] (auto it)
            {
                const auto* pstats = std::get_if<mclass_stats_t>(&targets_stats);
                critical(
                    pstats == nullptr ||
                    pstats->classes() != feature.classes(),
                    "dataset_generator_t: mis-matching multi-label targets statistics, expecting ",
                    feature.classes(), " classes, got ",
                    pstats == nullptr ? tensor_size_t(0) : pstats->classes(), " instead!");

                return pstats->sample_weights(feature, it);
            },
            [&] (auto)
            {
                tensor1d_t weights(samples.size());
                weights.full(1.0);
                return weights;
            });
        });
    }
}

void dataset_generator_t::undrop() const
{
    for (const auto& generator : m_generators)
    {
        generator->undrop();
    }
}

void dataset_generator_t::drop(tensor_size_t feature) const
{
    byfeature(feature)->drop(m_feature_mapping(feature, 1));
}

void dataset_generator_t::unshuffle() const
{
    for (const auto& generator : m_generators)
    {
        generator->unshuffle();
    }
}

void dataset_generator_t::shuffle(tensor_size_t feature) const
{
    byfeature(feature)->shuffle(m_feature_mapping(feature, 1));
}

indices_t dataset_generator_t::shuffled(indices_cmap_t samples, tensor_size_t feature) const
{
    return byfeature(feature)->shuffled(samples, m_feature_mapping(feature, 1));
}

const rgenerator_t& dataset_generator_t::byfeature(tensor_size_t feature) const
{
    check(feature);

    return m_generators[static_cast<size_t>(m_feature_mapping(feature, 0))];
}

void dataset_generator_t::check(tensor_size_t feature) const
{
    critical(
        feature < 0 || feature >= features(),
        "dataset_generator_t: invalid feature index, expecting in [0, ", features(), "), got ", feature, "!");
}

void dataset_generator_t::check(indices_cmap_t samples) const
{
    critical(
        samples.min() < 0 ||
        samples.max() > m_dataset.samples(),
        "dataset_generator_t: invalid sample range, expecting in [0, ", m_dataset.samples(), "), got ",
        "[", samples.min(), ", ", samples.max(), ")!");
}
