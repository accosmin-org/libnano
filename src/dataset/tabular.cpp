#include <fstream>
#include "tabular.h"
#include <nano/logger.h>
#include <nano/mlearn.h>
#include <nano/random.h>

using namespace nano;

tabular_dataset_t::tabular_dataset_t(std::vector<feature_t> features) :
    m_features(std::move(features))
{
}

json_t tabular_dataset_t::config() const
{
    json_t json;

    json["delim"] = m_delim;
    json["folds"] = strcat(m_splits.size(), "[1,100]");
    if (m_target < m_features.size())
    {
        json["target"] = m_features[m_target].name();
    }
    json["train_per"] = strcat(m_train_per, "[10%,90%]");

    auto&& json_paths = (json["paths"] = json_t::array());
    for (const auto& path : m_paths)
    {
        json_paths.push_back(path);
    }

    auto&& json_features = (json["features"] = json_t::array());
    for (const auto& feature : m_features)
    {
        json_features.push_back(feature.config());
    }

    return json;
}

void tabular_dataset_t::config(const json_t& json)
{
    require_json(json, "delim");
    require_json(json, "folds");
    require_json(json, "paths");
    require_json(json, "features");

    from_json(json, "delim", m_delim);
    from_json_range(json, "train_per", m_train_per, 10, 90);

    size_t folds = 0;
    from_json_range(json, "folds", folds, 1, 100);
    m_splits = std::vector<split_t>(folds, split_t{});

    m_paths.clear();
    for (const auto& json_path : json["paths"])
    {
        m_paths.push_back(json_path.get<string_t>());
    }

    m_features.clear();
    for (const auto& json_feature : json["features"])
    {
        auto feature = feature_t{};
        feature.config(json_feature);
        m_features.push_back(feature);
    }

    if (m_features.empty())
    {
        throw std::invalid_argument("at least one feature needs to be defined");
    }

    if (json.count("target"))
    {
        const auto target_name = json["target"].get<string_t>();

        const auto op = [&] (const auto& feature) { return feature.name() == target_name; };
        const auto it = std::find_if(m_features.begin(), m_features.end(), op);

        if (it == m_features.end())
        {
            throw std::invalid_argument("the target feature not matching any of the features' names");
        }

        if (!it->placeholder().empty())
        {
            throw std::invalid_argument("the target feature cannot be optional");
        }

        m_target = static_cast<size_t>(std::distance(m_features.begin(), it));
    }
    else
    {
        m_target = string_t::npos;
    }
}

bool tabular_dataset_t::load()
{
    if (m_features.empty())
    {
        log_error() << "tabular dataset: need to set at least one feature!";
        return false;
    }

    tensor_size_t data_size = 0;
    for (const auto& path : m_paths)
    {
        data_size += lines(path);
    }

    tensor_size_t n_inputs = 0, n_targets = 0;
    for (size_t f = 0; f < m_features.size(); ++ f)
    {
        const auto& feature = m_features[f];
        if (f == m_target)
        {
            n_targets += feature.discrete() ?
                static_cast<tensor_size_t>(feature.labels().size()):
                tensor_size_t(1);
        }
        else
        {
            n_inputs ++;
        }
    }

    m_inputs.resize(data_size, n_inputs, 1, 1);
    m_targets.resize(data_size, n_targets, 1, 1);

    tensor_size_t row_offset = 0;
    for (const auto& path : m_paths)
    {
        log_info() << "tabular dataset: reading " << path << "...";
        if (!parse(path, row_offset))
        {
            return false;
        }
        log_info() << "tabular dataset: read " << row_offset << " samples!";
    }

    if (row_offset != samples())
    {
        log_error() << "tabular dataset: read " << row_offset << " samples, expecting " << samples() << "!";
        return false;
    }

    return split();
}

void tabular_dataset_t::store(const tensor_size_t row, const size_t f, const scalar_t value)
{
    if (f != m_target)
    {
        m_inputs(row, static_cast<tensor_size_t>((f > m_target) ? (f - 1) : (f)), 0, 0) = value;
    }
    else
    {
        assert(m_target < m_features.size());
        assert(m_targets.dims() == make_dims(samples(), 1, 1, 1));

        m_targets(row, 0, 0, 0) = value;
    }
}

void tabular_dataset_t::store(const tensor_size_t row, const size_t f, const tensor_size_t category)
{
    if (f != m_target)
    {
        m_inputs(row, static_cast<tensor_size_t>((f > m_target) ? (f - 1) : (f)), 0, 0) = static_cast<scalar_t>(category);
    }
    else
    {
        const auto& feature = m_features[f];
        const auto labels_size = static_cast<tensor_size_t>(feature.labels().size());

        assert(feature.discrete());
        assert(category < labels_size);
        assert(m_target < m_features.size());
        assert(m_targets.dims() == make_dims(samples(), labels_size, 1, 1));

        m_targets.tensor(row) = class_target(labels_size, category);
    }
}

bool tabular_dataset_t::parse(const string_t& path, tensor_size_t& row_offset)
{
    string_t line;
    std::ifstream stream(path);
    for (tensor_size_t row = 0; std::getline(stream, line); )
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        if (row_offset >= samples())
        {
            log_error() << "tabular dataset: too many samples, expecting " << samples() << "!";
            return false;
        }

        for (auto tokenizer = tokenizer_t{line, m_delim.c_str()}; tokenizer; ++ tokenizer)
        {
            if (tokenizer.count() > m_features.size())
            {
                log_error() << "tabular dataset: invalid line " << path << ":" << row
                    << ", expecting " << m_features.size() << " tokens!";
                return false;
            }

            const auto f = tokenizer.count() - 1;
            const auto token = tokenizer.get();
            const auto& feature = m_features[f];

            if (token == feature.placeholder())
            {
                assert(f != m_target);
                store(row_offset, f, feature_t::placeholder_value());
            }
            else if (!feature.discrete())
            {
                try
                {
                    store(row_offset, f, from_string<scalar_t>(token));
                }
                catch (std::exception& e)
                {
                    log_error() << "tabular dataset: invalid line " << path << ":" << row
                        << ", expecting arithmetic token [" << token << "] for feature " << feature.name() << "!";
                    return false;
                }
            }
            else
            {
                const auto& labels = feature.labels();

                const auto it = std::find(labels.begin(), labels.end(), token);
                if (it == labels.end())
                {
                    log_error() << "tabular dataset: invalid line " << path << ":" << row
                        << ", invalid label [" << token << "] for feature " << feature.name() << "!";
                    return false;
                }

                store(row_offset, f, std::distance(labels.begin(), it));
            }
        }

        ++ row;
        ++ row_offset;
    }

    return true;
}

tensor_size_t tabular_dataset_t::lines(const string_t& path)
{
    string_t line;
    tensor_size_t count = 0;
    std::ifstream stream(path);

    while (std::getline(stream, line))
    {
        if (!line.empty() && line[0] != '#')
        {
            ++ count;
        }
    }

    return count;
}


void tabular_dataset_t::split(split_t& split) const
{
    std::tie(split.m_tr_indices, split.m_vd_indices, split.m_te_indices) = nano::split3(
        m_targets.size<0>(), m_train_per, (100 - m_train_per) / 2);
}

bool tabular_dataset_t::split()
{
    for (auto& split : m_splits)
    {
        this->split(split);
    }

    return true;
}

size_t tabular_dataset_t::folds() const
{
    return m_splits.size();
}

size_t tabular_dataset_t::ifeatures() const
{
    const auto size = m_features.size();
    return (m_target == string_t::npos) ? size : (size > 0 ? (size - 1) : 0);
}

feature_t tabular_dataset_t::ifeature(size_t index) const
{
    if (index >= m_target) ++ index;
    return m_features.at(index);
}

feature_t tabular_dataset_t::tfeature() const
{
    return m_features.at(m_target);
}

indices_t& tabular_dataset_t::indices(const fold_t& fold)
{
    auto& split = m_splits.at(fold.m_index);
    return split.indices(fold);
}

const indices_t& tabular_dataset_t::indices(const fold_t& fold) const
{
    const auto& split = m_splits.at(fold.m_index);
    return split.indices(fold);
}

tensor4d_t tabular_dataset_t::inputs(const fold_t& fold) const
{
    return index(m_inputs, indices(fold));
}

tensor4d_t tabular_dataset_t::targets(const fold_t& fold) const
{
    return index(m_targets, indices(fold));
}

void tabular_dataset_t::shuffle(const fold_t& fold)
{
    auto& indices = this->indices(fold);
    std::shuffle(begin(indices), end(indices), make_rng());
}

tensor4d_t tabular_dataset_t::index(const tensor4d_t& data, const indices_t& indices)
{
    assert(indices.minCoeff() >= 0 && indices.maxCoeff() < data.size<0>());

    tensor4d_t idata(indices.size(), data.size<1>(), data.size<2>(), data.size<3>());
    for (tensor_size_t i = 0, size = indices.size(); i < size; ++ i)
    {
        idata.tensor(i) = data.tensor(indices(i));
    }

    return idata;
}
