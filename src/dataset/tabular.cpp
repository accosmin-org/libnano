#include <fstream>
#include <nano/logger.h>
#include <nano/mlearn.h>
#include <nano/dataset/tabular.h>

using namespace nano;

static tensor_size_t lines(const string_t& path, const char skip, bool header)
{
    string_t line;
    tensor_size_t count = 0;
    std::ifstream stream(path);
    while (std::getline(stream, line))
    {
        if (!line.empty() && line[0] != skip)
        {
            if (header && count == 0)
            {
                header = false;
            }
            else
            {
                ++ count;
            }
        }
    }

    return count;
}

static tensor4d_t index4d(const tensor4d_t& data, const indices_t& indices)
{
    assert(indices.minCoeff() >= 0 && indices.maxCoeff() < data.size<0>());

    tensor4d_t idata(indices.size(), data.size<1>(), data.size<2>(), data.size<3>());
    for (tensor_size_t i = 0, size = indices.size(); i < size; ++ i)
    {
        idata.tensor(i) = data.tensor(indices(i));
    }

    return idata;
}

void tabular_dataset_t::csvs(std::vector<csv_t> csvs)
{
    m_csvs = std::move(csvs);
}

void tabular_dataset_t::folds(const size_t folds)
{
    m_splits = std::vector<split_t>(folds, split_t{});
}

void tabular_dataset_t::features(std::vector<feature_t> features, const size_t target)
{
    m_target = target;
    m_features = std::move(features);
}

bool tabular_dataset_t::load()
{
    // check features
    if (m_features.empty())
    {
        log_error() << "tabular dataset: need to set at least one feature!";
        return false;
    }

    if (m_target != string_t::npos && m_target >= m_features.size())
    {
        log_error() << "tabular dataset: the target feature (" << m_target
            << ") is not valid, expecting in the [0, " << m_features.size() << ") range!";
        return false;
    }

    if (m_target != string_t::npos && m_features[m_target].optional())
    {
        log_error() << "tabular dataset: the target feature (" << m_features[m_target].name()
            << ") cannot be optional!";
        return false;
    }

    // allocate storage
    tensor_size_t data_size = 0;
    for (const auto& csv : m_csvs)
    {
        data_size += lines(csv.m_path, csv.m_skip, csv.m_header);
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

    if (data_size == 0)
    {
        log_error() << "tabular dataset: no data to read, check paths!";
        return false;
    }

    m_inputs.resize(data_size, n_inputs, 1, 1);
    m_targets.resize(data_size, n_targets, 1, 1);

    // load data
    tensor_size_t row_offset = 0;
    for (const auto& csv : m_csvs)
    {
        log_info() << "tabular dataset: reading " << csv.m_path << "...";

        const auto old_row_offset = row_offset;
        if (!parse(csv.m_path, row_offset, csv.m_delim, csv.m_skip, csv.m_header))
        {
            return false;
        }

        const auto samples_read = row_offset - old_row_offset;
        if (csv.m_expected > 0 && samples_read != csv.m_expected)
        {
            log_error() << "tabular dataset: read " << samples_read << ", expecting " << csv.m_expected << " samples!";
            return false;
        }

        log_info() << "tabular dataset: read " << row_offset << " samples!";
    }

    if (row_offset != samples())
    {
        log_error() << "tabular dataset: read " << row_offset << " samples, expecting " << samples() << "!";
        return false;
    }

    // generate and check splits
    for (auto& split : m_splits)
    {
        this->split(samples(), split);

        if (split.m_tr_indices.size() == 0)
        {
            log_error() << "tabular dataset: invalid training set, expecting at least on element!";
            return false;
        }
        if (split.m_vd_indices.size() == 0)
        {
            log_error() << "tabular dataset: invalid validation set, expecting at least on element!";
            return false;
        }
        if (split.m_te_indices.size() == 0)
        {
            log_error() << "tabular dataset: invalid test set, expecting at least on element!";
            return false;
        }

        if (split.m_tr_indices.size() + split.m_vd_indices.size() + split.m_te_indices.size() != samples())
        {
            log_error() << "tabular dataset: invalid split, got "
                << (split.m_tr_indices.size() + split.m_vd_indices.size() + split.m_te_indices.size())
                << " samples, expecting " << samples() << "!";
            return false;
        }

        if (split.m_tr_indices.minCoeff() < 0 || split.m_tr_indices.maxCoeff() >= samples())
        {
            log_error() << "tabular dataset: invalid training index, expected in the [0, " << samples() << ") range!";
            return false;
        }
        if (split.m_vd_indices.minCoeff() < 0 || split.m_vd_indices.maxCoeff() >= samples())
        {
            log_error() << "tabular dataset: invalid validation index, expected in the [0, " << samples() << ") range!";
            return false;
        }
        if (split.m_te_indices.minCoeff() < 0 || split.m_te_indices.maxCoeff() >= samples())
        {
            log_error() << "tabular dataset: invalid test index, expected in the [0, " << samples() << ") range!";
            return false;
        }
    }

    return true;
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

bool tabular_dataset_t::parse(const string_t& path, tensor_size_t& row_offset,
    const string_t& delim, const char skip, bool header)
{
    string_t line;
    std::ifstream stream(path);
    for (tensor_size_t row = 0; std::getline(stream, line); )
    {
        if (line.empty() || line[0] == skip)
        {
            continue;
        }

        if (header && row == 0)
        {
            header = false;
            continue;
        }

        if (row_offset >= samples())
        {
            log_error() << "tabular dataset: too many samples, expecting " << samples() << "!";
            return false;
        }

        for (auto tokenizer = tokenizer_t{line, delim.c_str()}; tokenizer; ++ tokenizer)
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
    return index4d(m_inputs, indices(fold));
}

tensor4d_t tabular_dataset_t::targets(const fold_t& fold) const
{
    return index4d(m_targets, indices(fold));
}

void tabular_dataset_t::shuffle(const fold_t& fold)
{
    auto& indices = this->indices(fold);
    std::shuffle(begin(indices), end(indices), make_rng());
}
