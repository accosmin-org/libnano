#include <mutex>
#include <fstream>
#include <nano/tabular.h>
#include <nano/util/logger.h>
#include <nano/util/tokenizer.h>

#include <nano/tabular/iris.h>
#include <nano/tabular/wine.h>
#include <nano/tabular/adult.h>
#include <nano/tabular/abalone.h>
#include <nano/tabular/poker_hand.h>
#include <nano/tabular/forest_fires.h>
#include <nano/tabular/breast_cancer.h>
#include <nano/tabular/bank_marketing.h>

using namespace nano;

template <typename toperator>
static auto parse(const string_t& path, const char skip, bool header, const toperator& op)
{
    string_t line;
    tensor_size_t line_index = 0;
    for (std::ifstream stream(path); std::getline(stream, line); ++ line_index)
    {
        if (header && line_index == 0)
        {
            header = false;
        }
        else if (!line.empty() && line[0] != skip)
        {
            if (!op(line, line_index))
            {
                return false;
            }
        }
    }

    return true;
}

void tabular_dataset_t::csvs(std::vector<csv_t> csvs)
{
    m_csvs = std::move(csvs);
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
        ::parse(csv.m_path, csv.m_skip, csv.m_header, [&] (const string_t&, const tensor_size_t)
        {
            ++ data_size;
            return true;
        });
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
    tensor_size_t row = 0;
    for (const auto& csv : m_csvs)
    {
        log_info() << "tabular dataset: reading " << csv.m_path << "...";

        const auto old_row = row;
        if (!::parse(csv.m_path, csv.m_skip, csv.m_header, [&] (const string_t& line, const tensor_size_t line_index)
            {
                return this->parse(csv.m_path, line, csv.m_delim, line_index, row ++);
            }))
        {
            return false;
        }

        const auto samples_read = row - old_row;
        if (csv.m_expected > 0 && samples_read != csv.m_expected)
        {
            log_error() << "tabular dataset: read " << samples_read << ", expecting " << csv.m_expected << " samples!";
            return false;
        }

        log_info() << "tabular dataset: read " << row << " samples!";
    }

    if (row != data_size)
    {
        log_error() << "tabular dataset: read " << row << " samples, expecting " << data_size << "!";
        return false;
    }

    // generate and check splits
    for (size_t f = 0; f < folds(); ++ f)
    {
        auto& split = (this->split(f) = make_split());
        if (!split.valid(samples()))
        {
            log_error() << "tabular dataset: invalid split!";
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
        assert(m_targets.dims() == make_dims(m_targets.size<0>(), 1, 1, 1));

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
        assert(m_targets.dims() == make_dims(m_targets.size<0>(), labels_size, 1, 1));

        m_targets.tensor(row) = class_target(labels_size, category);
    }
}

bool tabular_dataset_t::parse(const string_t& path, const string_t& line, const string_t& delim,
    const tensor_size_t line_index, const tensor_size_t row)
{
    if (row >= m_inputs.size<0>())
    {
        log_error() << "tabular dataset: too many samples, expecting " << m_inputs.size<0>() << "!";
        return false;
    }

    for (auto tokenizer = tokenizer_t{line, delim.c_str()}; tokenizer; ++ tokenizer)
    {
        if (tokenizer.count() > m_features.size())
        {
            log_error() << "tabular dataset: invalid line " << path << ":" << line_index
                << ", expecting " << m_features.size() << " tokens!";
            return false;
        }

        const auto f = tokenizer.count() - 1;
        const auto token = tokenizer.get();
        const auto& feature = m_features[f];

        if (token == feature.placeholder())
        {
            assert(f != m_target);
            store(row, f, feature_t::placeholder_value());
        }
        else if (!feature.discrete())
        {
            try
            {
                store(row, f, from_string<scalar_t>(token));
            }
            catch (std::exception& e)
            {
                log_error() << "tabular dataset: invalid line " << path << ":" << line_index
                    << ", expecting arithmetic token [" << token << "] for feature [" << feature.name() << "]!";
                return false;
            }
        }
        else
        {
            const auto& labels = feature.labels();

            const auto it = std::find(labels.begin(), labels.end(), token);
            if (it == labels.end())
            {
                log_error() << "tabular dataset: invalid line " << path << ":" << line_index
                    << ", invalid label [" << token << "] for feature [" << feature.name() << "]!";
                return false;
            }

            store(row, f, std::distance(labels.begin(), it));
        }
    }

    return true;
}

size_t tabular_dataset_t::ifeatures() const
{
    const auto size = m_features.size();
    return (m_target == string_t::npos) ? size : (size > 0 ? (size - 1) : 0);
}

feature_t tabular_dataset_t::ifeature(size_t index) const
{
    if (index >= m_target)
    {
        ++ index;
    }
    return m_features.at(index);
}

feature_t tabular_dataset_t::tfeature() const
{
    return m_features.at(m_target);
}

tensor_size_t tabular_dataset_t::samples() const
{
    return m_inputs.size<0>();
}

tensor_size_t tabular_dataset_t::samples(const fold_t& fold) const
{
    return indices(fold).size();
}

tensor4d_t tabular_dataset_t::inputs(const fold_t& fold) const
{
    return m_inputs.indexed<scalar_t>(indices(fold));
}

tensor4d_t tabular_dataset_t::targets(const fold_t& fold) const
{
    return m_targets.indexed<scalar_t>(indices(fold));
}

tensor4d_t tabular_dataset_t::inputs(const fold_t& fold, const tensor_size_t begin, const tensor_size_t end) const
{
    return m_inputs.indexed<scalar_t>(indices(fold).segment(begin, end - begin));
}

tensor4d_t tabular_dataset_t::targets(const fold_t& fold, const tensor_size_t begin, const tensor_size_t end) const
{
    return m_targets.indexed<scalar_t>(indices(fold).segment(begin, end - begin));
}

tabular_dataset_factory_t& tabular_dataset_t::all()
{
    static tabular_dataset_factory_t manager;

    static std::once_flag flag;
    std::call_once(flag, [] ()
    {
        manager.add<iris_dataset_t>("iris",
            "classify flowers from physical measurements of the sepal and petal (Fisher, 1936)");
        manager.add<wine_dataset_t>("wine",
            "predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)");
        manager.add<adult_dataset_t>("adult",
            "predict if a person makes more than 50K per year (Kohavi & Becker, 1994)");
        manager.add<abalone_dataset_t>("abalone",
            "predict the age of abalone from physical measurements (Waugh, 1995)");
        manager.add<poker_hand_dataset_t>("poker-hand",
            "predict the poker hand from 5 cards (Cattral, Oppacher & Deugo, 2002)");
        manager.add<forest_fires_dataset_t>("forest-fires",
            "predict the burned area of the forest (Cortez & Morais, 2007)");
        manager.add<breast_cancer_dataset_t>("breast-cancer",
            "diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992)");
        manager.add<bank_marketing_dataset_t>("bank-marketing",
            "predict if a client has subscribed a term deposit (Moro, Laureano & Cortez, 2011)");
    });

    return manager;
}
