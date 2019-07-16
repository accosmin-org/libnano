#include <fstream>
#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

class CSVFixture final : public tabular_dataset_t
{
public:

    static auto data_path() { return "test_dataset_tabular_data.csv"; }
    static auto test_path() { return "test_dataset_tabular_test.csv"; }

    CSVFixture()
    {
        std::remove(data_path());
        std::remove(test_path());

        write_data(data_path());
        write_test(test_path());
    }

    ~CSVFixture() override
    {
        std::remove(data_path());
        std::remove(test_path());
    }

    void split(
        const tensor_size_t tr_begin, const tensor_size_t tr_end,
        const tensor_size_t vd_begin, const tensor_size_t vd_end,
        const tensor_size_t te_begin, const tensor_size_t te_end)
    {
        m_tr_begin = tr_begin; m_tr_end = tr_end;
        m_vd_begin = vd_begin; m_vd_end = vd_end;
        m_te_begin = te_begin; m_te_end = te_end;
    }

    json_t config() const override
    {
        return json_t{};
    }

    void config(const json_t&) override
    {
    }

    void split(const tensor_size_t samples, split_t& split) const override
    {
        UTEST_CHECK_EQUAL(samples, 30);

        split.m_tr_indices = indices_t::LinSpaced(m_tr_end - m_tr_begin, m_tr_begin, m_tr_end);
        split.m_vd_indices = indices_t::LinSpaced(m_vd_end - m_vd_begin, m_vd_begin, m_vd_end);
        split.m_te_indices = indices_t::LinSpaced(m_te_end - m_te_begin, m_te_begin, m_te_end);
    }

    static void write_data(const char* path)
    {
        std::ofstream os(path);
        write(os, 1, 20);
        UTEST_REQUIRE(os);
    }

    static void write_test(const char* path)
    {
        std::ofstream os(path);
        write(os, 21, 10);
        UTEST_REQUIRE(os);
    }

    static void check(const scalar_t value, int row, const int col)
    {
        ++ row;
        switch (col)
        {
        case 0:     check(value, row); break;
        case 1:     check(value, (row % 2 == 0) ? feature_t::placeholder_value() : (3.0 - 0.2 * row)); break;
        case 2:     check(value, row % 3); break;
        case 3:     check(value, (row % 4 == 0) ? feature_t::placeholder_value() : (row % 2)); break;
        default:    UTEST_REQUIRE(false);
        }
    }

    static void check(const scalar_t value, const scalar_t ground)
    {
        UTEST_CHECK_EQUAL(std::isfinite(value), std::isfinite(ground));

        if (std::isfinite(value))
        {
            UTEST_CHECK_CLOSE(value, ground, 1e-8);
        }
    }

    static void write(std::ostream& os, const int begin, const int size)
    {
        for (auto index = begin; index < begin + size; ++ index)
        {
            os << index << ",";
            (index % 2 == 0) ? (os << "?,") : (os << (3.0 - 0.2 * index) << ",");
            os << "cate" << (index % 3) << ",";
            (index % 4 == 0) ? (os << "?,") : (os << "cate_opt" << (index % 2) << ",");
            os << "\n";

            if (index % 7 == 0) { os << "\n"; }
            if (index % 9 == 0) { os << "#\n"; }
        }
    }

private:

    // attributes
    tensor_size_t   m_tr_begin{0}, m_tr_end{20};    ///<
    tensor_size_t   m_vd_begin{20}, m_vd_end{26};   ///<
    tensor_size_t   m_te_begin{26}, m_te_end{30};   ///<
};

static auto make_feature_cont()
{
    const auto feature = feature_t::make_scalar("cont");

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(!feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto make_feature_cont_opt()
{
    const auto feature = feature_t::make_scalar("cont_opt", "?");

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto make_feature_cate()
{
    const auto feature = feature_t::make_discrete("cate", {"cate0", "cate1", "cate2"});

    UTEST_CHECK(feature.discrete());
    UTEST_CHECK(!feature.optional());
    UTEST_CHECK_EQUAL(feature.label(0), "cate0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate1");
    UTEST_CHECK_EQUAL(feature.label(2), "cate2");
    UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
    UTEST_CHECK_THROW(feature.label(+3), std::out_of_range);
    UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());

    return feature;
}

static auto make_feature_cate_opt()
{
    const auto feature = feature_t::make_discrete("cate_opt", {"cate_opt0", "cate_opt1"}, "?");

    UTEST_CHECK(feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
    UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
    UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
    UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());

    return feature;
}

static const auto feature_cont = make_feature_cont();
static const auto feature_cont_opt = make_feature_cont_opt();
static const auto feature_cate = make_feature_cate();
static const auto feature_cate_opt = make_feature_cate_opt();

UTEST_BEGIN_MODULE(test_dataset_tabular)

UTEST_CASE(empty)
{
    auto dataset = CSVFixture{};

    UTEST_CHECK_EQUAL(dataset.ifeatures(), 0);
    UTEST_CHECK_THROW(dataset.ifeature(0), std::out_of_range);
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);
}

UTEST_CASE(config_no_target)
{
    auto dataset = CSVFixture{};

    dataset.folds(7);
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});

    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate);
    UTEST_CHECK_EQUAL(dataset.ifeature(3), feature_cate_opt);
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);
}

UTEST_CASE(config_with_target)
{
    auto dataset = CSVFixture{};

    dataset.folds(7);
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 0);

    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cate);
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cont);
}

UTEST_CASE(noload_no_data)
{
    auto dataset = CSVFixture{};

    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 4);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_no_features)
{
    auto dataset = CSVFixture{};

    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_invalid_target)
{
    auto dataset = CSVFixture{};

    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 4);
    UTEST_CHECK(!dataset.load());

    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 1);
    UTEST_CHECK(!dataset.load());

    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 3);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_invalid_splits)
{
    auto dataset = CSVFixture{};

    dataset.split(-1, 10, 20, 26, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(20, 31, 20, 26, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, -1, 1, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 30, 31, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, -1, 1);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 29, 31);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 0, 20, 26, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 20, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 26, 26);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 27, 26, 30);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 26, 29);
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(load_no_target)
{
    auto dataset = CSVFixture{};

    dataset.folds(3);
    dataset.delim(",");
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt});

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate);
    UTEST_CHECK_EQUAL(dataset.ifeature(3), feature_cate_opt);
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(20, 4, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(6, 4, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(4, 4, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(20, 0, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(6, 0, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(4, 0, 1, 1));

        for (auto index = 0; index < 30; ++ index)
        {
            const auto row = (index < 20) ? index : (index < 26 ? (index - 20) : (index - 26));
            const auto& inputs = (index < 20) ? tr_inputs : (index < 26 ? vd_inputs : te_inputs);

            CSVFixture::check(inputs(row, 0, 0, 0), index, 0);
            CSVFixture::check(inputs(row, 1, 0, 0), index, 1);
            CSVFixture::check(inputs(row, 2, 0, 0), index, 2);
            CSVFixture::check(inputs(row, 3, 0, 0), index, 3);
        }
    }
}

UTEST_CASE(load_with_cont_target)
{
    auto dataset = CSVFixture{};

    dataset.folds(2);
    dataset.delim(",");
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 0);

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 2);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cate);
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cont);

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(20, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(6, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(4, 3, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(20, 1, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(6, 1, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(4, 1, 1, 1));

        for (auto index = 0; index < 30; ++ index)
        {
            const auto row = (index < 20) ? index : (index < 26 ? (index - 20) : (index - 26));
            const auto& inputs = (index < 20) ? tr_inputs : (index < 26 ? vd_inputs : te_inputs);
            const auto& targets = (index < 20) ? tr_targets : (index < 26 ? vd_targets : te_targets);

            CSVFixture::check(targets(row, 0, 0, 0), index, 0);
            CSVFixture::check(inputs(row, 0, 0, 0), index, 1);
            CSVFixture::check(inputs(row, 1, 0, 0), index, 2);
            CSVFixture::check(inputs(row, 2, 0, 0), index, 3);
        }
    }
}

UTEST_CASE(load_with_cate_target)
{
    auto dataset = CSVFixture{};

    dataset.folds(7);
    dataset.delim(",");
    dataset.paths({CSVFixture::data_path(), CSVFixture::test_path()});
    dataset.features({feature_cont, feature_cont_opt, feature_cate, feature_cate_opt}, 2);

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont);
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt);
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt);
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cate);

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_inputs.dims(), make_dims(20, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_inputs.dims(), make_dims(6, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_inputs.dims(), make_dims(4, 3, 1, 1));

        UTEST_CHECK_EQUAL(tr_targets.dims(), make_dims(20, 3, 1, 1));
        UTEST_CHECK_EQUAL(vd_targets.dims(), make_dims(6, 3, 1, 1));
        UTEST_CHECK_EQUAL(te_targets.dims(), make_dims(4, 3, 1, 1));

        for (auto index = 0; index < 30; ++ index)
        {
            const auto row = (index < 20) ? index : (index < 26 ? (index - 20) : (index - 26));
            const auto& inputs = (index < 20) ? tr_inputs : (index < 26 ? vd_inputs : te_inputs);
            const auto& targets = (index < 20) ? tr_targets : (index < 26 ? vd_targets : te_targets);

            CSVFixture::check(inputs(row, 0, 0, 0), index, 0);
            CSVFixture::check(inputs(row, 1, 0, 0), index, 1);
            tensor_size_t cate = 0;
            targets.vector(row).maxCoeff(&cate);
            CSVFixture::check(cate, index, 2);
            CSVFixture::check(inputs(row, 2, 0, 0), index, 3);
        }
    }
}

UTEST_END_MODULE()
