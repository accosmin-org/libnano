#include <fstream>
#include <utest/utest.h>
#include <nano/dataset/tabular.h>

using namespace nano;

static std::ostream& operator<<(std::ostream& os, const feature_t& f)
{
    os << "name=" << f.name() << ",labels[";
    for (const auto& label : f.labels())
    {
        os << label;
        if (&label != &(*(f.labels().rbegin())))
        {
            os << ",";
        }
    }
    return os << "],placeholder=" << f.placeholder();
}

class Fixture final : public tabular_dataset_t
{
public:

    static auto data_path() { return "test_dataset_tabular_data.csv"; }
    static auto test_path() { return "test_dataset_tabular_test.csv"; }

    Fixture()
    {
        std::remove(data_path());
        std::remove(test_path());

        write_data(data_path());
        write_test(test_path());

        paths();
    }

    Fixture(Fixture&&) = default;
    Fixture(const Fixture&) = delete;
    Fixture& operator=(Fixture&&) = default;
    Fixture& operator=(const Fixture&) = delete;

    ~Fixture() override
    {
        std::remove(data_path());
        std::remove(test_path());
    }

    void paths(const tensor_size_t data_expected = 20, const tensor_size_t test_expected = 10)
    {
        csvs(
        {
            csv_t{data_path()}.delim(",").header(false).expected(data_expected).skip('@'),
            csv_t{test_path()}.delim(",").header(true).expected(test_expected).skip('@')
        });
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

    [[nodiscard]] split_t make_split() const override
    {
        UTEST_CHECK_EQUAL(samples(), 30);

        split_t split;
        split.indices(protocol::train) = arange(m_tr_begin, m_tr_end);
        split.indices(protocol::valid) = arange(m_vd_begin, m_vd_end);
        split.indices(protocol::test) = arange(m_te_begin, m_te_end);
        return split;
    }

    static void write_data(const char* path)
    {
        std::ofstream os(path);
        write(os, 1, 20, false);
        UTEST_REQUIRE(os);
    }

    static void write_test(const char* path)
    {
        std::ofstream os(path);
        write(os, 21, 10, true);
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

    static void write(std::ostream& os, const int begin, const int size, const bool header)
    {
        if (header)
        {
            os << "cont,cont_opt,cate,cate_opt\n";
        }

        for (auto index = begin; index < begin + size; ++ index)
        {
            os << index << ",";
            (index % 2 == 0) ? (os << "?,") : (os << (3.0 - 0.2 * index) << ",");
            os << "cate" << (index % 3) << ",";
            (index % 4 == 0) ? (os << "?,") : (os << "cate_opt" << (index % 2) << ",");
            os << "\n";

            if (index % 7 == 0) { os << "\n"; }
            if (index % 9 == 0) { os << "@ this line should be skipped\n"; }
        }
    }

private:

    // attributes
    tensor_size_t   m_tr_begin{0}, m_tr_end{20};    ///<
    tensor_size_t   m_vd_begin{20}, m_vd_end{26};   ///<
    tensor_size_t   m_te_begin{26}, m_te_end{30};   ///<
};

static auto feature_cont()
{
    auto feature = feature_t{"cont"};

    UTEST_CHECK(feature_t::missing(feature_t::placeholder_value()));
    UTEST_CHECK(!feature_t::missing(0));

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(!feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto feature_cont_opt()
{
    auto feature = feature_t{"cont_opt"}.placeholder("?");

    UTEST_CHECK(!feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_THROW(feature.label(0), std::invalid_argument);
    UTEST_CHECK_THROW(feature.label(feature_t::placeholder_value()), std::invalid_argument);

    return feature;
}

static auto feature_cate()
{
    auto feature = feature_t{"cate"}.labels({"cate0", "cate1", "cate2"});

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

static auto feature_cate_opt()
{
    auto feature = feature_t{"cate_opt"}.labels({"cate_opt0", "cate_opt1"}).placeholder("?");

    UTEST_CHECK(feature.discrete());
    UTEST_CHECK(feature.optional());
    UTEST_CHECK_EQUAL(feature.label(0), "cate_opt0");
    UTEST_CHECK_EQUAL(feature.label(1), "cate_opt1");
    UTEST_CHECK_THROW(feature.label(-1), std::out_of_range);
    UTEST_CHECK_THROW(feature.label(+2), std::out_of_range);
    UTEST_CHECK_EQUAL(feature.label(feature_t::placeholder_value()), string_t());

    return feature;
}

UTEST_BEGIN_MODULE(test_tabular)

UTEST_CASE(empty)
{
    auto dataset = Fixture{};

    UTEST_CHECK_EQUAL(dataset.ifeatures(), 0);
    UTEST_CHECK_THROW(dataset.ifeature(0), std::out_of_range);
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);
}

UTEST_CASE(config)
{
    auto dataset = Fixture{};

    UTEST_CHECK_THROW(dataset.folds(0), std::invalid_argument);
    UTEST_CHECK_THROW(dataset.folds(101), std::invalid_argument);
    UTEST_CHECK_THROW(dataset.train_percentage(9), std::invalid_argument);
    UTEST_CHECK_THROW(dataset.train_percentage(91), std::invalid_argument);

    UTEST_CHECK_NOTHROW(dataset.folds(1));
    UTEST_CHECK_EQUAL(dataset.folds(), size_t(1));

    UTEST_CHECK_NOTHROW(dataset.folds(100));
    UTEST_CHECK_EQUAL(dataset.folds(), size_t(100));

    UTEST_CHECK_NOTHROW(dataset.train_percentage(10));
    UTEST_CHECK_EQUAL(dataset.train_percentage(), 10);

    UTEST_CHECK_NOTHROW(dataset.train_percentage(90));
    UTEST_CHECK_EQUAL(dataset.train_percentage(), 90);
}

UTEST_CASE(config_no_target)
{
    auto dataset = Fixture{};

    dataset.folds(7);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});

    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate());
    UTEST_CHECK_EQUAL(dataset.ifeature(3), feature_cate_opt());
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);
}

UTEST_CASE(config_with_target)
{
    auto dataset = Fixture{};

    dataset.folds(7);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0);

    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cate());
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cont());
}

UTEST_CASE(noload_no_data)
{
    auto dataset = Fixture{};

    dataset.csvs({});
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_no_features)
{
    auto dataset = Fixture{};

    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_few_features)
{
    auto dataset = Fixture{};

    dataset.features({feature_cont(), feature_cont_opt(), feature_cate()}, 0);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_wrong_features)
{
    auto dataset = Fixture{};

    dataset.features({feature_cont_opt(), feature_cont(), feature_cate(), feature_cate_opt()}, 1);
    UTEST_CHECK(!dataset.load());

    dataset.features({feature_cont(), feature_cont_opt(), feature_cate_opt(), feature_cate()}, 0);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_wrong_expected)
{
    auto dataset = Fixture{};

    dataset.paths(21, 10);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0);
    UTEST_CHECK(!dataset.load());

    dataset.paths(20, 9);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_invalid_target)
{
    auto dataset = Fixture{};

    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 4);
    UTEST_CHECK(!dataset.load());

    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 1);
    UTEST_CHECK(!dataset.load());

    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 3);
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(noload_invalid_splits)
{
    auto dataset = Fixture{};

    dataset.split(-1, 10, 10, 26, 26, 29);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(10, 31, 20, 26, 26, 29);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, -1, 5, 26, 30);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 30, 36, 26, 30);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, -1, 3);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 27, 31);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 0, 20, 26, 26, 30);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 20, 26, 30);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 26, 26);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 27, 26, 30);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());

    dataset.split(0, 20, 20, 26, 26, 29);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});
    UTEST_CHECK(!dataset.load());
}

UTEST_CASE(load_no_target)
{
    auto dataset = Fixture{};

    dataset.folds(3);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()});

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 4);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate());
    UTEST_CHECK_EQUAL(dataset.ifeature(3), feature_cate_opt());
    UTEST_CHECK_THROW(dataset.tfeature(), std::out_of_range);

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_samples = dataset.samples(fold_t{f, protocol::train});
        const auto vd_samples = dataset.samples(fold_t{f, protocol::valid});
        const auto te_samples = dataset.samples(fold_t{f, protocol::test});

        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_samples, 20);
        UTEST_CHECK_EQUAL(vd_samples, 6);
        UTEST_CHECK_EQUAL(te_samples, 4);

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

            Fixture::check(inputs(row, 0, 0, 0), index, 0);
            Fixture::check(inputs(row, 1, 0, 0), index, 1);
            Fixture::check(inputs(row, 2, 0, 0), index, 2);
            Fixture::check(inputs(row, 3, 0, 0), index, 3);
        }

        for (auto begin = 0; begin < 20; begin += 10)
        {
            const auto inputs = dataset.inputs(fold_t{f, protocol::train}, make_range(begin, begin + 10));

            for (auto index = 0; index < 10; ++ index)
            {
                Fixture::check(inputs(index, 0, 0, 0), begin + index, 0);
                Fixture::check(inputs(index, 1, 0, 0), begin + index, 1);
                Fixture::check(inputs(index, 2, 0, 0), begin + index, 2);
                Fixture::check(inputs(index, 3, 0, 0), begin + index, 3);
            }
        }

        UTEST_CHECK_NOTHROW(dataset.shuffle(fold_t{f, protocol::train}));
        UTEST_CHECK_NOTHROW(dataset.shuffle(fold_t{f, protocol::valid}));
        UTEST_CHECK_NOTHROW(dataset.shuffle(fold_t{f, protocol::test}));
    }
}

UTEST_CASE(load_with_cont_target)
{
    auto dataset = Fixture{};

    dataset.folds(2);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 0);

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 2);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cate());
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cont());

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_samples = dataset.samples(fold_t{f, protocol::train});
        const auto vd_samples = dataset.samples(fold_t{f, protocol::valid});
        const auto te_samples = dataset.samples(fold_t{f, protocol::test});

        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_samples, 20);
        UTEST_CHECK_EQUAL(vd_samples, 6);
        UTEST_CHECK_EQUAL(te_samples, 4);

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

            Fixture::check(targets(row, 0, 0, 0), index, 0);
            Fixture::check(inputs(row, 0, 0, 0), index, 1);
            Fixture::check(inputs(row, 1, 0, 0), index, 2);
            Fixture::check(inputs(row, 2, 0, 0), index, 3);
        }
    }
}

UTEST_CASE(load_with_cate_target)
{
    auto dataset = Fixture{};

    dataset.folds(7);
    dataset.features({feature_cont(), feature_cont_opt(), feature_cate(), feature_cate_opt()}, 2);

    UTEST_REQUIRE(dataset.load());
    UTEST_CHECK_EQUAL(dataset.folds(), 7);
    UTEST_CHECK_EQUAL(dataset.ifeatures(), 3);
    UTEST_CHECK_EQUAL(dataset.ifeature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.ifeature(1), feature_cont_opt());
    UTEST_CHECK_EQUAL(dataset.ifeature(2), feature_cate_opt());
    UTEST_CHECK_EQUAL(dataset.tfeature(), feature_cate());

    for (size_t f = 0, folds = dataset.folds(); f < folds; ++ f)
    {
        const auto tr_samples = dataset.samples(fold_t{f, protocol::train});
        const auto vd_samples = dataset.samples(fold_t{f, protocol::valid});
        const auto te_samples = dataset.samples(fold_t{f, protocol::test});

        const auto tr_inputs = dataset.inputs(fold_t{f, protocol::train});
        const auto vd_inputs = dataset.inputs(fold_t{f, protocol::valid});
        const auto te_inputs = dataset.inputs(fold_t{f, protocol::test});

        const auto tr_targets = dataset.targets(fold_t{f, protocol::train});
        const auto vd_targets = dataset.targets(fold_t{f, protocol::valid});
        const auto te_targets = dataset.targets(fold_t{f, protocol::test});

        UTEST_CHECK_EQUAL(tr_samples, 20);
        UTEST_CHECK_EQUAL(vd_samples, 6);
        UTEST_CHECK_EQUAL(te_samples, 4);

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

            Fixture::check(inputs(row, 0, 0, 0), index, 0);
            Fixture::check(inputs(row, 1, 0, 0), index, 1);
            tensor_size_t cate = 0;
            targets.vector(row).maxCoeff(&cate);
            Fixture::check(cate, index, 2);
            Fixture::check(inputs(row, 2, 0, 0), index, 3);
        }
    }
}

UTEST_END_MODULE()
