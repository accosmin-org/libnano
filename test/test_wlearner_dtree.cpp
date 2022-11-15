#include "fixture/wlearner.h"
#include <nano/wlearner/dtree.h>

using namespace nano;

static auto make_wdtree(const int min_split, const int max_depth)
{
    auto wlearner                                    = dtree_wlearner_t{};
    wlearner.parameter("wlearner::dtree::min_split") = min_split;
    wlearner.parameter("wlearner::dtree::max_depth") = max_depth;
    return wlearner;
}

class wdtree_datasource_t : public wlearner_datasource_t
{
public:
    wdtree_datasource_t(const tensor_size_t samples, const tensor_size_t groups)
        : wlearner_datasource_t(samples, groups)
    {
    }

    virtual dtree_wlearner_t make_wlearner() const     = 0;
    virtual dtree_nodes_t    expected_nodes() const    = 0;
    virtual tensor4d_t       expected_tables() const   = 0;
    virtual indices_t        expected_features() const = 0;

    void check_wlearner(const dtree_wlearner_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.nodes(), expected_nodes());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-8);
    }

protected:
    template <typename tfvalues>
    void set_stump_target(const tensor_size_t sample, const tensor_size_t feature, const tfvalues& fvalues,
                          const scalar_t threshold, const scalar_t pred_lower, const scalar_t pred_upper,
                          const tensor_size_t cluster_offset)
    {
        const auto itarget                   = this->features(); // NB: the last feature is the target!
        const auto [fvalue, target, cluster] = make_stump_target(fvalues(sample), threshold, pred_lower, pred_upper);

        set(sample, feature, fvalue);
        set(sample, itarget, target);
        assign(sample, cluster + cluster_offset);
    }

    template <typename tfvalues>
    void set_table_target(const tensor_size_t sample, const tensor_size_t feature, const tfvalues& fvalues,
                          const tensor4d_t& tables, const tensor_size_t cluster_offset)
    {
        const auto itarget                   = this->features(); // NB: the last feature is the target!
        const auto [fvalue, target, cluster] = make_table_target(fvalues(sample), tables);

        set(sample, feature, fvalue);
        set(sample, itarget, target);
        assign(sample, cluster + cluster_offset);
    }
};

class wdtree_stump1_datasource_t final : public wdtree_datasource_t
{
public:
    explicit wdtree_stump1_datasource_t(const tensor_size_t samples)
        : wdtree_datasource_t(samples, 2)
    {
    }

    static auto expected_feature() { return 6; }

    static auto expected_threshold() { return 1.5; }

    static auto expected_pred_lower() { return -4.0; }

    static auto expected_pred_upper() { return +3.7; }

    rdatasource_t clone() const override { return std::make_unique<wdtree_stump1_datasource_t>(*this); }

    dtree_wlearner_t make_wlearner() const override { return make_wdtree(1, 1); }

    indices_t expected_features() const override { return make_indices(expected_feature()); }

    tensor4d_t expected_tables() const override
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_pred_lower(), expected_pred_upper());
    }

    dtree_nodes_t expected_nodes() const override
    {
        return {
            dtree_node_t{expected_feature(), -1, expected_threshold(), 0U, +0},
            dtree_node_t{expected_feature(), -1, expected_threshold(), 0U, +1}
        };
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), -5, +4);

        set_targets(feature,
                    [&](const tensor_size_t sample) {
                        return make_stump_target(fvalues(sample), expected_threshold(), expected_pred_lower(),
                                                 expected_pred_upper());
                    });
    }
};

class wdtree_table1_datasource_t final : public wdtree_datasource_t
{
public:
    explicit wdtree_table1_datasource_t(const tensor_size_t samples)
        : wdtree_datasource_t(samples, 3)
    {
    }

    static auto expected_feature() { return 1; }

    rdatasource_t clone() const override { return std::make_unique<wdtree_table1_datasource_t>(*this); }

    dtree_wlearner_t make_wlearner() const override { return make_wdtree(1, 1); }

    indices_t expected_features() const override { return make_indices(expected_feature()); }

    tensor4d_t expected_tables() const override
    {
        return make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -4.2, +0.7, -1.3);
    }

    dtree_nodes_t expected_nodes() const override
    {
        return {
            dtree_node_t{expected_feature(), +3, 0, 0U, +0},
            dtree_node_t{expected_feature(), +3, 0, 0U, +1},
            dtree_node_t{expected_feature(), +3, 0, 0U, +2}
        };
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto tables  = expected_tables();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes - 1);

        assert(classes == tables.size<0>());

        set_targets(feature, [&](const tensor_size_t sample) { return make_table_target(fvalues(sample), tables); });
    }
};

class wdtree_depth2_datasource_t final : public wdtree_datasource_t
{
public:
    explicit wdtree_depth2_datasource_t(const tensor_size_t samples)
        : wdtree_datasource_t(samples, 6)
    {
    }

    static auto expected_feature0() { return 1; }

    static auto expected_feature10() { return 5; }

    static auto expected_feature11() { return 6; }

    static auto expected_feature12() { return 5; }

    static auto expected_threshold10() { return -1.5; }

    static auto expected_threshold11() { return +2.5; }

    static auto expected_threshold12() { return +1.5; }

    static auto expected_pred_lower10() { return -0.1; }

    static auto expected_pred_upper10() { return +0.2; }

    static auto expected_pred_lower11() { return +3.2; }

    static auto expected_pred_upper11() { return +3.3; }

    static auto expected_pred_lower12() { return +6.1; }

    static auto expected_pred_upper12() { return +6.4; }

    rdatasource_t clone() const override { return std::make_unique<wdtree_depth2_datasource_t>(*this); }

    dtree_wlearner_t make_wlearner() const override { return make_wdtree(1, 2); }

    indices_t expected_features() const override
    {
        return make_indices(expected_feature0(), expected_feature10(), expected_feature11());
    }

    tensor4d_t expected_tables() const override
    {
        return make_tensor<scalar_t>(make_dims(6, 1, 1, 1), expected_pred_lower10(), expected_pred_upper10(),
                                     expected_pred_lower11(), expected_pred_upper11(), expected_pred_lower12(),
                                     expected_pred_upper12());
    }

    dtree_nodes_t expected_nodes() const override
    {
        return {
            dtree_node_t{ expected_feature0(), +3,                    0.0, 3U, -1},
            dtree_node_t{ expected_feature0(), +3,                    0.0, 5U, -1},
            dtree_node_t{ expected_feature0(), +3,                    0.0, 7U, -1},
            dtree_node_t{expected_feature10(), -1, expected_threshold10(), 0U, +0},
            dtree_node_t{expected_feature10(), -1, expected_threshold10(), 0U, +1},
            dtree_node_t{expected_feature11(), -1, expected_threshold11(), 0U, +2},
            dtree_node_t{expected_feature11(), -1, expected_threshold11(), 0U, +3},
            dtree_node_t{expected_feature12(), -1, expected_threshold12(), 0U, +4},
            dtree_node_t{expected_feature12(), -1, expected_threshold12(), 0U, +5}
        };
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature0  = expected_feature0();
        const auto feature10 = expected_feature10();
        const auto feature11 = expected_feature11();
        const auto feature12 = expected_feature12();

        const auto classes0 = this->feature(feature0).classes();
        const auto fvalues0 = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes0 - 1);

        const auto fvalues10 = make_random_tensor<int32_t>(make_dims(this->samples()), -5, +7);
        const auto fvalues11 = make_random_tensor<int32_t>(make_dims(this->samples()), -7, +5);
        const auto fvalues12 = make_random_tensor<int32_t>(make_dims(this->samples()), -3, +9);

        const auto hits    = this->hits();
        const auto samples = this->samples();

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature0) != 0)
            {
                const auto fvalue0 = fvalues0(sample);
                set(sample, feature0, fvalue0);

                switch (fvalue0)
                {
                case 0:
                    set_stump_target(sample, feature10, fvalues10, expected_threshold10(), expected_pred_lower10(),
                                     expected_pred_upper10(), 0);
                    break;

                case 1:
                    set_stump_target(sample, feature11, fvalues11, expected_threshold11(), expected_pred_lower11(),
                                     expected_pred_upper11(), 2);
                    break;

                default:
                    set_stump_target(sample, feature12, fvalues12, expected_threshold12(), expected_pred_lower12(),
                                     expected_pred_upper12(), 4);
                    break;
                }
            }
        }
    }
};

class wdtree_depth3_datasource_t final : public wdtree_datasource_t
{
public:
    explicit wdtree_depth3_datasource_t(const tensor_size_t samples)
        : wdtree_datasource_t(samples, 9)
    {
    }

    static auto expected_feature0() { return 5; }

    static auto expected_feature10() { return 6; }

    static auto expected_feature11() { return 5; }

    static auto expected_feature20() { return 0; }

    static auto expected_feature21() { return 7; }

    static auto expected_feature22() { return 1; }

    static auto expected_feature23() { return 2; }

    static auto expected_threshold0() { return +1.5; }

    static auto expected_threshold10() { return -1.5; }

    static auto expected_threshold11() { return +3.5; }

    static auto expected_threshold21() { return -2.5; }

    static auto expected_pred_lower0() { return -3.0; }

    static auto expected_pred_upper0() { return +4.0; }

    static auto expected_pred_lower10() { return -3.1; }

    static auto expected_pred_upper10() { return -2.9; }

    static auto expected_pred_lower11() { return +3.9; }

    static auto expected_pred_upper11() { return +4.1; }

    static auto expected_pred_lower21() { return -3.2; }

    static auto expected_pred_upper21() { return -3.0; }

    static auto expected_table20_0() { return -3.0; }

    static auto expected_table20_1() { return -2.8; }

    static auto expected_table22_0() { return +3.8; }

    static auto expected_table22_1() { return +3.9; }

    static auto expected_table22_2() { return +4.0; }

    static auto expected_table23_0() { return +4.2; }

    static auto expected_table23_1() { return +4.3; }

    rdatasource_t clone() const override { return std::make_unique<wdtree_depth3_datasource_t>(*this); }

    dtree_wlearner_t make_wlearner() const override { return make_wdtree(1, 3); }

    indices_t expected_features() const override
    {
        return make_indices(expected_feature20(), expected_feature22(), expected_feature23(), expected_feature0(),
                            expected_feature10(), expected_feature21());
    }

    tensor4d_t expected_tables() const override
    {
        return make_tensor<scalar_t>(make_dims(9, 1, 1, 1), expected_table20_0(), expected_table20_1(),
                                     expected_pred_lower21(), expected_pred_upper21(), expected_table22_0(),
                                     expected_table22_1(), expected_table22_2(), expected_table23_0(),
                                     expected_table23_1());
    }

    dtree_nodes_t expected_nodes() const override
    {
        return {
            dtree_node_t{ expected_feature0(), -1,  expected_threshold0(),  2U, -1},
            dtree_node_t{ expected_feature0(), -1,  expected_threshold0(),  4U, -1},
            dtree_node_t{expected_feature10(), -1, expected_threshold10(),  6U, -1},
            dtree_node_t{expected_feature10(), -1, expected_threshold10(),  8U, -1},
            dtree_node_t{expected_feature11(), -1, expected_threshold11(), 10U, -1},
            dtree_node_t{expected_feature11(), -1, expected_threshold11(), 13U, -1},
            dtree_node_t{expected_feature20(), +2,                    0.0,  0U, +0},
            dtree_node_t{expected_feature20(), +2,                    0.0,  0U, +1},
            dtree_node_t{expected_feature21(), -1, expected_threshold21(),  0U, +2},
            dtree_node_t{expected_feature21(), -1, expected_threshold21(),  0U, +3},
            dtree_node_t{expected_feature22(), +3,                    0.0,  0U, +4},
            dtree_node_t{expected_feature22(), +3,                    0.0,  0U, +5},
            dtree_node_t{expected_feature22(), +3,                    0.0,  0U, +6},
            dtree_node_t{expected_feature23(), +2,                    0.0,  0U, +7},
            dtree_node_t{expected_feature23(), +2,                    0.0,  0U, +8},
        };
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature0  = expected_feature0();
        const auto feature10 = expected_feature10();
        const auto feature11 = expected_feature11();
        const auto feature20 = expected_feature20();
        const auto feature21 = expected_feature21();
        const auto feature22 = expected_feature22();
        const auto feature23 = expected_feature23();

        const auto classes20 = this->feature(feature20).classes();
        const auto classes22 = this->feature(feature22).classes();
        const auto classes23 = this->feature(feature23).classes();

        const auto fvalues20 = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes20 - 1);
        const auto fvalues22 = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes22 - 1);
        const auto fvalues23 = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes23 - 1);

        const auto  fvalues0  = make_random_tensor<int32_t>(make_dims(this->samples()), -5, +7);
        const auto  fvalues10 = make_random_tensor<int32_t>(make_dims(this->samples()), -7, +9);
        const auto& fvalues11 = fvalues0;
        const auto  fvalues21 = make_random_tensor<int32_t>(make_dims(this->samples()), -8, +8);

        const auto tables20 = make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_table20_0(), expected_table20_1());
        const auto tables22 = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), expected_table22_0(), expected_table22_1(),
                                                    expected_table22_2());
        const auto tables23 = make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_table23_0(), expected_table23_1());

        const auto hits    = this->hits();
        const auto samples = this->samples();

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature0) != 0)
            {
                const auto fvalue0  = fvalues0(sample);
                const auto fvalue10 = fvalues10(sample);
                const auto fvalue11 = fvalues11(sample);
                const auto fvalue21 = fvalues21(sample);

                set(sample, feature0, fvalue0);
                set(sample, feature10, fvalue10);
                set(sample, feature11, fvalue11);
                set(sample, feature21, fvalue21);

                if (static_cast<scalar_t>(fvalue0) < expected_threshold0())
                {
                    if (static_cast<scalar_t>(fvalue10) < expected_threshold10())
                    {
                        set_table_target(sample, feature20, fvalues20, tables20, 0);
                    }
                    else
                    {
                        set_stump_target(sample, feature21, fvalues21, expected_threshold21(), expected_pred_lower21(),
                                         expected_pred_upper21(), 2);
                    }
                }
                else
                {
                    if (static_cast<scalar_t>(fvalue11) < expected_threshold11())
                    {
                        set_table_target(sample, feature22, fvalues22, tables22, 4);
                    }
                    else
                    {
                        set_table_target(sample, feature23, fvalues23, tables23, 7);
                    }
                }
            }
        }
    }
};

UTEST_BEGIN_MODULE(test_wlearner_dtree)

UTEST_CASE(print)
{
    const auto nodes = dtree_nodes_t{
        dtree_node_t{+5, +3, 0.0, 0U, +2},
        dtree_node_t{+0, -1, 3.5, 0U, -1},
    };

    {
        std::stringstream stream;
        stream << nodes[0];
        UTEST_CHECK_EQUAL(stream.str(),
                          scat("node: feature=5,classes=3,threshold=", nodes[0].m_threshold, ",next=0,table=2"));
    }
    {
        std::stringstream stream;
        stream << nodes;
        UTEST_CHECK_EQUAL(stream.str(),
                          scat("nodes:{\n", "\tnode: feature=5,classes=3,threshold=", nodes[0].m_threshold,
                               ",next=0,table=2\n", "\tnode: feature=0,classes=-1,threshold=", nodes[1].m_threshold,
                               ",next=0,table=-1\n"
                               "}"));
    }
}

UTEST_CASE(fit_predict_stump1)
{
    const auto datasource0 = make_datasource<wdtree_stump1_datasource_t>(200);

    check_wlearner(datasource0);
}

UTEST_CASE(fit_predict_table1)
{
    const auto datasource0 = make_datasource<wdtree_table1_datasource_t>(200);

    check_wlearner(datasource0);
}

UTEST_CASE(fit_predict_depth2)
{
    const auto datasource0 = make_datasource<wdtree_depth2_datasource_t>(600);

    check_wlearner(datasource0);
}

UTEST_CASE(fit_predict_depth3)
{
    const auto datasource0 = make_datasource<wdtree_depth3_datasource_t>(1800);

    check_wlearner(datasource0);
}

UTEST_END_MODULE()
