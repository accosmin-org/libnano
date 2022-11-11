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

    static auto expected_pred_lower10() { return -1.1; }

    static auto expected_pred_upper10() { return -2.7; }

    static auto expected_pred_lower11() { return +0.1; }

    static auto expected_pred_upper11() { return +0.7; }

    static auto expected_pred_lower12() { return +3.2; }

    static auto expected_pred_upper12() { return +3.7; }

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
        const auto itarget = this->features(); // NB: the last feature is the target!

        const auto stump_split = [&](const auto sample, const auto feature, const auto& fvalues, const auto threshold,
                                     const auto pred_lower, const auto pred_upper, const auto cluster_offset)
        {
            if (hits(sample, feature) != 0)
            {
                const auto [fvalue, target, cluster] =
                    make_stump_target(fvalues(sample), threshold, pred_lower, pred_upper);
                set(sample, feature, fvalue);
                set(sample, itarget, target);
                assign(sample, cluster + cluster_offset);
            }
        };

        for (tensor_size_t sample = 0; sample < samples; ++sample)
        {
            if (hits(sample, feature0) != 0)
            {
                const auto fvalue0 = fvalues0(sample);
                set(sample, feature0, fvalue0);

                switch (fvalue0)
                {
                case 0:
                    stump_split(sample, feature10, fvalues10, expected_threshold10(), expected_pred_lower10(),
                                expected_pred_upper10(), 0);
                    break;

                case 1:
                    stump_split(sample, feature11, fvalues11, expected_threshold11(), expected_pred_lower11(),
                                expected_pred_upper11(), 2);
                    break;

                default:
                    stump_split(sample, feature12, fvalues12, expected_threshold12(), expected_pred_lower12(),
                                expected_pred_upper12(), 4);
                    break;
                }
            }
        }
    }
};

/*class wdtree_depth3_datasource_t : public wdtree_datasource_t
{
public:
    wdtree_depth3_datasource_t() = default;

    int min_split() const override { return 1; }

    int max_depth() const override { return 3; }

    tensor_size_t groups() const override { return 11; }

    tensor_size_t the_discrete_feature() const { return gt_feature22(); }

    tensor_size_t gt_feature0(bool discrete = false) const { return get_feature(discrete); }

    tensor_size_t gt_feature10(bool discrete = false) const { return get_feature(gt_feature0(), discrete); }

    tensor_size_t gt_feature11(bool discrete = false) const { return get_feature(gt_feature10(), discrete); }

    tensor_size_t gt_feature20(bool discrete = true) const { return get_feature(discrete); }

    tensor_size_t gt_feature21(bool discrete = false) const { return get_feature(gt_feature11(), discrete); }

    tensor_size_t gt_feature22(bool discrete = true) const { return get_feature(gt_feature20(), discrete); }

    tensor_size_t gt_feature23(bool discrete = true) const { return get_feature(gt_feature22(), discrete); }

    void make_target(const tensor_size_t sample) override
    {
        auto input  = this->input(sample);
        auto target = this->target(sample);

        const auto tf0  = gt_feature0();
        const auto tf10 = gt_feature10();
        const auto tf11 = gt_feature11();

        if (!feature_t::missing(input(tf0)))
        {
            if ((input(tf0) = static_cast<scalar_t>(sample % 7)) < 3.0)
            {
                if (!feature_t::missing(input(tf10)))
                {
                    if ((input(tf10) = static_cast<scalar_t>(sample % 9)) < 5.0)
                    {
                        target.full(make_table_target(sample, gt_feature20(), 3, 2.0, 0));
                    }
                    else
                    {
                        target.full(make_stump_target(sample, gt_feature21(), 5, 3.5, +1.9, -0.7, 3));
                    }
                }
            }
            else
            {
                if (!feature_t::missing(input(tf11)))
                {
                    if ((input(tf11) = static_cast<scalar_t>(sample % 11)) < 7.0)
                    {
                        target.full(make_table_target(sample, gt_feature22(), 3, 3.0, 5));
                        target.array() -= 20.0;
                    }
                    else
                    {
                        target.full(make_table_target(sample, gt_feature23(), 3, 3.0, 8));
                        target.array() -= 30.0;
                    }
                }
            }
        }
    }

    indices_t features() const override
    {
        // NB: features = {3, 4, 5, 6, 7, 8, 9} aka {stump21, table23, stump11, table22, stump10, table20, stump0}
        return make_tensor<tensor_size_t>(make_dims(7), gt_feature21(), gt_feature23(), gt_feature11(), gt_feature22(),
                                          gt_feature10(), gt_feature20(), gt_feature0());
    }

    tensor4d_t tables() const override
    {
        return make_tensor<scalar_t>(make_dims(11, 1, 1, 1), -2.0, +0.0, +2.0, +1.9, -0.7, -23.0, -20.0, -17.0, -33.0,
                                     -30.0, -27.0);
    }

    dtree_nodes_t nodes() const override
    {
        // NB: features = {3, 4, 5, 6, 7, 8, 9} aka {stump21, table23, stump11, table22, stump10, table20, stump0}
        return {
  // stump0
            dtree_node_t{+6, -1, 2.5,  2U,  -1},
            dtree_node_t{+6, -1, 2.5,  4U,  -1},

 // stump10
            dtree_node_t{+4, -1, 4.5,  6U,  -1},
            dtree_node_t{+4, -1, 4.5,  9U,  -1},

 // stump11
            dtree_node_t{+2, -1, 6.5, 11U,  -1},
            dtree_node_t{+2, -1, 6.5, 14U,  -1},

 // table20
            dtree_node_t{+5, +3, 0.0,  0U,  +0},
            dtree_node_t{+5, +3, 0.0,  0U,  +1},
            dtree_node_t{+5, +3, 0.0,  0U,  +2},

 // stump21
            dtree_node_t{+0, -1, 3.5,  0U,  +3},
            dtree_node_t{+0, -1, 3.5,  0U,  +4},

 // table22
            dtree_node_t{+3, +3, 0.0,  0U,  +5},
            dtree_node_t{+3, +3, 0.0,  0U,  +6},
            dtree_node_t{+3, +3, 0.0,  0U,  +7},

 // table23
            dtree_node_t{+1, +3, 0.0,  0U,  +8},
            dtree_node_t{+1, +3, 0.0,  0U,  +9},
            dtree_node_t{+1, +3, 0.0,  0U, +10}
        };
    }
};*/

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
    const auto datasource0 = make_datasource<wdtree_stump1_datasource_t>(300);

    check_wlearner(datasource0);
}

UTEST_CASE(fit_predict_table1)
{
    const auto datasource0 = make_datasource<wdtree_table1_datasource_t>(300);

    check_wlearner(datasource0);
}

UTEST_CASE(fit_predict_depth2)
{
    const auto datasource0 = make_datasource<wdtree_depth2_datasource_t>(400);

    check_wlearner(datasource0);
}

/*UTEST_CASE(fit_predict_depth3)
{
    const auto datasource   = make_datasource<wdtree_depth3_datasource_t>(10, 1, 1600);
    const auto datasourcex1 = make_datasource<wdtree_depth3_datasource_t>(datasource.isize(), datasource.tsize() + 1);
    const auto datasourcex2 = make_datasource<wdtree_depth3_datasource_t>(datasource.features().max(),
datasource.tsize()); const auto datasourcex3 =
make_datasource<no_discrete_features_datasource_t<wdtree_depth3_datasource_t>>(); const auto datasourcex4 =
make_datasource<no_continuous_features_datasource_t<wdtree_depth3_datasource_t>>(); const auto datasourcex5 =
make_datasource<different_discrete_feature_datasource_t<wdtree_depth3_datasource_t>>();

    auto wlearner = make_wdtree(datasource);
    check_wlearner(wlearner, datasource, datasourcex1, datasourcex2, datasourcex3, datasourcex4, datasourcex5);
}*/

UTEST_END_MODULE()
