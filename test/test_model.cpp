#include <nano/model.h>
#include <utest/utest.h>
#include "fixture/enum.h"

using namespace nano;

class fixture_model_t final : public model_t
{
public:

    fixture_model_t() = default;

    rmodel_t clone() const override
    {
        return std::make_unique<fixture_model_t>(*this);
    }

    scalar_t fit(const loss_t&, const dataset_t&, const indices_t&, const solver_t&) override
    {
        assert(false);
        return 0.0;
    }

    tensor4d_t predict(const dataset_t&, const indices_t&) const override
    {
        assert(false);
        return tensor4d_t{};
    }
};

static void check_equal(const model_param_t& param, const model_param_t& xparam)
{
    UTEST_CHECK_EQUAL(xparam.name(), param.name());
    UTEST_CHECK_EQUAL(xparam.is_evalue(), param.is_evalue());
    UTEST_CHECK_EQUAL(xparam.is_ivalue(), param.is_ivalue());
    UTEST_CHECK_EQUAL(xparam.is_svalue(), param.is_svalue());
    if (xparam.is_svalue())
    {
        UTEST_CHECK_CLOSE(xparam.svalue(), param.svalue(), 1e-12);
    }
    else if (xparam.is_ivalue())
    {
        UTEST_CHECK_EQUAL(xparam.ivalue(), param.ivalue());
    }
    else
    {
        UTEST_CHECK_EQUAL(xparam.evalue<enum_type>(), param.evalue<enum_type>());
    }
}

static void check_stream(const model_param_t& param)
{
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(param.write(stream));
        str = stream.str();
    }
    {
        model_param_t xparam;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xparam.read(stream));
        check_equal(param, xparam);
    }
}

static auto check_stream(const fixture_model_t& model)
{
    string_t str;
    {
        const auto clone = model.clone();
        UTEST_CHECK_EQUAL(clone->params().size(), model.params().size());
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(clone->write(stream));
        str = stream.str();
    }
    {
        fixture_model_t xmodel;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xmodel.read(stream));
        return xmodel;
    }
}

UTEST_BEGIN_MODULE(test_model)

UTEST_CASE(model_param_empty)
{
    auto param = model_param_t{};

    UTEST_CHECK_EQUAL(param.name(), "");
    UTEST_CHECK_EQUAL(param.is_evalue(), true);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);
}

UTEST_CASE(model_param_eparam)
{
    auto param = model_param_t{eparam1_t{"eparam", enum_type::type1}};

    UTEST_CHECK_EQUAL(param.name(), "eparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), true);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);

    UTEST_CHECK_THROW(param.svalue(), std::runtime_error);
    UTEST_CHECK_THROW(param.ivalue(), std::runtime_error);
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type1);

    UTEST_CHECK_THROW(param.set(int32_t{1}), std::runtime_error);
    UTEST_CHECK_THROW(param.set(int64_t{1}), std::runtime_error);

    UTEST_CHECK_NOTHROW(param.set(enum_type::type2));
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type2);
    UTEST_CHECK_THROW(param.set(static_cast<enum_type>(-1)), std::runtime_error);
    UTEST_CHECK_EQUAL(param.evalue<enum_type>(), enum_type::type2);

    check_stream(param);
}

UTEST_CASE(model_param_iparam)
{
    auto param = model_param_t{iparam1_t{"iparam", 0, LE, 1, LE, 5}};

    UTEST_CHECK_EQUAL(param.name(), "iparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), false);
    UTEST_CHECK_EQUAL(param.is_ivalue(), true);
    UTEST_CHECK_EQUAL(param.is_svalue(), false);

    UTEST_CHECK_THROW(param.svalue(), std::runtime_error);
    UTEST_CHECK_THROW(param.evalue<enum_type>(), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 1);

    UTEST_CHECK_NOTHROW(param.set(int32_t{0}));
    UTEST_CHECK_EQUAL(param.ivalue(), 0);

    UTEST_CHECK_NOTHROW(param.set(int64_t{5}));
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(int64_t{7}), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(int32_t{-1}), std::runtime_error);
    UTEST_CHECK_EQUAL(param.ivalue(), 5);

    UTEST_CHECK_THROW(param.set(scalar_t{0}), std::runtime_error);
    UTEST_CHECK_THROW(param.set(enum_type::type1), std::runtime_error);

    check_stream(param);
}

UTEST_CASE(model_param_sparam)
{
    auto param = model_param_t{sparam1_t{"sparam", 0, LE, 1, LE, 5}};

    UTEST_CHECK_EQUAL(param.name(), "sparam");
    UTEST_CHECK_EQUAL(param.is_evalue(), false);
    UTEST_CHECK_EQUAL(param.is_ivalue(), false);
    UTEST_CHECK_EQUAL(param.is_svalue(), true);

    UTEST_CHECK_CLOSE(param.svalue(), 1.0, 1e-12);
    UTEST_CHECK_THROW(param.evalue<enum_type>(), std::runtime_error);
    UTEST_CHECK_THROW(param.ivalue(), std::runtime_error);

    UTEST_CHECK_NOTHROW(param.set(0.1));
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_THROW(param.set(-1.1), std::runtime_error);
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_THROW(param.set(5.1), std::runtime_error);
    UTEST_CHECK_CLOSE(param.svalue(), 0.1, 1e-12);

    UTEST_CHECK_NOTHROW(param.set(int32_t{0}));
    UTEST_CHECK_CLOSE(param.svalue(), 0.0, 1e-12);

    UTEST_CHECK_NOTHROW(param.set(int64_t{1}));
    UTEST_CHECK_CLOSE(param.svalue(), 1.0, 1e-12);

    UTEST_CHECK_THROW(param.set(enum_type::type1), std::runtime_error);

    check_stream(param);
}

UTEST_CASE(empty)
{
    const auto check_params = [] (const model_t& model)
    {
        UTEST_CHECK(model.params().empty());
    };

    auto model = fixture_model_t{};
    check_params(model);

    UTEST_CHECK_THROW(model.set("nonexistent_param_name", enum_type::type1), std::runtime_error);
    UTEST_CHECK_THROW(model.set("nonexistent_param_name", 10), std::runtime_error);
    UTEST_CHECK_THROW(model.set("nonexistent_param_name", 4.2), std::runtime_error);

    check_params(check_stream(model));
}

UTEST_CASE(parameters)
{
    const auto check_params = [] (const model_t& model)
    {
        UTEST_CHECK_EQUAL(model.params().size(), 6U);

        UTEST_CHECK_EQUAL(model.evalue<enum_type>("eparam1"), enum_type::type3);
        UTEST_CHECK_EQUAL(model.ivalue("iparam1"), 1);
        UTEST_CHECK_EQUAL(model.ivalue("iparam2"), 2);
        UTEST_CHECK_CLOSE(model.svalue("sparam1"), 1.5, 1e-12);
        UTEST_CHECK_CLOSE(model.svalue("sparam2"), 2.5, 1e-12);
        UTEST_CHECK_CLOSE(model.svalue("sparam3"), 3.5, 1e-12);
    };

    auto model = fixture_model_t{};
    model.register_param(eparam1_t{"eparam1", enum_type::type3});
    model.register_param(iparam1_t{"iparam1", 0, LE, 1, LE, 10});
    model.register_param(iparam1_t{"iparam2", 1, LE, 2, LE, 10});
    model.register_param(sparam1_t{"sparam1", 1.0, LT, 1.5, LT, 2.0});
    model.register_param(sparam1_t{"sparam2", 2.0, LT, 2.5, LT, 3.0});
    model.register_param(sparam1_t{"sparam3", 3.0, LT, 3.5, LT, 4.0});

    check_params(model);
    check_params(check_stream(model));

    UTEST_CHECK_THROW(model.set("nonexistent_param_name", enum_type::type1), std::runtime_error);
    UTEST_CHECK_THROW(model.set("nonexistent_param_name", 10), std::runtime_error);
    UTEST_CHECK_THROW(model.set("nonexistent_param_name", 4.2), std::runtime_error);

    UTEST_CHECK_THROW(model.set("eparam1", static_cast<enum_type>(-1)), std::runtime_error);
    UTEST_CHECK_EQUAL(model.evalue<enum_type>("eparam1"), enum_type::type3);

    UTEST_CHECK_NOTHROW(model.set("eparam1", enum_type::type2));
    UTEST_CHECK_EQUAL(model.evalue<enum_type>("eparam1"), enum_type::type2);

    UTEST_CHECK_THROW(model.set("iparam2", 100), std::runtime_error);
    UTEST_CHECK_EQUAL(model.ivalue("iparam2"), 2);

    UTEST_CHECK_NOTHROW(model.set("iparam2", 3));
    UTEST_CHECK_EQUAL(model.ivalue("iparam2"), 3);

    UTEST_CHECK_THROW(model.set("sparam3", 4.1), std::runtime_error);
    UTEST_CHECK_CLOSE(model.svalue("sparam3"), 3.5, 1e-12);

    UTEST_CHECK_NOTHROW(model.set("sparam3", 3.9));
    UTEST_CHECK_CLOSE(model.svalue("sparam3"), 3.9, 1e-12);

    auto config = model_config_t{};
    config.add("iparam1", int32_t{3});
    config.add("iparam2", int64_t{6});
    config.add("sparam3", scalar_t{3.1});
    UTEST_CHECK_NOTHROW(model.set(config));
    UTEST_CHECK_EQUAL(model.ivalue("iparam1"), 3);
    UTEST_CHECK_EQUAL(model.ivalue("iparam2"), 6);
    UTEST_CHECK_CLOSE(model.svalue("sparam3"), 3.1, 1e-12);
}

UTEST_END_MODULE()
