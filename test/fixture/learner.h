#include <nano/model.h>
#include <utest/utest.h>

using namespace nano;

template <typename tmodel, typename... tfit_args>
static auto check_fit(const dataset_t& dataset, const tfit_args&... fit_args)
{
    auto model = tmodel{};
    UTEST_CHECK_NOTHROW(model.fit(dataset, fit_args...));
    return model;
}

static void check_predict(const model_t& model, const dataset_t& dataset, const indices_t& samples,
                          const tensor4d_t& expected_predictions)
{
    UTEST_CHECK_EQUAL(model.predict(dataset, samples), expected_predictions);
}

static void check_predict_fails(const model_t& model, const dataset_t& dataset, const indices_t& samples)
{
    UTEST_CHECK_THROW(model.predict(dataset, samples), std::runtime_error);
}
