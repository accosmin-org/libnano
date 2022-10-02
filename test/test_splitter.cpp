#include <nano/splitter.h>
#include <utest/utest.h>

using namespace nano;

static auto make_kfold_splits(const tensor_size_t samples, const tensor_size_t folds, const uint64_t seed)
{
    auto splitter = splitter_t::all().get("k-fold");
    UTEST_REQUIRE(splitter);

    splitter->parameter("splitter::seed")  = seed;
    splitter->parameter("splitter::folds") = folds;

    return splitter->split(nano::arange(0, samples));
}

UTEST_BEGIN_MODULE(test_splitter)

UTEST_CASE(kfold)
{
    const auto folds   = 5;
    const auto samples = 21;

    for (const auto seed : {42U, 11U, 122U})
    {
        tensor_size_t fold = 0;
        for (const auto& [train, valid] : make_kfold_splits(samples, folds, seed))
        {
            UTEST_CHECK_EQUAL(train.size() + valid.size(), samples);
            UTEST_CHECK_EQUAL(valid.size(), (fold + 1 == folds) ? 5 : 4);

            UTEST_CHECK_LESS(train.max(), samples);
            UTEST_CHECK_LESS(valid.max(), samples);

            UTEST_CHECK_GREATER_EQUAL(train.min(), 0);
            UTEST_CHECK_GREATER_EQUAL(valid.min(), 0);

            UTEST_CHECK(std::is_sorted(begin(train), end(train)));
            UTEST_CHECK(std::is_sorted(begin(valid), end(valid)));

            for (tensor_size_t sample = 0; sample < samples; ++sample)
            {
                const auto* const it = std::find(begin(train), end(train), sample);
                const auto* const iv = std::find(begin(valid), end(valid), sample);

                UTEST_CHECK((it == end(train) && iv != end(valid)) || (it != end(train) && iv == end(valid)));
            }

            ++fold;
        }

        UTEST_CHECK_EQUAL(fold, folds);
    }
}

UTEST_CASE(kfold_repeat)
{
    const auto folds   = 5;
    const auto samples = 21;

    const auto splits0 = make_kfold_splits(samples, folds, 42U);
    const auto splits1 = make_kfold_splits(samples, folds, 42U);

    UTEST_REQUIRE_EQUAL(splits0.size(), folds);
    UTEST_REQUIRE_EQUAL(splits1.size(), folds);

    for (size_t fold = 0U; fold < splits0.size(); ++fold)
    {
        UTEST_CHECK_EQUAL(splits0[fold].first, splits1[fold].first);
        UTEST_CHECK_EQUAL(splits0[fold].second, splits1[fold].second);
    }
}

UTEST_CASE(kfold_seed42)
{
    const auto folds   = 5;
    const auto samples = 21;

    const auto splits10  = make_kfold_splits(samples, folds, 10U);
    const auto splits11  = make_kfold_splits(samples, folds, 11U);
    const auto splits42a = make_kfold_splits(samples, folds, 42U);
    const auto splits42b = make_kfold_splits(samples, folds, 42U);

    UTEST_REQUIRE_EQUAL(splits10.size(), folds);
    UTEST_REQUIRE_EQUAL(splits11.size(), folds);
    UTEST_REQUIRE_EQUAL(splits42a.size(), folds);
    UTEST_REQUIRE_EQUAL(splits42b.size(), folds);

    for (size_t fold = 0U; fold < splits10.size(); ++fold)
    {
        UTEST_CHECK_EQUAL(splits42a[fold].first, splits42b[fold].first);
        UTEST_CHECK_EQUAL(splits42a[fold].second, splits42b[fold].second);

        UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits11[fold].first);
        UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits42a[fold].first);
        UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits42b[fold].first);
        UTEST_CHECK_NOT_EQUAL(splits11[fold].first, splits42a[fold].first);
        UTEST_CHECK_NOT_EQUAL(splits11[fold].first, splits42b[fold].first);

        UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits11[fold].second);
        UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits42a[fold].second);
        UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits42b[fold].second);
        UTEST_CHECK_NOT_EQUAL(splits11[fold].second, splits42a[fold].second);
        UTEST_CHECK_NOT_EQUAL(splits11[fold].second, splits42b[fold].second);
    }
}

UTEST_END_MODULE()
