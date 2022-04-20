#include <utest/utest.h>
#include <nano/model/kfold.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_model_kfold)

UTEST_CASE(kfold)
{
    for (const auto seed : {seed_t{}, seed_t{42}})
    {
        const auto folds = 5;
        const auto samples = 21;
        const auto kfold = kfold_t{nano::arange(0, samples), folds, seed};

        for (tensor_size_t fold = 0; fold < folds; ++ fold)
        {
            const auto [train, valid] = kfold.split(fold);

            UTEST_CHECK_EQUAL(train.size() + valid.size(), samples);
            UTEST_CHECK_EQUAL(valid.size(), (fold + 1 == folds) ? 5 : 4);

            UTEST_CHECK_LESS(train.max(), samples);
            UTEST_CHECK_LESS(valid.max(), samples);

            UTEST_CHECK_GREATER_EQUAL(train.min(), 0);
            UTEST_CHECK_GREATER_EQUAL(valid.min(), 0);

            UTEST_CHECK(std::is_sorted(begin(train), end(train)));
            UTEST_CHECK(std::is_sorted(begin(valid), end(valid)));

            for (tensor_size_t sample = 0; sample < samples; ++ sample)
            {
                const auto *const it = std::find(begin(train), end(train), sample);
                const auto *const iv = std::find(begin(valid), end(valid), sample);

                UTEST_CHECK(
                    (it == end(train) && iv != end(valid)) ||
                    (it != end(train) && iv == end(valid)));
            }
        }
    }
}

UTEST_CASE(kfold_repeat)
{
    for (const auto seed : {seed_t{}, seed_t{42}})
    {
        const auto folds = 5;
        const auto samples = 21;
        const auto kfold = kfold_t{nano::arange(0, samples), folds, seed};

        std::vector<indices_t> trains, valids;
        for (tensor_size_t fold = 0; fold < folds; ++ fold)
        {
            const auto [train, valid] = kfold.split(fold);
            trains.push_back(train);
            valids.push_back(valid);
        }

        for (tensor_size_t fold = 0; fold < folds; ++ fold)
        {
            const auto [train, valid] = kfold.split(fold);
            UTEST_CHECK_EQUAL(train, trains[static_cast<size_t>(fold)]);
            UTEST_CHECK_EQUAL(valid, valids[static_cast<size_t>(fold)]);
        }
    }
}

UTEST_CASE(kfold_seed42)
{
    const auto folds = 5;
    const auto samples = 21;

    const auto kfoldr1 = kfold_t{nano::arange(0, samples), folds, seed_t{10}};
    const auto kfoldr2 = kfold_t{nano::arange(0, samples), folds, seed_t{11}};
    const auto kfoldf1 = kfold_t{nano::arange(0, samples), folds, seed_t{42}};
    const auto kfoldf2 = kfold_t{nano::arange(0, samples), folds, seed_t{42}};

    for (tensor_size_t fold = 0; fold < folds; ++ fold)
    {
        const auto [trainf1, validf1] = kfoldf1.split(fold);
        const auto [trainf2, validf2] = kfoldf2.split(fold);

        UTEST_CHECK_EQUAL(trainf1, trainf2);
        UTEST_CHECK_EQUAL(validf1, validf2);

        const auto [trainr1, validr1] = kfoldr1.split(fold);
        const auto [trainr2, validr2] = kfoldr2.split(fold);

        UTEST_CHECK_NOT_EQUAL(trainr1, trainr2);
        UTEST_CHECK_NOT_EQUAL(trainr1, trainf2);
        UTEST_CHECK_NOT_EQUAL(validr1, validr2);
        UTEST_CHECK_NOT_EQUAL(validr1, validf2);
    }
}

UTEST_END_MODULE()
