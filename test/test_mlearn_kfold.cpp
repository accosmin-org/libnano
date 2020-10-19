#include <utest/utest.h>
#include <nano/mlearn/kfold.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_mlearn_kfold)

UTEST_CASE(kfold)
{
    const auto folds = 5;
    const auto samples = 21;
    const auto kfold = kfold_t{nano::arange(0, samples), folds};

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

UTEST_END_MODULE()
