#include <nano/dataset/hash.h>
#include <utest/utest.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_dataset_hash)

UTEST_CASE(hash_sclass)
{
    const auto fvalues = make_tensor<int32_t, 1>(make_dims(4), 0, 1, 2, 3);

    UTEST_CHECK_EQUAL(::nano::hash(fvalues(0)), 0U);
    UTEST_CHECK_EQUAL(::nano::hash(fvalues(1)), 1U);
    UTEST_CHECK_EQUAL(::nano::hash(fvalues(2)), 2U);
    UTEST_CHECK_EQUAL(::nano::hash(fvalues(3)), 3U);
}

UTEST_CASE(hash_mclass)
{
    const auto fvalues = make_tensor<int8_t, 2>(make_dims(10, 3), 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                                0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1);

    const auto hash0 = ::nano::hash(fvalues.array(0));
    const auto hash1 = ::nano::hash(fvalues.array(1));
    const auto hash2 = ::nano::hash(fvalues.array(2));
    const auto hash3 = ::nano::hash(fvalues.array(3));
    const auto hash4 = ::nano::hash(fvalues.array(4));
    const auto hash5 = ::nano::hash(fvalues.array(5));
    const auto hash6 = ::nano::hash(fvalues.array(6));
    const auto hash7 = ::nano::hash(fvalues.array(7));
    const auto hash8 = ::nano::hash(fvalues.array(8));
    const auto hash9 = ::nano::hash(fvalues.array(9));

    UTEST_CHECK_EQUAL(hash4, hash0);
    UTEST_CHECK_EQUAL(hash5, hash6);
    UTEST_CHECK_EQUAL(hash7, hash1);
    UTEST_CHECK_EQUAL(hash8, hash3);

    UTEST_CHECK_NOT_EQUAL(hash0, hash1);
    UTEST_CHECK_NOT_EQUAL(hash0, hash2);
    UTEST_CHECK_NOT_EQUAL(hash0, hash3);
    UTEST_CHECK_NOT_EQUAL(hash0, hash5);
    UTEST_CHECK_NOT_EQUAL(hash0, hash6);
    UTEST_CHECK_NOT_EQUAL(hash0, hash7);
    UTEST_CHECK_NOT_EQUAL(hash0, hash8);
    UTEST_CHECK_NOT_EQUAL(hash0, hash9);

    UTEST_CHECK_NOT_EQUAL(hash1, hash2);
    UTEST_CHECK_NOT_EQUAL(hash1, hash3);
    UTEST_CHECK_NOT_EQUAL(hash1, hash4);
    UTEST_CHECK_NOT_EQUAL(hash1, hash6);
    UTEST_CHECK_NOT_EQUAL(hash1, hash8);
    UTEST_CHECK_NOT_EQUAL(hash1, hash9);

    UTEST_CHECK_NOT_EQUAL(hash2, hash3);
    UTEST_CHECK_NOT_EQUAL(hash2, hash4);
    UTEST_CHECK_NOT_EQUAL(hash2, hash5);
    UTEST_CHECK_NOT_EQUAL(hash2, hash6);
    UTEST_CHECK_NOT_EQUAL(hash2, hash7);
    UTEST_CHECK_NOT_EQUAL(hash2, hash8);
    UTEST_CHECK_NOT_EQUAL(hash2, hash9);

    UTEST_CHECK_NOT_EQUAL(hash3, hash4);
    UTEST_CHECK_NOT_EQUAL(hash3, hash5);
    UTEST_CHECK_NOT_EQUAL(hash3, hash6);
    UTEST_CHECK_NOT_EQUAL(hash3, hash7);
    UTEST_CHECK_NOT_EQUAL(hash3, hash9);

    UTEST_CHECK_NOT_EQUAL(hash4, hash5);
    UTEST_CHECK_NOT_EQUAL(hash4, hash6);
    UTEST_CHECK_NOT_EQUAL(hash4, hash7);
    UTEST_CHECK_NOT_EQUAL(hash4, hash8);
    UTEST_CHECK_NOT_EQUAL(hash4, hash9);

    UTEST_CHECK_NOT_EQUAL(hash5, hash7);
    UTEST_CHECK_NOT_EQUAL(hash5, hash8);
    UTEST_CHECK_NOT_EQUAL(hash5, hash9);

    UTEST_CHECK_NOT_EQUAL(hash6, hash7);
    UTEST_CHECK_NOT_EQUAL(hash6, hash8);
    UTEST_CHECK_NOT_EQUAL(hash6, hash9);

    UTEST_CHECK_NOT_EQUAL(hash7, hash8);
    UTEST_CHECK_NOT_EQUAL(hash7, hash9);

    UTEST_CHECK_NOT_EQUAL(hash8, hash9);
}

UTEST_CASE(hash_sclass_make_and_find)
{
    const auto fvalues = make_tensor<int32_t, 1>(make_dims(12), 0, 1, 2, 0, 1, 0, 2, 1, 1, 2, 2, 0);

    const auto hashes = ::nano::make_hashes(fvalues);
    UTEST_CHECK_EQUAL(hashes.size(), 3);

    const auto fvalues_test = make_tensor<int32_t, 1>(make_dims(7), 0, 1, 3, 2, 1, -1, 4);

    const auto index0 = ::nano::find(hashes, fvalues_test(0));
    const auto index1 = ::nano::find(hashes, fvalues_test(1));
    const auto index2 = ::nano::find(hashes, fvalues_test(2));
    const auto index3 = ::nano::find(hashes, fvalues_test(3));
    const auto index4 = ::nano::find(hashes, fvalues_test(4));
    const auto index5 = ::nano::find(hashes, fvalues_test(5));
    const auto index6 = ::nano::find(hashes, fvalues_test(6));

    UTEST_CHECK_EQUAL(index0, +0);
    UTEST_CHECK_EQUAL(index1, +1);
    UTEST_CHECK_EQUAL(index2, -1);
    UTEST_CHECK_EQUAL(index3, +2);
    UTEST_CHECK_EQUAL(index4, +1);
    UTEST_CHECK_EQUAL(index5, -1);
    UTEST_CHECK_EQUAL(index6, -1);
}

UTEST_CASE(hash_mclass_make_and_find)
{
    const auto fvalues = make_tensor<int8_t, 2>(make_dims(12, 3), 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
                                                0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, -1, -1, -1, 0, 0, 0);

    const auto hashes = ::nano::make_hashes(fvalues);
    UTEST_CHECK_EQUAL(hashes.size(), 6);

    const auto fvalues_test =
        make_tensor<int8_t, 2>(make_dims(7, 3), -1, -1, -1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1);

    const auto index0 = ::nano::find(hashes, fvalues_test.array(0));
    const auto index1 = ::nano::find(hashes, fvalues_test.array(1));
    const auto index2 = ::nano::find(hashes, fvalues_test.array(2));
    const auto index3 = ::nano::find(hashes, fvalues_test.array(3));
    const auto index4 = ::nano::find(hashes, fvalues_test.array(4));
    const auto index5 = ::nano::find(hashes, fvalues_test.array(5));
    const auto index6 = ::nano::find(hashes, fvalues_test.array(6));

    UTEST_CHECK_EQUAL(index0, -1);
    UTEST_CHECK_EQUAL(index1, +0);
    UTEST_CHECK_EQUAL(index2, +4);
    UTEST_CHECK_EQUAL(index3, +2);
    UTEST_CHECK_EQUAL(index4, +5);
    UTEST_CHECK_EQUAL(index5, +3);
    UTEST_CHECK_EQUAL(index6, +1);
}

UTEST_END_MODULE()
