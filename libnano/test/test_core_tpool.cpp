#include <numeric>
#include <nano/arch.h>
#include <nano/tpool.h>
#include <nano/random.h>
#include <utest/utest.h>
#include <nano/numeric.h>

using namespace nano;

namespace
{
    // single-threaded
    template <typename tscalar, typename toperator>
    tscalar test_st(const size_t size, const toperator op)
    {
        std::vector<tscalar> results(size);
        for (size_t i = 0; i < results.size(); ++ i)
        {
            results[i] = op(i);
        }

        return std::accumulate(results.begin(), results.end(), tscalar(0));
    }

    // multi-threaded
    template <int tchunk_size, typename tscalar, typename toperator>
    tscalar test_mt(const size_t size, const toperator op)
    {
        std::vector<tscalar> results(size, -1);
        nano::loopi<tchunk_size>(size, [&results = results, size = size, op = op] (const size_t i)
        {
            assert(i < size);
            NANO_UNUSED1_RELEASE(size);
            results[i] = op(i);
        });

        return std::accumulate(results.begin(), results.end(), tscalar(0));
    }
}

UTEST_BEGIN_MODULE(test_core_tpool)

UTEST_CASE(empty)
{
    auto& pool = tpool_t::instance();

    UTEST_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
    UTEST_CHECK_EQUAL(pool.tasks(), 0u);
}

UTEST_CASE(enqueue)
{
    auto& pool = tpool_t::instance();

    UTEST_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
    UTEST_CHECK_EQUAL(pool.tasks(), 0u);

    const size_t max_tasks = 1024;
    const auto tasks = urand<size_t>(1u, max_tasks, make_rng());

    std::vector<size_t> tasks_done;

    std::mutex mutex;
    {
        tpool_section_t<future_t> futures;
        for (size_t j = 0; j < tasks; ++ j)
        {
            futures.push_back(pool.enqueue([=, &mutex, &tasks_done]()
            {
                const auto sleep1 = urand<size_t>(1, 5, make_rng());
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));

                {
                    const std::lock_guard<std::mutex> lock(mutex);

                    tasks_done.push_back(j + 1);
                }
            }));
        }
    }

    UTEST_CHECK_EQUAL(pool.workers(), nano::physical_cpus());
    UTEST_CHECK_EQUAL(pool.tasks(), 0u);

    UTEST_CHECK_EQUAL(tasks_done.size(), tasks);
    for (size_t j = 0; j < tasks; ++ j)
    {
        UTEST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
    }
}

UTEST_CASE(evaluate)
{
    const size_t min_size = 1;
    const size_t max_size = 9 * 9 * 9 * 1;

    using scalar_t = double;

    // operator to test
    const auto op = [](const size_t i)
    {
        const auto ii = static_cast<scalar_t>(i);
        return ii * ii + 1 - ii;
    };

    // test for different problems size
    for (size_t size = min_size; size <= max_size; size *= 3)
    {
        const auto st = test_st<scalar_t>(size, op);
        const auto mt_m1 = test_mt<-1, scalar_t>(size, op);
        const auto mt_p0 = test_mt<+0, scalar_t>(size, op);
        const auto mt_p1 = test_mt<+1, scalar_t>(size, op);
        const auto mt_p2 = test_mt<+2, scalar_t>(size, op);
        const auto mt_p3 = test_mt<+3, scalar_t>(size, op);
        const auto mt_p33 = test_mt<+33, scalar_t>(size, op);
        const auto mt_p333 = test_mt<+333, scalar_t>(size, op);
        const auto mt_p3333 = test_mt<+3333, scalar_t>(size, op);

        UTEST_CHECK_CLOSE(st, mt_m1, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p0, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p1, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p2, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p3, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p33, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p333, nano::epsilon1<scalar_t>());
        UTEST_CHECK_CLOSE(st, mt_p3333, nano::epsilon1<scalar_t>());
    }
}

UTEST_END_MODULE()
