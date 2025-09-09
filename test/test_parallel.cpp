#include <algorithm>
#include <nano/core/numeric.h>
#include <nano/core/parallel.h>
#include <nano/core/random.h>
#include <numeric>
#include <set>
#include <utest/utest.h>

using namespace nano;

namespace
{
// single-threaded
template <class toperator>
auto test_single(size_t size, const toperator op)
{
    std::vector<double> results(size);
    for (size_t i = 0; i < results.size(); ++i)
    {
        results[i] = op(i);
    }

    return std::accumulate(results.begin(), results.end(), 0.0);
}

// multi-threaded (by index)
template <class toperator>
auto test_loopi(parallel::pool_t& pool, size_t size, const toperator op)
{
    std::vector<double> results(size, -1);
    pool.map(size,
             [&](size_t i, size_t tnum)
             {
                 UTEST_CHECK_LESS(i, size);
                 UTEST_CHECK_LESS(tnum, pool.size());

                 results[i] = op(i);
             });

    return std::accumulate(results.begin(), results.end(), 0.0);
}

// multi-threaded (by range)
template <class toperator>
auto test_loopr(parallel::pool_t& pool, size_t size, size_t chunk, const toperator op)
{
    std::vector<double> results(size, -1);
    pool.map(size, chunk, // NOLINT(readability-suspicious-call-argument)
             [&](size_t begin, size_t end, size_t tnum)
             {
                 UTEST_CHECK_LESS(begin, end);
                 UTEST_CHECK_LESS_EQUAL(end, size);
                 UTEST_CHECK_LESS(tnum, pool.size());
                 UTEST_CHECK_LESS_EQUAL(end - begin, chunk);

                 for (auto i = begin; i < end; ++i)
                 {
                     results[i] = op(i);
                 }
             });

    return std::accumulate(results.begin(), results.end(), 0.0);
}

auto thread_counts()
{
    std::set<size_t> threads;
    threads.insert(1U);
    threads.insert(std::thread::hardware_concurrency() - 1U);
    threads.insert(std::thread::hardware_concurrency() + 1U);
    threads.insert(std::thread::hardware_concurrency());
    return threads;
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(init)
{
    UTEST_CHECK_EQUAL(parallel::pool_t::max_size(), std::thread::hardware_concurrency());

    {
        auto pool = parallel::pool_t{0U};
        UTEST_CHECK_EQUAL(pool.size(), 1U);
    }
    {
        auto pool = parallel::pool_t{1U};
        UTEST_CHECK_EQUAL(pool.size(), 1U);
    }
    {
        auto pool = parallel::pool_t{};
        UTEST_CHECK_EQUAL(pool.size(), std::thread::hardware_concurrency());
    }
    {
        auto pool = parallel::pool_t{std::thread::hardware_concurrency()};
        UTEST_CHECK_EQUAL(pool.size(), std::thread::hardware_concurrency());
    }
    {
        auto pool = parallel::pool_t{std::thread::hardware_concurrency() + 1U};
        UTEST_CHECK_EQUAL(pool.size(), std::thread::hardware_concurrency());
    }
}

UTEST_CASE(future)
{
    std::packaged_task<int(int, int)> task([](int a, int b) { return static_cast<int>(std::pow(a, b)); });

    auto future = task.get_future();
    task(2, 9);

    UTEST_CHECK_EQUAL(future.get(), 512);
}

UTEST_CASE(future_join)
{
    std::packaged_task<int(int, int)> task([](int a, int b) { return static_cast<int>(std::pow(a, b)); });

    auto future = task.get_future();
    auto thread = std::thread{std::move(task), 2, 10};
    thread.join();

    UTEST_CHECK_EQUAL(future.get(), 1024);
}

UTEST_CASE(future_detach)
{
    std::packaged_task<int(int, int)> task([](int a, int b) { return static_cast<int>(std::pow(a, b)); });

    auto future = task.get_future();
    auto thread = std::thread{std::move(task), 2, 11};
    thread.detach();

    UTEST_CHECK_EQUAL(future.get(), 2048);
}

UTEST_CASE(enqueue)
{
    auto pool = parallel::pool_t{};

    const auto max_tasks = size_t{1024};
    const auto tasks     = urand<size_t>(1U, max_tasks);

    std::mutex          mutex;
    std::vector<size_t> tasks_done;
    {
        parallel::section_t futures;
        for (size_t j = 0; j < tasks; ++j)
        {
            futures.push_back(pool.enqueue(
                [=, &mutex, &tasks_done](size_t)
                {
                    const auto sleep1 = urand<size_t>(1, 5);
                    std::this_thread::sleep_for(std::chrono::milliseconds(sleep1));
                    {
                        const std::lock_guard<std::mutex> lock(mutex);
                        tasks_done.push_back(j + 1);
                    }
                }));
        }
        // NB: all futures have finished here!
    }

    UTEST_CHECK_EQUAL(tasks_done.size(), tasks);
    for (size_t j = 0; j < tasks; ++j)
    {
        UTEST_CHECK(std::find(tasks_done.begin(), tasks_done.end(), j + 1) != tasks_done.end());
    }
}

UTEST_CASE(loopi)
{
    const auto op = [](size_t i) { return std::sin(i); };

    for (const auto threads : thread_counts())
    {
        auto pool = parallel::pool_t{threads};

        for (size_t size = 1; size <= size_t(123); size *= 3)
        {
            const auto eps = epsilon1<double>();
            const auto ref = test_single(size, op);

            UTEST_CHECK_CLOSE(ref, test_loopi(pool, size, op), eps);
        }
    }
}

UTEST_CASE(loopr)
{
    const auto op = [](size_t i) { return std::cos(i); };

    for (const auto threads : thread_counts())
    {
        auto pool = parallel::pool_t{threads};

        for (size_t size = 1; size <= size_t(128); size *= 2)
        {
            const auto eps = epsilon1<double>();
            const auto ref = test_single(size, op);

            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, 1, op), eps);
            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, 2, op), eps);
            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, 3, op), eps);
            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, 4, op), eps);
            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, size, op), eps);
            UTEST_CHECK_CLOSE(ref, test_loopr(pool, size, size + 1, op), eps);
        }
    }
}

UTEST_END_MODULE()
