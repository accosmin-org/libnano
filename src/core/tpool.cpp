#include <iterator>
#include <algorithm>
#include <nano/core/tpool.h>

using namespace nano;

tpool_queue_t::tpool_queue_t() = default;

tpool_worker_t::tpool_worker_t(tpool_queue_t& queue, size_t tnum) :
    m_queue(queue),
    m_tnum(tnum)
{
}

void tpool_worker_t::operator()() const
{
    while (true)
    {
        tpool_task_t task;

        // wait for a new task to be available in the queue
        {
            std::unique_lock<std::mutex> lock(m_queue.m_mutex);

            m_queue.m_condition.wait(lock, [&]
            {
                return m_queue.m_stop || !m_queue.m_tasks.empty();
            });

            if (m_queue.m_stop)
            {
                m_queue.m_tasks.clear();
                m_queue.m_condition.notify_all();
                break;
            }

            task = std::move(m_queue.m_tasks.front());
            m_queue.m_tasks.pop_front();
        }

        // execute the task
        task(m_tnum);
    }
}

tpool_t::tpool_t()
{
    const auto n_workers = size();

    m_workers.reserve(n_workers);
    for (size_t tnum = 0; tnum < n_workers; ++ tnum)
    {
        m_workers.emplace_back(m_queue, tnum);
    }

    std::transform(
        m_workers.begin(), m_workers.end(), std::back_inserter(m_threads),
        [] (const auto& worker) { return std::thread(std::cref(worker)); });
}

tpool_t::~tpool_t()
{
    {
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);
        m_queue.m_stop = true;
        m_queue.m_condition.notify_all();
    }

    for (auto& thread : m_threads)
    {
        thread.join();
    }
}
