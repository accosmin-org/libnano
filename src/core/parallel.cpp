#include <algorithm>
#include <iterator>
#include <nano/core/parallel.h>

using namespace nano;
using namespace nano::parallel;

queue_t::queue_t() = default;

worker_t::worker_t(queue_t& queue, size_t tnum)
    : m_queue(queue)
    , m_tnum(tnum)
{
}

void worker_t::operator()() const
{
    while (true)
    {
        task_t task;

        // wait for a new task to be available in the queue
        {
            std::unique_lock lock(m_queue.m_mutex);

            m_queue.m_condition.wait(lock, [&] { return m_queue.m_stop || !m_queue.m_tasks.empty(); });

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

void section_t::block(const bool raise)
{
    for (const auto& future : *this)
    {
        if (future.valid())
        {
            raise ? (void)future.get() : future.wait();
        }
    }
}

section_t::~section_t()
{
    block(false);
}

pool_t::pool_t()
    : pool_t(max_size())
{
}

pool_t::pool_t(const size_t threads)
{
    const auto n_workers = std::clamp(threads, size_t(1), max_size());

    m_workers.reserve(n_workers);
    for (size_t tnum = 0; tnum < n_workers; ++tnum)
    {
        m_workers.emplace_back(m_queue, tnum);
    }

    std::transform(m_workers.begin(), m_workers.end(), std::back_inserter(m_threads),
                   [](const auto& worker) { return std::thread(std::cref(worker)); });
}

size_t pool_t::max_size()
{
    return std::max(size_t(1), static_cast<size_t>(std::thread::hardware_concurrency()));
}

pool_t::~pool_t()
{
    {
        const std::scoped_lock lock(m_queue.m_mutex);
        m_queue.m_stop = true;
    }
    m_queue.m_condition.notify_all();

    for (auto& thread : m_threads)
    {
        thread.join();
    }
}
