#pragma once

#include <cassert>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <nano/arch.h>
#include <thread>
#include <vector>

namespace nano::parallel
{
    using future_t = std::future<void>;
    using task_t   = std::packaged_task<void(size_t)>;

    ///
    /// \brief enqueue tasks to be run in a thread pool.
    ///
    class NANO_PUBLIC queue_t
    {
    public:
        ///
        /// \brief constructor
        ///
        queue_t();

        ///
        /// \brief enqueue a new task to execute
        ///
        template <typename tfunction>
        auto enqueue(tfunction&& f)
        {
            auto task   = task_t(f);
            auto future = task.get_future();
            {
                const std::lock_guard<std::mutex> lock(m_mutex);
                m_tasks.emplace_back(std::move(task));
            }
            m_condition.notify_one();
            return future;
        }

        // attributes
        std::deque<task_t>              m_tasks;       ///< tasks to execute
        mutable std::mutex              m_mutex;       ///< synchronization
        mutable std::condition_variable m_condition;   ///< signaling
        bool                            m_stop{false}; ///< stop requested
    };

    ///
    /// \brief worker to process tasks enqueued in a thread pool.
    ///
    class NANO_PUBLIC worker_t
    {
    public:
        ///
        /// \brief constructor
        ///
        worker_t(queue_t& queue, size_t tnum);

        ///
        /// \brief execute tasks when available.
        ///
        void operator()() const;

    private:
        // attributes
        queue_t& m_queue;    ///< task queue to process
        size_t   m_tnum{0U}; ///< thread number
    };

    ///
    /// \brief RAII object to wait for a given set of futures (aka barrier).
    ///
    class NANO_PUBLIC section_t : public std::vector<future_t>
    {
    public:
        using std::vector<future_t>::vector;

        ///
        /// \brief default constructor
        ///
        section_t() = default;

        ///
        /// \brief disable copying
        ///
        section_t(const section_t&)            = delete;
        section_t& operator=(const section_t&) = delete;

        ///
        /// \brief enable moving
        ///
        section_t(section_t&&) noexcept            = default;
        section_t& operator=(section_t&&) noexcept = default;

        ///
        /// \brief destructor (wait all futures).
        ///
        ~section_t();

        ///
        /// \brief block until all futures are done w/o raising any exception produced by the worker thread.
        ///
        void block(bool raise);
    };

    ///
    /// \brief thread pool with a fixed number of threads.
    ///
    /// NB: the default number of threads is the number of logical cores.
    /// NB: the given number of threads is clamped to [1, number of logical cores].
    ///
    class NANO_PUBLIC pool_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit pool_t();
        explicit pool_t(size_t threads);

        ///
        /// \brief disable copying
        ///
        pool_t(const pool_t&)            = delete;
        pool_t& operator=(const pool_t&) = delete;

        ///
        /// \brief disable moving
        ///
        pool_t(pool_t&&) noexcept            = delete;
        pool_t& operator=(pool_t&&) noexcept = delete;

        ///
        /// \brief destructor
        ///
        ~pool_t();

        ///
        /// \brief enqueue a new task to execute.
        ///
        template <typename tfunction>
        auto enqueue(tfunction&& f)
        {
            return m_queue.enqueue(std::move(f));
        }

        ///
        /// \brief returns the number of available worker threads.
        ///
        auto size() const { return m_threads.size(); }

        ///
        /// \brief returns the maximum number of available threads (typically the number of logical cores).
        ///
        static size_t max_size();

        ///
        /// \brief process the given number of elements in parallel.
        ///
        /// NB: the operator receives the element index to process and the assigned thread index:
        ///     op(index, tnum)
        ///
        template <typename tsize, typename toperator, std::enable_if_t<std::is_integral_v<tsize>, bool> = true>
        void loopi(tsize elements, const toperator& op, bool raise = true)
        {
            if (size() == 1)
            {
                for (tsize index = 0; index < elements; ++index)
                {
                    op(index, 0U);
                }
            }
            else
            {
                section_t section;
                for (tsize index = 0; index < elements; ++index)
                {
                    section.emplace_back(enqueue([&op, index = index](size_t tnum) { op(index, tnum); }));
                }

                section.block(raise);
            }
        }

        ///
        /// \brief process the given number of elements in parallel in chunks of fixed size.
        ///
        /// NB: the operator receives the range [begin, end) of elements to process and the assigned thread index:
        ///     op(begin, end, tnum)
        ///
        template <typename tsize, typename toperator, std::enable_if_t<std::is_integral_v<tsize>, bool> = true>
        void loopr(tsize elements, tsize chunksize, const toperator& op, bool raise = true)
        {
            assert(chunksize >= tsize(1));

            if (size() == 1)
            {
                for (tsize begin = 0; begin < elements; begin += chunksize)
                {
                    op(begin, std::min(begin + chunksize, elements), 0U);
                }
            }
            else
            {
                section_t section;
                for (tsize begin = 0; begin < elements; begin += chunksize)
                {
                    section.emplace_back(
                        enqueue([&op, begin = begin, elements = elements, chunksize = chunksize](size_t tnum)
                                { op(begin, std::min(begin + chunksize, elements), tnum); }));
                }

                section.block(raise);
            }
        }

    private:
        // attributes
        std::vector<std::thread> m_threads; ///<
        std::vector<worker_t>    m_workers; ///<
        queue_t                  m_queue;   ///< tasks to execute + synchronization
    };
} // namespace nano::parallel
