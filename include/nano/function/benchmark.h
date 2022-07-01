#pragma once

#include <nano/core/factory.h>
#include <nano/function.h>

namespace nano
{
    class benchmark_function_t;
    using function_factory_t = factory_t<benchmark_function_t>;

    enum class convexity
    {
        ignore,
        yes,
        no
    };

    enum class smoothness
    {
        ignore,
        yes,
        no
    };

    ///
    /// \brief test function useful for benchmarking numerical optimization methods.
    ///
    class NANO_PUBLIC benchmark_function_t : public function_t
    {
    public:
        using function_t::function_t;

        ///
        /// \brief returns the available implementations.
        ///
        static function_factory_t& all();

        ///
        /// \brief construct test functions having:
        ///     - the number of dimensions within the given range,
        ///     - the given number of summands and
        ///     - the given requirements in terms of smoothness and convexity.
        ///
        struct config_t
        {
            tensor_size_t m_min_dims{2};                    ///<
            tensor_size_t m_max_dims{8};                    ///<
            convexity     m_convexity{convexity::ignore};   ///<
            smoothness    m_smoothness{smoothness::ignore}; ///<
            tensor_size_t m_summands{1000};                 ///<
        };

        static rfunctions_t make(config_t, const std::regex& id_regex = std::regex(".+"));

        ///
        /// \brief construct a test function with the given number of free dimensions and summands (if possible).
        ///
        virtual rfunction_t make(tensor_size_t dims, tensor_size_t summands) const = 0;
    };
} // namespace nano
