#pragma once

#include <nano/function.h>
#include <nano/core/factory.h>

namespace nano
{
    class benchmark_function_t;
    using function_factory_t = factory_t<benchmark_function_t>;
    using rfunction_t = std::unique_ptr<function_t>;
    using rfunctions_t = std::vector<rfunction_t>;

    enum class convexity
    {
        ignore, yes, no
    };

    enum class smoothness
    {
        ignore, yes, no
    };

    struct benchmark_function_config_t
    {
        tensor_size_t   m_min_dims{2};                      ///<
        tensor_size_t   m_max_dims{8};                      ///<
        convexity       m_convexity{convexity::ignore};     ///<
        smoothness      m_smoothness{smoothness::ignore};   ///<
    };

    ///
    /// \brief construct test functions having the number of dimensions within the given range.
    ///
    NANO_PUBLIC rfunctions_t make_benchmark_functions(
        benchmark_function_config_t,
        const std::regex& id_regex = std::regex(".+"));

    ///
    /// \brief test function useful for benchmarking numerical optimization methods.
    ///
    class NANO_PUBLIC benchmark_function_t : public function_t
    {
    public:

        using function_t::function_t;

        ///
        /// \brief returns the available implementations
        ///
        static function_factory_t& all();

        ///
        /// \brief construct a test function with the given number of free dimensions.
        ///
        virtual rfunction_t make(tensor_size_t dims) const = 0;
    };
}
