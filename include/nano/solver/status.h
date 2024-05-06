#pragma once

#include <nano/core/strutil.h>

namespace nano
{
enum class solver_status : uint8_t
{
    max_iters,  ///< maximum number of iterations reached without convergence (default)
    converged,  ///< convergence criterion reached
    failed,     ///< optimization failed (e.g. line-search failed)
    stopped,    ///< user requested stop
    unfeasible, ///< cannot find any feasible point (if constrained)
    unbounded,  ///< problem is not lower-bounded
};

template <>
inline enum_map_t<solver_status> enum_string<solver_status>()
{
    return {
        { solver_status::max_iters,  "max_iters"},
        { solver_status::converged,  "converged"},
        {    solver_status::failed,     "failed"},
        {   solver_status::stopped,    "stopped"},
        {solver_status::unfeasible, "unfeasible"},
        { solver_status::unbounded,  "unbounded"}
    };
}
} // namespace nano
