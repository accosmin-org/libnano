#pragma once

#include <nano/lsearch/lsearchk.h>

namespace nano
{
    ///
    /// \brief backtracking line-search that stops when the Armijo condition is satisfied,
    ///     see "Numerical optimization", Nocedal & Wright, 2nd edition
    ///
    class lsearchk_backtrack_t final : public lsearchk_t
    {
    public:

        ///
        /// \brief interpolation method using the information at step length zero and the current step length.
        ///
        enum class interpolation
        {
            bisect,                 ///<
            cubic                   ///<
        };

        lsearchk_backtrack_t() = default;

        json_t config() const final;
        void config(const json_t&) final;
        bool get(const solver_state_t& state0, solver_state_t& state) final;

    private:

        // attributes
        interpolation   m_method{interpolation::cubic}; ///<
    };

    template <>
    inline enum_map_t<lsearchk_backtrack_t::interpolation> enum_string<lsearchk_backtrack_t::interpolation>()
    {
        return
        {
            { lsearchk_backtrack_t::interpolation::bisect,      "bisect"},
            { lsearchk_backtrack_t::interpolation::cubic,       "cubic"}
        };
    }
}
