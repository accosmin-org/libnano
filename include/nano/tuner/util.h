#pragma once

#include <nano/tuner/callback.h>
#include <nano/tuner/space.h>
#include <nano/tuner/step.h>

namespace nano
{
class logger_t;
using igrid_t  = indices_t;
using igrids_t = std::vector<igrid_t>;

///
/// \brief returns the grid point with the minimum grid indices (useful for limiting local search).
///
NANO_PUBLIC igrid_t make_min_igrid(const param_spaces_t&);

///
/// \brief returns the grid point with the maximum grid indices (useful for limiting local search).
///
NANO_PUBLIC igrid_t make_max_igrid(const param_spaces_t&);

///
/// \brief returns the grid point with the average grid indices.
///
NANO_PUBLIC igrid_t make_avg_igrid(const param_spaces_t&);

///
/// \brief map the given grid points to hyper-parameter values.
///
NANO_PUBLIC tensor2d_t map_to_grid(const param_spaces_t&, const igrids_t& igrids);

///
/// \brief returns the grid points in a given radius from the source grid point.
///
NANO_PUBLIC igrids_t local_search(const igrid_t& min_igrid, const igrid_t& max_igrid, const igrid_t& src_igrid,
                                  tensor_size_t radius);

///
/// \brief evaluate the given grid points (if not already) and update the given tuner steps.
///     returns true if at least one new grid point needs to be evaluated.
///
NANO_PUBLIC bool evaluate(const param_spaces_t&, const tuner_callback_t&, igrids_t igrids, const logger_t&,
                          tuner_steps_t&);
} // namespace nano
