#include <nano/tuner/local.h>
#include <nano/tuner/util.h>

using namespace nano;

local_search_tuner_t::local_search_tuner_t()
    : tuner_t("local-search")
{
}

rtuner_t local_search_tuner_t::clone() const
{
    return std::make_unique<local_search_tuner_t>(*this);
}

void local_search_tuner_t::do_optimize(const param_spaces_t& spaces, const tuner_callback_t& callback,
                                       tuner_steps_t& steps) const
{
    const auto max_evals = parameter("tuner::max_evals").value<size_t>();
    const auto min_igrid = make_min_igrid(spaces);
    const auto max_igrid = make_max_igrid(spaces);

    // local search around current optimum iteratively...
    for (; !steps.empty() && steps.size() < max_evals;)
    {
        const auto igrids = local_search(min_igrid, max_igrid, steps.begin()->m_igrid, 1);
        if (!evaluate(spaces, callback, igrids, steps))
        {
            break;
        }
    }
}
