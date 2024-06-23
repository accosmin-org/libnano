#pragma once

#include <nano/core/chrono.h>
#include <nano/dataset.h>

using namespace nano;

// FIXME: move it to library!
[[maybe_unused]] inline dataset_t load_dataset(const datasource_t& datasource, const strings_t& generator_ids)
{
    const auto timer   = ::nano::timer_t{};
    auto       dataset = dataset_t{datasource};
    for (const auto& generator_id : generator_ids)
    {
        dataset.add(generator_t::all().get(generator_id));
    }

    const auto logger  = make_stdout_logger();
    const auto elapsed = timer.elapsed();
    logger.log(log_type::info, "=> dataset loaded with feature generators loaded in <", elapsed, ">.\n");
    logger.log(log_type::info, "..columns=", dataset.columns(), "\n");
    logger.log(log_type::info, "..target=[", dataset.target(), "]\n");
    return dataset;
}
