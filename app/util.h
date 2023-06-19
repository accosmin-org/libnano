#pragma once

#include <nano/core/chrono.h>
#include <nano/dataset.h>

using namespace nano;

[[maybe_unused]] inline auto load_dataset(const datasource_t& datasource, const strings_t& generator_ids)
{
    const auto timer   = ::nano::timer_t{};
    auto       dataset = dataset_t{datasource};
    for (const auto& generator_id : generator_ids)
    {
        dataset.add(generator_t::all().get(generator_id));
    }
    const auto elapsed = timer.elapsed();
    log_info() << "=> dataset loaded with feature generators loaded in <" << elapsed << ">.";
    log_info() << "..columns=" << dataset.columns();
    log_info() << "..target=[" << dataset.target() << "]";
    return dataset;
}
