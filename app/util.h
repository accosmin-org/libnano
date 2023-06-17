#pragma once

#include <nano/core/chrono.h>
#include <nano/core/cmdline.h>
#include <nano/core/factory_util.h>
#include <nano/core/logger.h>
#include <nano/dataset/iterator.h>
#include <nano/loss.h>
#include <nano/lsearch0.h>
#include <nano/lsearchk.h>
#include <nano/solver.h>
#include <nano/splitter.h>
#include <nano/tuner.h>
#include <nano/version.h>

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

///
/// \brief handle common command line requests:
///     (e.g. help, version, list builtin factory objects, list available parameters for selected objects).
///
template <typename tobject>
void process_list(const string_t& name, const cmdline_t::result_t& options)
{
    if (options.has("list-" + name))
    {
        std::cout << make_table(name, tobject::all(), options.get<string_t>(name));
        std::exit(EXIT_SUCCESS);
    }

    if constexpr (std::is_base_of_v<configurable_t, tobject>)
    {
        if (options.has("list-" + name + "-params"))
        {
            std::cout << make_table_with_params(name, tobject::all(), options.get<string_t>(name));
            std::exit(EXIT_SUCCESS);
        }
    }
}

inline auto process(const cmdline_t& cmdline, const int argc, const char* argv[])
{
    auto options = cmdline.process(argc, argv);

    if (options.has("help"))
    {
        cmdline.usage();
        std::exit(EXIT_SUCCESS);
    }
    if (options.has("version"))
    {
        std::cout << nano::major_version << "." << nano::minor_version << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    if (options.has("git-hash"))
    {
        std::cout << nano::git_commit_hash << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    process_list<lsearch0_t>("lsearch0", options);
    process_list<lsearchk_t>("lsearchk", options);
    process_list<solver_t>("solver", options);
    process_list<function_t>("function", options);
    process_list<loss_t>("loss", options);
    process_list<datasource_t>("datasource", options);
    process_list<generator_t>("generator", options);
    process_list<splitter_t>("splitter", options);
    process_list<tuner_t>("tuner", options);

    return options;
}
