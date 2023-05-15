#pragma once

#include <nano/core/cmdline.h>
#include <nano/core/factory_util.h>
#include <nano/core/logger.h>
#include <nano/datasource.h>
#include <nano/generator.h>
#include <nano/loss.h>
#include <nano/lsearch0.h>
#include <nano/lsearchk.h>
#include <nano/solver.h>
#include <nano/splitter.h>
#include <nano/tuner.h>
#include <nano/version.h>

using namespace nano;

namespace
{
///
/// \brief RAII utility to keep track of the used parameters and
///     log all unused parameters at the end (e.g. typos, not matching to any solver).
///
class parameter_tracker_t
{
public:
    parameter_tracker_t(const cmdline_t::result_t& options)
        : m_options(options)
    {
        for (const auto& [param_name, param_value] : m_options.m_xvalues)
        {
            m_params_usage[param_name] = 0;
        }
    }

    void setup(configurable_t& configurable)
    {
        for (const auto& [param_name, param_value] : m_options.m_xvalues)
        {
            if (configurable.parameter_if(param_name) != nullptr)
            {
                configurable.parameter(param_name) = param_value;
                m_params_usage[param_name]++;
            }
        }
    }

    ~parameter_tracker_t()
    {
        for (const auto& [param_name, count] : m_params_usage)
        {
            if (count == 0)
            {
                log_warning() << "parameter \"" << param_name << "\" was not used.";
            }
        }
    }

private:
    // attributes
    const cmdline_t::result_t& m_options;      ///<
    std::map<string_t, int>    m_params_usage; ///<
};

[[maybe_unused]] void setup_mlearn(cmdline_t& cmdline)
{
    cmdline.add("", "loss", "regex to select loss functions");
    cmdline.add("", "datasource", "regex to select machine learning datasets");
    cmdline.add("", "generator", "regex to select feature generation methods");
    cmdline.add("", "splitter", "regex to select train-validation splitting methods");
    cmdline.add("", "tuner", "regex to select hyper-parameter tuning methods");

    cmdline.add("", "list-loss", "list the available loss functions");
    cmdline.add("", "list-datasource", "list the available machine learning datasets");
    cmdline.add("", "list-generator", "list the available feature generation methods");
    cmdline.add("", "list-splitter", "list the available train-validation splitting methods");
    cmdline.add("", "list-tuner", "list the available hyper-parameter tuning methods");

    cmdline.add("", "list-solver-params", "list the available parameters of the selected solvers");
    cmdline.add("", "list-lsearch0-params",
                "list the available parameters of the selected line-search initialization methods");
    cmdline.add("", "list-lsearchk-params", "list the available parameters of the selected line-search strategies");
}

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

///
/// \brief handle common command line requests:
///     (e.g. help, version, list available implementations, list available parameters).
///
auto process(const cmdline_t& cmdline, const int argc, const char* argv[])
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
}
