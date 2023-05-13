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
[[maybe_unused]] void setup_solver(cmdline_t& cmdline)
{
    cmdline.add("", "lsearch0", "regex to select line-search initialization methods");
    cmdline.add("", "lsearchk", "regex to select line-search strategies");
    cmdline.add("", "solver", "regex to select solvers");

    cmdline.add("", "list-lsearch0", "list the available line-search initialization methods");
    cmdline.add("", "list-lsearchk", "list the available line-search strategies");
    cmdline.add("", "list-solver", "list the available solvers");

    cmdline.add("", "list-solver-params", "list the available parameters of the selected solvers");
    cmdline.add("", "list-lsearch0-params",
                "list the available parameters of the selected line-search initialization methods");
    cmdline.add("", "list-lsearchk-params", "list the available parameters of the selected line-search strategies");
}

[[maybe_unused]] void setup_function(cmdline_t& cmdline)
{
    cmdline.add("", "function", "regex to select test functions", ".+");
}

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

auto process(const cmdline_t& cmdline, const int argc, const char* argv[])
{
    auto options = cmdline.process(argc, argv);

    const auto list_lsearch0   = options.has("list-lsearch0");
    const auto list_lsearchk   = options.has("list-lsearchk");
    const auto list_solver     = options.has("list-solver");
    const auto list_function   = options.has("list-function");
    const auto list_loss       = options.has("list-loss");
    const auto list_datasource = options.has("list-datasource");
    const auto list_generator  = options.has("list-generator");
    const auto list_splitter   = options.has("list-splitter");
    const auto list_tuner      = options.has("list-tuner");
    const auto list_version    = options.has("version");
    const auto list_git_hash   = options.has("git-hash");

    if (options.has("help"))
    {
        cmdline.usage();
        std::exit(EXIT_SUCCESS);
    }
    if (list_lsearch0)
    {
        std::cout << make_table("lsearch0", lsearch0_t::all(), options.get<string_t>("lsearch0"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_lsearchk)
    {
        std::cout << make_table("lsearchk", lsearchk_t::all(), options.get<string_t>("lsearchk"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_solver)
    {
        std::cout << make_table("solver", solver_t::all(), options.get<string_t>("solver"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_function)
    {
        std::cout << make_table("function", function_t::all(), options.get<string_t>("function"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_loss)
    {
        std::cout << make_table("loss", loss_t::all(), options.get<string_t>("loss"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_datasource)
    {
        std::cout << make_table("datasource", datasource_t::all(), options.get<string_t>("datasource"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_generator)
    {
        std::cout << make_table("generator", generator_t::all(), options.get<string_t>("generator"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_splitter)
    {
        std::cout << make_table("splitter", splitter_t::all(), options.get<string_t>("splitter"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_tuner)
    {
        std::cout << make_table("tuner", tuner_t::all(), options.get<string_t>("tuner"));
        std::exit(EXIT_SUCCESS);
    }
    if (list_version)
    {
        std::cout << nano::major_version << "." << nano::minor_version << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    if (list_git_hash)
    {
        std::cout << nano::git_commit_hash << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    return options;
}
}
