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

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("display the registered objects");
    cmdline.add("", "lsearch0", "regex to select line-search initialization methods", ".+");
    cmdline.add("", "lsearchk", "regex to select line-search strategies", ".+");
    cmdline.add("", "solver", "regex to select solvers", ".+");
    cmdline.add("", "loss", "regex to select loss functions", ".+");
    cmdline.add("", "datasource", "regex to select machine learning datasets", ".+");
    cmdline.add("", "generator", "regex to select feature generation methods", ".+");
    cmdline.add("", "splitter", "regex to select train-validation splitting methods", ".+");
    cmdline.add("", "tuner", "regex to select hyper-parameter tuning methods", ".+");
    cmdline.add("", "list-lsearch0", "list the available line-search initialization methods");
    cmdline.add("", "list-lsearchk", "list the available line-search strategies");
    cmdline.add("", "list-solver", "list the available solvers");
    cmdline.add("", "list-loss", "list the available loss functions");
    cmdline.add("", "list-datasource", "list the available machine learning datasets");
    cmdline.add("", "list-generator", "list the available feature generation methods");
    cmdline.add("", "list-splitter", "list the available train-validation splitting methods");
    cmdline.add("", "list-tuner", "list the available hyper-parameter tuning methods");
    cmdline.add("", "version", "library version");
    cmdline.add("", "git-hash", "git commit hash");

    const auto options = cmdline.process(argc, argv);

    const auto has_lsearch0   = options.has("list-lsearch0");
    const auto has_lsearchk   = options.has("list-lsearchk");
    const auto has_loss       = options.has("list-loss");
    const auto has_solver     = options.has("list-solver");
    const auto has_datasource = options.has("list-datasource");
    const auto has_generator  = options.has("list-generator");
    const auto has_splitter   = options.has("list-splitter");
    const auto has_tuner      = options.has("list-tuner");
    const auto has_version    = options.has("version");
    const auto has_git_hash   = options.has("git-hash");

    if (options.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (!has_lsearch0 && !has_lsearchk && !has_solver && !has_loss && !has_datasource && !has_generator &&
        !has_splitter && !has_tuner && !has_version && !has_git_hash)
    {
        cmdline.usage();
        return EXIT_FAILURE;
    }

    // check arguments and options
    if (has_lsearch0)
    {
        std::cout << make_table("lsearch0", lsearch0_t::all(), options.get<string_t>("lsearch0"));
    }
    if (has_lsearchk)
    {
        std::cout << make_table("lsearchk", lsearchk_t::all(), options.get<string_t>("lsearchk"));
    }
    if (has_solver)
    {
        std::cout << make_table("solver", solver_t::all(), options.get<string_t>("solver"));
    }
    if (has_loss)
    {
        std::cout << make_table("loss", loss_t::all(), options.get<string_t>("loss"));
    }
    if (has_datasource)
    {
        std::cout << make_table("datasource", datasource_t::all(), options.get<string_t>("datasource"));
    }
    if (has_generator)
    {
        std::cout << make_table("generator", generator_t::all(), options.get<string_t>("generator"));
    }
    if (has_splitter)
    {
        std::cout << make_table("splitter", splitter_t::all(), options.get<string_t>("splitter"));
    }
    if (has_tuner)
    {
        std::cout << make_table("tuner", tuner_t::all(), options.get<string_t>("tuner"));
    }
    if (has_version)
    {
        std::cout << nano::major_version << "." << nano::minor_version << std::endl;
    }
    if (has_git_hash)
    {
        std::cout << nano::git_commit_hash << std::endl;
    }

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
