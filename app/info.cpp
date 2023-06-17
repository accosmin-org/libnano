
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

int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("display the registered implementations by type and their parameters");

    cmdline.add("", "lsearch0", "regex to select line-search initialization methods", ".+");
    cmdline.add("", "lsearchk", "regex to select line-search strategies", ".+");
    cmdline.add("", "solver", "regex to select solvers", ".+");
    cmdline.add("", "function", "use this regex to select test functions", ".+");

    cmdline.add("", "loss", "regex to select loss functions", ".+");
    cmdline.add("", "tuner", "regex to select hyper-parameter tuning methods", ".+");
    cmdline.add("", "splitter", "regex to select train-validation splitting methods", ".+");
    cmdline.add("", "datasource", "regex to select machine learning datasets", ".+");
    cmdline.add("", "generator", "regex to select feature generation methods", ".+");

    cmdline.add("", "list-lsearch0", "list the available line-search initialization methods");
    cmdline.add("", "list-lsearchk", "list the available line-search strategies");
    cmdline.add("", "list-solver", "list the available solvers");
    cmdline.add("", "list-function", "list the available test functions");

    cmdline.add("", "list-loss", "list the available loss functions");
    cmdline.add("", "list-tuner", "list the available hyper-parameter tuning methods");
    cmdline.add("", "list-splitter", "list the available train-validation splitting methods");
    cmdline.add("", "list-datasource", "list the available machine learning datasets");
    cmdline.add("", "list-generator", "list the available feature generation methods");

    cmdline.add("", "list-lsearch0-params", "list the parameters of the selected line-search initialization methods");
    cmdline.add("", "list-lsearchk-params", "list the parameters of the selected line-search strategies");
    cmdline.add("", "list-solver-params", "list the parameters of the selected solvers");

    cmdline.add("", "list-loss-params", "list the parameters of the selected loss functions");
    cmdline.add("", "list-tuner-params", "list the parameters of the selected hyper-parameter tuning methods");
    cmdline.add("", "list-splitter-params", "list the parameters of the selected train-validation splitting methods");
    cmdline.add("", "list-datasource-params", "list the parameters of the selected machine learning datasets");
    cmdline.add("", "list-generator-params", "list the parameters of the selected feature generation methods");

    const auto options = cmdline.process(argc, argv);

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
    process_list<tuner_t>("tuner", options);
    process_list<splitter_t>("splitter", options);
    process_list<datasource_t>("datasource", options);
    process_list<generator_t>("generator", options);

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(::unsafe_main, argc, argv);
}
