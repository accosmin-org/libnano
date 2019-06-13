#include <fstream>
#include <iostream>
#include <nano/table.h>
#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/solver.h>
#include <nano/version.h>

using namespace nano;

namespace
{
    template <typename tobject>
    void print(const string_t& name, const factory_t<tobject>& factory, const string_t& regex,
        const bool as_table, const bool as_json)
    {
        if (as_table)
        {
            table_t table;
            table.header() << name << "description";
            table.delim();
            for (const auto& id : factory.ids(std::regex(regex)))
            {
                table.append() << id << factory.description(id);
            }
            std::cout << table;
        }

        if (as_json)
        {
            for (const auto& id : factory.ids(std::regex(regex)))
            {
                const auto json = factory.get(id)->config_with_id(id);
                std::cout << json.dump(4) << std::endl;
            }
        }
    }
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("display the registered objects");
    cmdline.add("", "lsearch0",             "regex to select the line-search initialization methods", ".+");
    cmdline.add("", "lsearchk",             "regex to select the line-search strategies", ".+");
    cmdline.add("", "solver",               "regex to select the numerical optimization methods", ".+");
    cmdline.add("", "as-table",             "display the selected objects in a table");
    cmdline.add("", "as-json",              "display the default configuration for the selected objects as JSON");
    cmdline.add("", "version",              "library version");
    cmdline.add("", "git-hash",             "git commit hash");

    cmdline.process(argc, argv);

    const auto has_lsearch0 = cmdline.has("lsearch0");
    const auto has_lsearchk = cmdline.has("lsearchk");
    const auto has_solver = cmdline.has("solver");
    const auto has_as_table = cmdline.has("as-table");
    const auto has_as_json = cmdline.has("as-json");
    const auto has_version = cmdline.has("version");
    const auto has_git_hash = cmdline.has("git-hash");

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (!has_lsearch0 &&
        !has_lsearchk &&
        !has_solver &&
        !has_version &&
        !has_git_hash)
    {
        cmdline.usage();
        return EXIT_FAILURE;
    }

    // check arguments and options
    if (has_lsearch0)
    {
        print("lsearch0", lsearch0_t::all(), cmdline.get<string_t>("lsearch0"), has_as_table, has_as_json);
    }
    if (has_lsearchk)
    {
        print("lsearchk", lsearchk_t::all(), cmdline.get<string_t>("lsearchk"), has_as_table, has_as_json);
    }
    if (has_solver)
    {
        print("solver", solver_t::all(), cmdline.get<string_t>("solver"), has_as_table, has_as_json);
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
    return nano::main(unsafe_main, argc, argv);
}
