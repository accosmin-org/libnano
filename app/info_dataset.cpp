#include <nano/tensor/index.h>
#include <nano/imclass.h>
#include <nano/tabular.h>
#include <nano/util/table.h>
#include <nano/util/chrono.h>
#include <nano/util/logger.h>
#include <nano/util/cmdline.h>

using namespace nano;

template <typename tdataset>
static void header(table_t& table, const std::unique_ptr<tdataset>& dataset)
{
    table.append() << "folds" << colspan(3) << dataset->folds();
    table.append() << "samples" << colspan(3) << strcat(
        dataset->samples(), " = ",
        dataset->samples(fold_t{0U, protocol::train}), "+",
        dataset->samples(fold_t{0U, protocol::valid}), "+",
        dataset->samples(fold_t{0U, protocol::test}));
}

static void append(table_t& table, const char* type, const feature_t& feature)
{
    table.append() << type << feature.name()
        << (feature.discrete() ? strcat("discrete x", feature.labels().size()) : "continuous")
        << (feature.optional() ? "optional" : "not optional");
}

static int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("report statistics on datasets");
    cmdline.add("", "imclass",          "regex to select image classification datasets", ".+");
    cmdline.add("", "tabular",          "regex to select tabular datasets", ".+");

    cmdline.process(argc, argv);

    const auto has_imclass = cmdline.has("imclass");
    const auto has_tabular = cmdline.has("tabular");

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (!has_imclass &&
        !has_tabular)
    {
        cmdline.usage();
        return EXIT_FAILURE;
    }

    if (has_imclass)
    {
        // load image classification datasets
        for (const auto& id : imclass_dataset_t::all().ids(std::regex(cmdline.get<string_t>("imclass"))))
        {
            const auto start = nano::timer_t{};

            auto dataset = imclass_dataset_t::all().get(id);
            critical(!dataset, strcat("invalid dataset '", id, "'"));
            critical(!dataset->load(), strcat("failed to load dataset '", id, "'"));
            log_info() << ">>> loading done in " << start.elapsed() << ".";

            table_t table;
            header(table, dataset);
            table.delim();
            table.append() << "input" << colspan(3) << dataset->idim();
            table.append() << "target" << colspan(3) << strcat(dataset->tdim(), " (", dataset->tfeature().name(), ")");
            std::cout << table << std::endl;
        }
    }

    if (has_tabular)
    {
        // load tabular datasets
        for (const auto& id : tabular_dataset_t::all().ids(std::regex(cmdline.get<string_t>("tabular"))))
        {
            const auto start = nano::timer_t{};

            auto dataset = tabular_dataset_t::all().get(id);
            critical(!dataset, strcat("invalid dataset '", id, "'"));
            critical(!dataset->load(), strcat("failed to load dataset '", id, "'"));
            log_info() << ">>> loading done in " << start.elapsed() << ".";

            table_t table;
            header(table, dataset);
            table.delim();
            if (dataset->ifeatures() > 10U)
            {
                for (size_t size = size_t(5U), i = 0U; i < size; ++ i)
                {
                    append(table, "input", dataset->ifeature(i));
                }
                table.append() << "..." << "..." << "..." << "...";
                for (size_t size = dataset->ifeatures(), i = size - 5U; i < size; ++ i)
                {
                    append(table, "input", dataset->ifeature(i));
                }
            }
            else
            {
                for (size_t size = dataset->ifeatures(), i = 0U; i < size; ++ i)
                {
                    append(table, "input", dataset->ifeature(i));
                }
            }
            table.delim();
            append(table, "target", dataset->tfeature());
            std::cout << table << std::endl;
        }
    }

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
