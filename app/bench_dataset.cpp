#include <nano/dataset.h>
#include <nano/generator.h>
#include <nano/core/chrono.h>
#include <nano/core/logger.h>
#include <nano/core/cmdline.h>
#include <nano/core/factory_util.h>

using namespace nano;

static auto benchmark_select(const string_t& generator_id, const dataset_generator_t& generator)
{
    const auto samples = arange(0, generator.dataset().samples());

    auto iterator = select_iterator_t{generator};
    iterator.exec(execution::seq);

    auto timer = ::nano::timer_t{};
    iterator.loop(samples, [] (tensor_size_t feature, size_t tnum, sclass_cmap_t values)
    {
        (void)feature; (void)tnum; (void)values;
    });
    iterator.loop(samples, [] (tensor_size_t feature, size_t tnum, mclass_cmap_t values)
    {
        (void)feature; (void)tnum; (void)values;
    });
    iterator.loop(samples, [] (tensor_size_t feature, size_t tnum, scalar_cmap_t values)
    {
        (void)feature; (void)tnum; (void)values;
    });
    iterator.loop(samples, [] (tensor_size_t feature, size_t tnum, struct_cmap_t values)
    {
        (void)feature; (void)tnum; (void)values;
    });
    log_info() << "generator [" << generator_id << "] feature selection in <" << timer.elapsed() << ">.";
}

static auto benchmark_flatten(const string_t& generator_id, const dataset_generator_t& generator)
{
    const auto samples = arange(0, generator.dataset().samples());

    auto timer = ::nano::timer_t{};
    auto iterator = flatten_iterator_t{generator, samples, execution::seq, 128};
    log_info() << "generator [" << generator_id << "] built in <" << timer.elapsed() << ">.";

    iterator.loop([] (tensor_range_t range, size_t tnum, tensor2d_cmap_t flatten)
    {
        (void)range; (void)tnum; (void)flatten;
    });
    log_info() << "generator [" << generator_id << "] flatten in <" << timer.elapsed() << ">.";

    timer.reset();
    iterator.loop([] (tensor_range_t range, size_t tnum, tensor4d_cmap_t targets)
    {
        (void)range; (void)tnum; (void)targets;
    });
    log_info() << "generator [" << generator_id << "] targets in <" << timer.elapsed() << ">.";
}

static auto benchmark(const string_t& generator_id, const dataset_generator_t& generator)
{
    log_info() << "generator [" << generator_id << "]:";
    log_info() << "  target=[" << generator.target() << "]";
    log_info() << "  columns=" << generator.columns();
    log_info() << "  features=" << generator.features();

    benchmark_select(generator_id, generator);
    benchmark_flatten(generator_id, generator);
}

static auto benchmark(const string_t& dataset_id, const strings_t& generator_ids)
{
    const auto rdataset = dataset_t::all().get(dataset_id);
    critical(rdataset == nullptr, "invalid dataset (", dataset_id, ")!");

    auto& dataset = *rdataset;

    const auto timer = ::nano::timer_t{};
    dataset.load();
    const auto elapsed = timer.elapsed();
    log_info() << string_t(80, '=');
    log_info() << "dataset [" << dataset_id << "] loaded in <" << elapsed << ">.";
    log_info() << "  type=" << dataset.type();
    log_info() << "  samples=" << dataset.samples();
    log_info() << "  features=" << dataset.features();
    log_info() << string_t(80, '=');

    for (const auto& id : generator_ids)
    {
        auto generator = dataset_generator_t{dataset};
        generator.add(generator_t::all().get(id));
        benchmark(id, generator);
    }
}

static int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark loading datasets and generating features");
    cmdline.add("", "dataset",          "regex to select the datasets to benchmark", ".+");
    cmdline.add("", "generator",        "regex to select the feature generators to benchmark", ".+");
    cmdline.add("", "list-dataset",     "list the available datasets");
    cmdline.add("", "list-generator",   "list the available feature generators");

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (cmdline.has("list-dataset"))
    {
        std::cout << make_table("dataset", dataset_t::all());
        return EXIT_SUCCESS;
    }

    if (cmdline.has("list-generator"))
    {
        std::cout << make_table("generator", generator_t::all());
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto dregex = std::regex(cmdline.get<string_t>("dataset"));
    const auto gregex = std::regex(cmdline.get<string_t>("generator"));

    // benchmark
    for (const auto& id : dataset_t::all().ids(dregex))
    {
        benchmark(id, generator_t::all().ids(gregex));
    }

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
