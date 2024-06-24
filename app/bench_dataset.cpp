#include <nano/core/chrono.h>
#include <nano/core/cmdline.h>
#include <nano/core/table.h>
#include <nano/dataset/iterator.h>
#include <nano/main.h>

using namespace nano;

namespace
{
auto benchmark_select(const string_t& generator_id, const dataset_t& dataset)
{
    const auto logger  = make_stdout_logger();
    const auto samples = arange(0, dataset.samples());

    auto timer    = ::nano::timer_t{};
    auto iterator = select_iterator_t{dataset};
    iterator.loop(samples,
                  [](const tensor_size_t feature, const size_t tnum, sclass_cmap_t values)
                  {
                      (void)feature;
                      (void)tnum;
                      (void)values;
                  });
    iterator.loop(samples,
                  [](const tensor_size_t feature, const size_t tnum, mclass_cmap_t values)
                  {
                      (void)feature;
                      (void)tnum;
                      (void)values;
                  });
    iterator.loop(samples,
                  [](const tensor_size_t feature, const size_t tnum, scalar_cmap_t values)
                  {
                      (void)feature;
                      (void)tnum;
                      (void)values;
                  });
    iterator.loop(samples,
                  [](const tensor_size_t feature, const size_t tnum, struct_cmap_t values)
                  {
                      (void)feature;
                      (void)tnum;
                      (void)values;
                  });

    logger.log(log_type::info, "generator[", generator_id, "]: feature selection in <", timer.elapsed(), ">.\n");
}

auto benchmark_flatten(const string_t& generator_id, const dataset_t& dataset)
{
    const auto samples = arange(0, dataset.samples());

    // vary the sample batch size
    auto table = table_t{};
    table.header() << "generator"
                   << "batch size"
                   << "build [time]"
                   << "flatten [time]"
                   << "targets [time]";
    table.delim();
    for (const auto batch : {10, 20, 50, 100, 200, 500, 1000, 2000, 5000})
    {
        auto& row = table.append();
        row << generator_id << batch;

        auto timer    = ::nano::timer_t{};
        auto iterator = flatten_iterator_t{dataset, samples};
        iterator.batch(batch);
        row << timer.elapsed();

        timer.reset();
        iterator.loop(
            [](const tensor_range_t range, const size_t tnum, tensor2d_cmap_t flatten)
            {
                (void)range;
                (void)tnum;
                (void)flatten;
            });
        row << timer.elapsed();

        timer.reset();
        iterator.loop(
            [](const tensor_range_t range, const size_t tnum, tensor4d_cmap_t targets)
            {
                (void)range;
                (void)tnum;
                (void)targets;
            });
        row << timer.elapsed();
    }
    std::cout << table;
}

auto benchmark(const string_t& generator_id, const dataset_t& dataset)
{
    const auto logger = make_stdout_logger();
    logger.log(log_type::info, "generator[", generator_id, "]: columns=", dataset.columns(),
               ",features=", dataset.features(), "\n");
    logger.log(log_type::info, "generator[", generator_id, "]: target=[", dataset.target(), "]\n");

    benchmark_select(generator_id, dataset);
    benchmark_flatten(generator_id, dataset);
}

auto benchmark(datasource_t& datasource, const strings_t& generator_ids)
{
    datasource.load();

    for (const auto& generator_id : generator_ids)
    {
        auto dataset = dataset_t{datasource};
        dataset.add(generator_t::all().get(generator_id));
        benchmark(generator_id, dataset);
    }
}

int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark loading datasets and generating features");
    cmdline.add("--datasource", "regex to select machine learning datasets", "mnist");
    cmdline.add("--generator", "regex to select feature generation methods", "identity.+");

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto dregex = std::regex(options.get<string_t>("--datasource"));
    const auto gregex = std::regex(options.get<string_t>("--generator"));

    // benchmark
    auto rconfig = cmdconfig_t{options};
    for (const auto& id : datasource_t::all().ids(dregex))
    {
        const auto rdatasource = datasource_t::all().get(id);
        critical(!rdatasource, "invalid data source (", id, ")!");

        rconfig.setup(*rdatasource);

        benchmark(*rdatasource, generator_t::all().ids(gregex));
    }

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
