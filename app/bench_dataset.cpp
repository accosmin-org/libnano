#include "util.h"

using namespace nano;

namespace
{
auto benchmark_select(const string_t& generator_id, const dataset_t& dataset)
{
    const auto samples = arange(0, dataset.samples());

    auto iterator = select_iterator_t{dataset};

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
    log_info() << "generator_t [" << generator_id << "] feature selection in <" << timer.elapsed() << ">.";
}

auto benchmark_flatten(const string_t& generator_id, const dataset_t& dataset)
{
    const auto samples = arange(0, dataset.samples());

    // TODO: benchmark batch size

    auto timer = ::nano::timer_t{};
    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(128);
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

auto benchmark(const string_t& generator_id, const dataset_t& dataset)
{
    log_info() << "generator [" << generator_id << "]:";
    log_info() << "  target=[" << dataset.target() << "]";
    log_info() << "  columns=" << dataset.columns();
    log_info() << "  features=" << dataset.features();

    benchmark_select(generator_id, dataset);
    benchmark_flatten(generator_id, dataset);
}

auto benchmark(const string_t& datasource_id, const strings_t& generator_ids)
{
    const auto rdatasource = datasource_t::all().get(datasource_id);
    critical(!rdatasource, "invalid data source (", datasource_id, ")!");

    auto& datasource = *rdatasource;
    ::load_datasource(datasource);

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
    cmdline.add("", "datasource", "regex to select machine learning datasets", "mnist");
    cmdline.add("", "generator", "regex to select feature generation methods", "identity.+");
    cmdline.add("", "list-datasource", "list the available machine learning datasets");
    cmdline.add("", "list-generator", "list the available feature generation methods");

    const auto options = ::process(cmdline, argc, argv);

    // check arguments and options
    const auto dregex = std::regex(options.get<string_t>("datasource"));
    const auto gregex = std::regex(options.get<string_t>("generator"));

    // benchmark
    for (const auto& id : datasource_t::all().ids(dregex))
    {
        benchmark(id, generator_t::all().ids(gregex));
    }

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
