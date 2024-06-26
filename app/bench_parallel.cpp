#include <iomanip>
#include <nano/core/chrono.h>
#include <nano/core/cmdline.h>
#include <nano/core/parallel.h>
#include <nano/core/table.h>
#include <nano/main.h>
#include <nano/tensor.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nano;

namespace
{
struct exp_t
{
    static const char* name() { return "exp"; }

    template <class tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (-targets * outputs).exp().sum();
    }
};

struct log_t
{
    static const char* name() { return "log"; }

    template <class tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return ((-targets * outputs).exp() + 1).log().sum();
    }
};

struct mse_t
{
    static const char* name() { return "mse"; }

    template <class tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (targets - outputs).square().sum();
    }
};

template <class toperator>
scalar_t sti(const tensor_size_t i, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    assert(0 <= i && i < targets.rows());

    return toperator::get(targets.row(i).array(), outputs.row(i).array());
}

template <class toperator>
scalar_t reduce_st(const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t i = 0, size = targets.rows(); i < size; ++ i)
    {
        value += sti<toperator>(i, targets, outputs);
    }
    return value / static_cast<scalar_t>(targets.rows());
}

template <class toperator>
scalar_t reduce_mt(parallel::pool_t& pool, const matrix_t& targets, const matrix_t& outputs)
{
    auto values = make_full_vector<scalar_t>(static_cast<tensor_size_t>(pool.size()), 0);
    pool.map(targets.rows(), [&](tensor_size_t i, size_t t)
             { values(static_cast<tensor_size_t>(t)) += sti<toperator>(i, targets, outputs); });

    return values.sum() / static_cast<scalar_t>(targets.rows());
}

#if defined(_OPENMP)
template <class toperator>
scalar_t reduce_op(const matrix_t& targets, const matrix_t& outputs)
{
    auto       values = make_full_vector<scalar_t>(omp_get_max_threads(), 0);
    const auto size = targets.rows();

    #pragma omp parallel for
    for (tensor_size_t i = 0; i < size; ++ i)
    {
        const auto t = omp_get_thread_num();
        values(t) += sti<toperator>(i, targets, outputs);
    }

    return values.sum() / static_cast<scalar_t>(targets.rows());
}
#endif

bool close(const scalar_t v1, const scalar_t v2, const char* name, const scalar_t epsilon)
{
    if (std::fabs(v1 - v2) > epsilon)
    {
        std::cerr << "mis-matching sum (" << name << "): delta=" << std::fabs(v1 - v2) << ")!\n";
        return false;
    }
    else
    {
        return true;
    }
}

template <class toperator>
bool evaluate(const tensor_size_t min_size, const tensor_size_t max_size, table_t& table)
{
    parallel::pool_t pool;

    std::vector<scalar_t> single_deltas;
    std::vector<scalar_t> single_values;
    std::vector<matrix_t> single_targets;
    std::vector<matrix_t> single_outputs;

    // single-thread
    auto& row1 = table.append();
    row1 << scat("reduce-", toperator::name()) << "single";
    for (tensor_size_t size = min_size; size <= max_size; size *= 2)
    {
        auto targets = make_full_matrix<scalar_t>(size, 10, -1);
        auto outputs = make_random_matrix<scalar_t>(size, 10);
        for (tensor_size_t i = 0; i < size; ++ i)
        {
            targets(i, i % 10) = +1;
        }

        scalar_t value = 0;
        const auto delta = measure<nanoseconds_t>([&] { value = reduce_st<toperator>(targets, outputs); }, 16);
        row1 << "1.00";

        single_deltas.push_back(static_cast<scalar_t>(delta.count()));
        single_values.push_back(value);
        single_targets.push_back(targets);
        single_outputs.push_back(outputs);
    }

    // multi-threaded (using the thread pool)
    auto& row2 = table.append();
    row2 << scat("reduce-", toperator::name()) << scat("tpool(x", pool.size(), ")");
    for (size_t i = 0; i < single_deltas.size(); ++ i)
    {
        const auto deltaST = single_deltas[i];
        const auto valueST = single_values[i];
        const auto& targets = single_targets[i];
        const auto& outputs = single_outputs[i];

        scalar_t valueMT = 0;
        const auto deltaMT =
            measure<nanoseconds_t>([&] { valueMT = reduce_mt<toperator>(pool, targets, outputs); }, 16);
        row2 << scat(std::setprecision(2), std::fixed, deltaST / static_cast<double>(deltaMT.count()));
        if (!close(valueST, valueMT, "tpool", epsilon1<scalar_t>())) { return false; }
    }

#ifdef _OPENMP
    // multi-threaded (using OpenMP)
    auto& row3 = table.append();
    row3 << scat("reduce-", toperator::name()) << "openmp";
    for (size_t i = 0; i < single_deltas.size(); ++ i)
    {
        const auto deltaST = single_deltas[i];
        const auto valueST = single_values[i];
        const auto& targets = single_targets[i];
        const auto& outputs = single_outputs[i];

        scalar_t valueMT = 0;
        const auto deltaMT = measure<nanoseconds_t>([&] { valueMT = reduce_op<toperator>(targets, outputs); }, 16);
        row3 << scat(std::setprecision(2), std::fixed, deltaST / static_cast<double>(deltaMT.count()));
        if (!close(valueST, valueMT, "openmp", epsilon1<scalar_t>())) { return false; }
    }
#endif

    // OK
    return true;
}

int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark thread pool");
    cmdline.add("--min-size", "minimum problem size (in kilo)", 1);
    cmdline.add("--max-size", "maximum problem size (in kilo)", 1024);

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto kilo = tensor_size_t(1024), mega = kilo * kilo, giga = mega * kilo;
    const auto cmd_min_size = std::clamp(kilo * options.get<tensor_size_t>("--min-size"), kilo, mega);
    const auto cmd_max_size = std::clamp(kilo * options.get<tensor_size_t>("--max-size"), cmd_min_size, giga);

    table_t table;
    auto& header = table.header();
    header << "problem" << "method";
    for (auto size = cmd_min_size; size <= cmd_max_size; size *= 2)
    {
        header << scat(size / kilo, "K");
    }
    table.delim();

    // benchmark for different problem sizes and processing chunk sizes
    if (!evaluate<exp_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<log_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<mse_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }

    // print results
    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
