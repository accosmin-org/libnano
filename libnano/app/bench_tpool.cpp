#include <iostream>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/chrono.h>
#include <nano/logger.h>
#include <nano/tensor.h>
#include <nano/cmdline.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace nano;

struct exp_t
{
    static const char* name() { return "exp"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (-targets * outputs).exp().sum();
    }
};

struct log_t
{
    static const char* name() { return "log"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return ((-targets * outputs).exp() + 1).log().sum();
    }
};

struct mse_t
{
    static const char* name() { return "mse"; }

    template <typename tarray>
    static scalar_t get(const tarray& targets, const tarray& outputs)
    {
        return (targets - outputs).square().sum();
    }
};

template <typename toperator>
static scalar_t sti(const tensor_size_t i, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    assert(0 <= i && i < targets.rows());

    return toperator::get(targets.row(i).array(), outputs.row(i).array());
}

template <typename toperator>
static scalar_t reduce_st(const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t i = 0, size = targets.rows(); i < size; ++ i)
    {
        value += sti<toperator>(i, targets, outputs);
    }
    return value;
}

template <typename toperator>
static scalar_t reduce_mt(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(tpool_t::size());
    nano::loopi(targets.rows(), [&] (const tensor_size_t i, const tensor_size_t t)
    {
        values(t) += sti<toperator>(i, targets, outputs);
    });

    return values.sum();
}

#if defined(_OPENMP)
template <typename toperator>
static scalar_t reduce_op(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(omp_get_max_threads());
    const auto size = targets.rows();

    #pragma omp parallel for
    for (tensor_size_t i = 0; i < size; ++ i)
    {
        const auto t = omp_get_thread_num();
        values(t) += sti<toperator>(i, targets, outputs);
    }

    return values.sum();
}
#endif

static bool close(const scalar_t v1, const scalar_t v2, const char* name, const scalar_t epsilon)
{
    if (std::fabs(v1 - v2) > epsilon)
    {
        std::cerr << "mis-matching sum (" << name << "): delta=" << std::fabs(v1 - v2) << ")!" << std::endl;
        return false;
    }
    return true;
}

template <typename toperator>
static bool evaluate(const tensor_size_t min_size, const tensor_size_t max_size, table_t& table)
{
    for (tensor_size_t size = min_size; size <= max_size; size *= 2)
    {
        matrix_t targets = matrix_t::Constant(size, 10, -1);
        matrix_t outputs = matrix_t::Random(size, 10);
        for (tensor_size_t i = 0; i < size; ++ i)
        {
            targets(i, i % 10) = +1;
        }

        auto& row = table.append();
        const auto kilo = tensor_size_t(1<<10);
        row << strcat("reduce-", toperator::name(), "[", to_string(size / kilo), "K]");

        scalar_t retST, retMT;

        const auto deltaST = measure<nanoseconds_t>([&] { retST = reduce_st<toperator>(targets, outputs); }, 16);
        const auto deltaMT = measure<nanoseconds_t>([&] { retMT = reduce_mt<toperator>(targets, outputs); }, 16);
        #ifdef _OPENMP
        scalar_t retOP;
        const auto deltaOP = measure<nanoseconds_t>([&] { retOP = reduce_op<toperator>(targets, outputs); }, 16);
        #endif

        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT.count());
        #ifdef _OPENMP
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaOP.count());
        #endif

        if (!close(retST, retMT, "MT", epsilon1<scalar_t>() * size)) { return false; }
        #ifdef _OPENMP
        if (!close(retST, retOP, "OP", epsilon1<scalar_t>() * size)) { return false; }
        #endif
    }

    // OK
    return true;
}

static int unsafe_main(int argc, const char *argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark thread pool");
    cmdline.add("", "min-size",     "minimum problem size (in kilo)", 1);
    cmdline.add("", "max-size",     "maximum problem size (in kilo)", 1024);

    cmdline.process(argc, argv);

    if (cmdline.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto kilo = tensor_size_t(1<<10), mega = tensor_size_t(1<<20), giga = tensor_size_t(1<<30);
    const auto cmd_min_size = clamp(kilo * cmdline.get<tensor_size_t>("min-size"), kilo, mega);
    const auto cmd_max_size = clamp(kilo * cmdline.get<tensor_size_t>("max-size"), cmd_min_size, giga);

    table_t table;
    auto& header = table.header();
    header << "problem" << "single" << strcat("tpool(x", tpool_t::size(), ")");
    #ifdef _OPENMP
    header << "OpenMP";
    #endif
    table.delim();

    // benchmark for different problem sizes and processing chunk sizes
    if (!evaluate<exp_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<log_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }
    table.delim();
    if (!evaluate<mse_t>(cmd_min_size, cmd_max_size, table)) { return EXIT_FAILURE; }

    // print results
    table.mark(make_marker_maximum_percentage_cols<double>(5));
    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}

int main(int argc, const char* argv[])
{
    return nano::main(unsafe_main, argc, argv);
}
