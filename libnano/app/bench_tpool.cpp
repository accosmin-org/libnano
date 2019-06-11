#include <iostream>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/chrono.h>
#include <nano/logger.h>
#include <nano/tensor.h>
#include <nano/cmdline.h>

using namespace nano;

template <typename tarray>
static scalar_t expa(const tarray& targets, const tarray& outputs)
{
    return ((-targets * outputs).exp() + 1).log().sum();
}

static scalar_t expi(const tensor_size_t i, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    assert(0 <= i && i < targets.rows());

    return expa(
        targets.row(i).array(),
        outputs.row(i).array());
}

static scalar_t expr(const tensor_size_t begin, const tensor_size_t end, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    assert(0 <= begin && begin < end && end <= targets.rows());

    return expa(
        targets.block(begin, 0, end - begin, targets.cols()).array(),
        outputs.block(begin, 0, end - begin, outputs.cols()).array());
}

static scalar_t reduce_st(const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t i = 0; i < targets.rows(); ++ i)
    {
        value += expi(i, targets, outputs);
    }
    return value;
}

static scalar_t reduce_st(const tensor_size_t chunk, const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t begin = 0; begin < targets.rows(); begin += chunk)
    {
        value += expr(begin, std::min(begin + chunk, targets.rows()), targets, outputs);
    }
    return value;
}

static scalar_t reduce_mt(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(tpool_t::instance().workers());
    nano::loopit(targets.rows(), [&] (const tensor_size_t i, const tensor_size_t t)
    {
        values(t) += expi(i, targets, outputs);
    });

    return values.sum();
}

static bool close(const scalar_t v1, const scalar_t v2, const char* name, const scalar_t epsilon)
{
    if (std::fabs(v1 - v2) > epsilon)
    {
        std::cerr << "mis-matching sum (" << name << "): delta=" << std::fabs(v1 - v2) << ")!" << std::endl;
        return false;
    }
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
    const auto kilo = tensor_size_t(1024);
    const auto cmd_min_size = clamp(kilo * cmdline.get<tensor_size_t>("min-size"), kilo, 1024 * kilo);
    const auto cmd_max_size = clamp(kilo * cmdline.get<tensor_size_t>("max-size"), cmd_min_size, 1024 * 1024 * kilo);

    //todo: serial - parallel
    //todo: index|range - index|range
    //todo: exponential - 1K-1M
    //todo: logistic - 1K-1M
    //todo: square - 1K-1M

    table_t table;
    auto& header = table.header();
    header << "problem" << "1thread" << "st1" << "st2" << "st4" << "st8" << "st16"
        << strcat(tpool_t::instance().workers(), "threads");
    table.delim();

    // benchmark for different problem sizes and processing chunk sizes
    for (tensor_size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
    {
        matrix_t targets = matrix_t::Constant(size, 10, -1);
        matrix_t outputs = matrix_t::Random(size, 10);
        for (tensor_size_t i = 0; i < size; ++ i)
        {
            targets(i, i % 10) = +1;
        }

        auto& row = table.append();
        row << ("reduce[" + to_string(size / kilo) + "K]");

        scalar_t retST, retST1, retST2, retST4, retST8, retST16, retMT;

        const auto deltaST = measure<nanoseconds_t>([&] { retST = reduce_st(targets, outputs); }, 16);
        const auto deltaST1 = measure<nanoseconds_t>([&] { retST1 = reduce_st(1, targets, outputs); }, 16);
        const auto deltaST2 = measure<nanoseconds_t>([&] { retST2 = reduce_st(2, targets, outputs); }, 16);
        const auto deltaST4 = measure<nanoseconds_t>([&] { retST4 = reduce_st(4, targets, outputs); }, 16);
        const auto deltaST8 = measure<nanoseconds_t>([&] { retST8 = reduce_st(8, targets, outputs); }, 16);
        const auto deltaST16 = measure<nanoseconds_t>([&] { retST16 = reduce_st(16, targets, outputs); }, 16);
        const auto deltaMT = measure<nanoseconds_t>([&] { retMT = reduce_mt(targets, outputs); }, 16);

        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST1.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST2.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST4.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST8.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST16.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT.count());

        if (!close(retST, retMT, "MT", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retST1, "ST1", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retST2, "ST2", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retST4, "ST4", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retST8, "ST8", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retST16, "ST16", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
        if (!close(retST, retMT, "MT", epsilon2<scalar_t>() * size)) { return EXIT_FAILURE; }
    }

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
