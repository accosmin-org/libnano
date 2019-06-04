#include <iostream>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/chrono.h>
#include <nano/logger.h>
#include <nano/tensor.h>
#include <nano/cmdline.h>

using namespace nano;

static scalar_t loss(const tensor_size_t i, const matrix_t& targets, const matrix_t& outputs)
{
    assert(targets.rows() == outputs.rows());
    assert(targets.cols() == outputs.cols());
    return ((-targets.row(i).array() * outputs.row(i).array()).exp() + 1).log().sum();
}

static scalar_t reduce_st(const matrix_t& targets, const matrix_t& outputs)
{
    scalar_t value = 0;
    for (tensor_size_t i = 0; i < targets.rows(); ++ i)
    {
        value += loss(i, targets, outputs);
    }
    return value;
}

template <int tchunk>
static scalar_t reduce_mt(const matrix_t& targets, const matrix_t& outputs)
{
    vector_t values = vector_t::Zero(tpool_t::instance().workers());
    nano::loopit<tchunk>(targets.rows(), [&] (const tensor_size_t i, const tensor_size_t t)
    {
        values(t) += loss(i, targets, outputs);
    });

    return values.sum();
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

    table_t table;
    auto& header = table.header();
    header << "function" << "st" << "mt" << "mt<32>" << "mt<64>" << "mt<128>" << "mt<256>" << "mt<512>" << "mt<1024>";
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

        scalar_t retST, retMT, retMT32, retMT64, retMT128, retMT256, retMT512, retMT1024;

        const auto deltaST = measure<nanoseconds_t>([&] { retST = reduce_st(targets, outputs); }, 16);
        const auto deltaMT = measure<nanoseconds_t>([&] { retMT = reduce_mt<-1>(targets, outputs); }, 16);
        const auto deltaMT32 = measure<nanoseconds_t>([&] { retMT32 = reduce_mt<32>(targets, outputs); }, 16);
        const auto deltaMT64 = measure<nanoseconds_t>([&] { retMT64 = reduce_mt<64>(targets, outputs); }, 16);
        const auto deltaMT128 = measure<nanoseconds_t>([&] { retMT128 = reduce_mt<128>(targets, outputs); }, 16);
        const auto deltaMT256 = measure<nanoseconds_t>([&] { retMT256 = reduce_mt<256>(targets, outputs); }, 16);
        const auto deltaMT512 = measure<nanoseconds_t>([&] { retMT512 = reduce_mt<512>(targets, outputs); }, 16);
        const auto deltaMT1024 = measure<nanoseconds_t>([&] { retMT1024 = reduce_mt<1024>(targets, outputs); }, 16);

        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT32.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT64.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT128.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT256.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT512.count());
        row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT1024.count());

        std::cout << "ST = " << retST << std::endl;
        std::cout << "MT = " << retMT
            << "/" << retMT32 << "/" << retMT64 << "/" << retMT128
            << "/" << retMT256 << "/" << retMT512 << "/" << retMT1024 << std::endl;

        int cnt = 0;
        for (const auto mt : std::vector<scalar_t>{retMT, retMT32, retMT64, retMT128, retMT256, retMT512, retMT1024})
        {
            if (std::fabs(retST - mt) > 1e-10)
            {
                std::cerr << "mis-matching sum: delta=" << std::fabs(retST - mt) << "), cnt = " << (++cnt) << "!" << std::endl;
//                return EXIT_FAILURE;
            }
        }
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
