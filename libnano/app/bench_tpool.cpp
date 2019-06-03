#include <iostream>
#include <nano/table.h>
#include <nano/tpool.h>
#include <nano/chrono.h>
#include <nano/logger.h>
#include <nano/cmdline.h>
#include <nano/numeric.h>

using namespace nano;

namespace
{
        template <typename tvector>
        void op(const size_t i, tvector& vector)
        {
                const auto x = static_cast<double>(i);
                vector[i] = std::sin(x) + std::cos(x);
        }

        template <typename tvector>
        void st_op(tvector& vector)
        {
                for (size_t i = 0; i < vector.size(); ++ i)
                {
                        op(i, vector);
                }
                (void)vector;
        }

        template <int tchunk, typename tvector>
        void mt_op(tvector& vector)
        {
                nano::loopi<tchunk>(vector.size(), [&] (const size_t i)
                {
                        op(i, vector);
                });
                (void)vector;
        }
}

static int unsafe_main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("benchmark thread pool");
        cmdline.add("", "min-size",     "minimum problem size (in kilo)", "1");
        cmdline.add("", "max-size",     "maximum problem size (in kilo)", "1024");

        cmdline.process(argc, argv);

        if (cmdline.has("help"))
        {
            cmdline.usage();
            return EXIT_SUCCESS;
        }

        // check arguments and options
        const auto kilo = size_t(1024);
        const auto cmd_min_size = clamp(kilo * cmdline.get<size_t>("min-size"), kilo, 1024 * kilo);
        const auto cmd_max_size = clamp(kilo * cmdline.get<size_t>("max-size"), cmd_min_size, 1024 * 1024 * kilo);

        table_t table;
        auto& header = table.header();
        header << "function" << "st" << "mt" << "mt<32>" << "mt<64>" << "mt<128>" << "mt<256>" << "mt<512>" << "mt<1024>";
        table.delim();

        // benchmark for different problem sizes and number of active workers
        for (size_t size = cmd_min_size; size <= cmd_max_size; size *= 2)
        {
                auto& row = table.append();
                row << ("sin+cos [" + to_string(size / kilo) + "K]");

                std::vector<double> vectorST(size);
                std::vector<double> vectorMT(size), vectorMT32(size), vectorMT64(size), vectorMT128(size), vectorMT256(size);

                const auto deltaST = measure<nanoseconds_t>([&] { st_op(vectorST); }, 16);
                const auto deltaMT = measure<nanoseconds_t>([&] { mt_op<-1>(vectorMT); }, 16);
                const auto deltaMT32 = measure<nanoseconds_t>([&] { mt_op<32>(vectorMT32); }, 16);
                const auto deltaMT64 = measure<nanoseconds_t>([&] { mt_op<64>(vectorMT64); }, 16);
                const auto deltaMT128 = measure<nanoseconds_t>([&] { mt_op<128>(vectorMT128); }, 16);
                const auto deltaMT256 = measure<nanoseconds_t>([&] { mt_op<256>(vectorMT256); }, 16);
                const auto deltaMT512 = measure<nanoseconds_t>([&] { mt_op<512>(vectorMT256); }, 16);
                const auto deltaMT1024 = measure<nanoseconds_t>([&] { mt_op<1024>(vectorMT256); }, 16);

                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaST.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT32.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT64.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT128.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT256.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT512.count());
                row << precision(2) << static_cast<double>(deltaST.count()) / static_cast<double>(deltaMT1024.count());

                for (size_t i = 0; i < size; ++ i)
                {
                        if (std::fabs(vectorST[i] - vectorMT[i]) > 1e-16)
                        {
                                std::cerr << "mis-matching vector (i=" << i
                                    << ",delta=" << std::fabs(vectorST[i] - vectorMT[i]) << ")!" << std::endl;
                                return EXIT_FAILURE;
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
