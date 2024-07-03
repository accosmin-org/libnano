#include <nano/mlearn/config.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::ml;

UTEST_BEGIN_MODULE(test_mlearn_config)

UTEST_CASE(_default)
{
    const auto config = config_t{};

    UTEST_CHECK(!config.logger());
    UTEST_CHECK_NOTHROW(config.tuner());
    UTEST_CHECK_NOTHROW(config.solver());
    UTEST_CHECK_NOTHROW(config.splitter());
}

UTEST_CASE(set_logger)
{
    const auto logger0 = config_t::logger_t{};
    const auto loggerX = [&](const result_t&, const string_t&) {};

    auto config = config_t{};
    UTEST_CHECK(!config.logger());

    UTEST_CHECK_NOTHROW(config.logger(logger0));
    UTEST_CHECK(!config.logger());

    UTEST_CHECK_NOTHROW(config.logger(loggerX));
    UTEST_CHECK(config.logger());
}

UTEST_CASE(set_tuner)
{
    const auto* const id    = "local-search";
    const auto        tuner = tuner_t::all().get(id);

    auto config = config_t{};
    UTEST_CHECK_NOTHROW(config.tuner(id));
    UTEST_CHECK_NOTHROW(config.tuner(*tuner));
    UTEST_CHECK_NOTHROW(config.tuner(tuner));

    UTEST_CHECK_THROW(config.tuner("what"), std::runtime_error);
    UTEST_CHECK_THROW(config.tuner(rtuner_t{}), std::runtime_error);

    UTEST_CHECK(&config.tuner() != &*tuner);
    UTEST_CHECK_EQUAL(config.tuner().type_id(), id);
}

UTEST_CASE(set_solver)
{
    const auto* const id     = "lbfgs";
    const auto        solver = solver_t::all().get(id);

    auto config = config_t{};
    UTEST_CHECK_NOTHROW(config.solver(id));
    UTEST_CHECK_NOTHROW(config.solver(*solver));
    UTEST_CHECK_NOTHROW(config.solver(solver));

    UTEST_CHECK_THROW(config.solver("what"), std::runtime_error);
    UTEST_CHECK_THROW(config.solver(rsolver_t{}), std::runtime_error);

    UTEST_CHECK(&config.solver() != &*solver);
    UTEST_CHECK_EQUAL(config.solver().type_id(), id);
}

UTEST_CASE(set_splitter)
{
    const auto* const id       = "random";
    const auto        splitter = splitter_t::all().get(id);

    auto config = config_t{};
    UTEST_CHECK_NOTHROW(config.splitter(id));
    UTEST_CHECK_NOTHROW(config.splitter(*splitter));
    UTEST_CHECK_NOTHROW(config.splitter(splitter));

    UTEST_CHECK_THROW(config.splitter("what"), std::runtime_error);
    UTEST_CHECK_THROW(config.splitter(rsplitter_t{}), std::runtime_error);

    UTEST_CHECK(&config.splitter() != &*splitter);
    UTEST_CHECK_EQUAL(config.splitter().type_id(), id);
}

UTEST_END_MODULE()
