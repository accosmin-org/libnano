#include <nano/machine/params.h>
#include <utest/utest.h>

using namespace nano;
using namespace nano::ml;

UTEST_BEGIN_MODULE()

UTEST_CASE(_default)
{
    const auto params = params_t{};
    UTEST_CHECK_NOTHROW(params.tuner());
    UTEST_CHECK_NOTHROW(params.solver());
    UTEST_CHECK_NOTHROW(params.splitter());
}

UTEST_CASE(set_tuner)
{
    const auto* const id    = "local-search";
    const auto        tuner = tuner_t::all().get(id);

    auto params = params_t{};
    UTEST_CHECK_NOTHROW(params.tuner(id));
    UTEST_CHECK_NOTHROW(params.tuner(*tuner));
    UTEST_CHECK_NOTHROW(params.tuner(tuner));

    UTEST_CHECK_THROW(params.tuner("what"), std::runtime_error);
    UTEST_CHECK_THROW(params.tuner(rtuner_t{}), std::runtime_error);

    UTEST_CHECK(&params.tuner() != &*tuner);
    UTEST_CHECK_EQUAL(params.tuner().type_id(), id);
}

UTEST_CASE(set_solver)
{
    const auto* const id     = "lbfgs";
    const auto        solver = solver_t::all().get(id);

    auto params = params_t{};
    UTEST_CHECK_NOTHROW(params.solver(id));
    UTEST_CHECK_NOTHROW(params.solver(*solver));
    UTEST_CHECK_NOTHROW(params.solver(solver));

    UTEST_CHECK_THROW(params.solver("what"), std::runtime_error);
    UTEST_CHECK_THROW(params.solver(rsolver_t{}), std::runtime_error);

    UTEST_CHECK(&params.solver() != &*solver);
    UTEST_CHECK_EQUAL(params.solver().type_id(), id);
}

UTEST_CASE(set_splitter)
{
    const auto* const id       = "random";
    const auto        splitter = splitter_t::all().get(id);

    auto params = params_t{};
    UTEST_CHECK_NOTHROW(params.splitter(id));
    UTEST_CHECK_NOTHROW(params.splitter(*splitter));
    UTEST_CHECK_NOTHROW(params.splitter(splitter));

    UTEST_CHECK_THROW(params.splitter("what"), std::runtime_error);
    UTEST_CHECK_THROW(params.splitter(rsplitter_t{}), std::runtime_error);

    UTEST_CHECK(&params.splitter() != &*splitter);
    UTEST_CHECK_EQUAL(params.splitter().type_id(), id);
}

UTEST_END_MODULE()
