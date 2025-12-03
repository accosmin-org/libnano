#include <fixture/solver.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

using namespace nano;
using namespace constraint;

namespace
{
rsolvers_t make_solvers()
{
    auto solvers = rsolvers_t{};
    for (const auto gamma : {1.0, 2.0, 3.0})
    {
        auto solver                             = make_solver("ipm");
        solver->parameter("solver::ipm::gamma") = gamma;
        solver->parameter("solver::max_evals")  = 100;
        solvers.emplace_back(std::move(solver));
    }
    return solvers;
}
} // namespace

UTEST_BEGIN_MODULE()

UTEST_CASE(constrain)
{
    const auto Q = matrix_t{matrix_t::zero(3, 3)};
    const auto c = vector_t::zero(3);
    const auto a = vector_t::zero(3);
    const auto b = vector_t::zero(2);
    const auto A = matrix_t::zero(2, 3);

    auto function = quadratic_program_t{"qp", Q, c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(A * function.variable() >= b);
    UTEST_REQUIRE(A * function.variable() <= b);
    UTEST_REQUIRE(a * function.variable() == 1.0);
    UTEST_REQUIRE(a * function.variable() >= 1.0);
    UTEST_REQUIRE(a * function.variable() <= 1.0);
    UTEST_REQUIRE(function.variable() >= 1.0);
    UTEST_REQUIRE(function.variable() <= 1.0);
    UTEST_REQUIRE(!function.constrain(functional_equality_t{function}));
    UTEST_REQUIRE(!function.constrain(functional_inequality_t{function}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_equality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_inequality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_equality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_inequality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
}

UTEST_CASE(program1)
{
    // see example 16.2, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(6, 2, 1, 5, 2, 4);
    const auto c = make_vector<scalar_t>(-8, -3, -3);
    const auto A = make_matrix<scalar_t>(2, 1, 0, 1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(3, 0);
    const auto x = make_vector<scalar_t>(2, -1, 1);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program2)
{
    // see example p.467, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(0, 2);
    const auto G = -matrix_t::identity(2, 2);
    const auto h = vector_t::zero(2);
    const auto x = make_vector<scalar_t>(0, 0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
    check_minimize(make_solvers(), function, make_vector<scalar_t>(0.1086039277146398, -0.5283505579626659));
    check_minimize(make_solvers(), function, make_vector<scalar_t>(-0.1403887120993625, 0.7972989463671512));
}

UTEST_CASE(program3)
{
    // see example 16.4, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-2, -5);
    const auto G = make_matrix<scalar_t>(5, -1, 2, 1, 2, 1, -2, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(2, 6, 2, 0, 0);
    const auto x = make_vector<scalar_t>(1.4, 1.7);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program4)
{
    // see exercise 16.1a, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(8, 2, 2);
    const auto c = make_vector<scalar_t>(2, 3);
    const auto G = make_matrix<scalar_t>(3, -1, 1, 1, 1, 1, 0);
    const auto h = make_vector<scalar_t>(0, 4, 3);
    const auto x = make_vector<scalar_t>(1.0 / 6.0, -5.0 / 3.0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program5)
{
    // see exercise 16.11, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, -2, 4);
    const auto c = make_vector<scalar_t>(-2, -6);
    const auto G = make_matrix<scalar_t>(4, 0.5, 0.5, -1, 2, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(1, 2, 0, 0);
    const auto x = make_vector<scalar_t>(0.8, 1.2);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(program6)
{
    // see exercise 16.17, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto q = make_vector<scalar_t>(2, 0, 2);
    const auto c = make_vector<scalar_t>(-6, -4);
    const auto G = make_matrix<scalar_t>(3, 1, 1, -1, 0, 0, -1);
    const auto h = make_vector<scalar_t>(3, 0, 0);
    const auto x = make_vector<scalar_t>(2.0, 1.0);

    auto function = quadratic_program_t{"qp", q, c};
    UTEST_REQUIRE(G * function.variable() <= h);
    UTEST_REQUIRE(function.optimum(x));

    check_function(function);
    check_minimize(make_solvers(), function);
}

UTEST_CASE(bundle_cases)
{
    // NB: quadratic programs generated by bundle methods,
    //     that are badly conditioned and hard to solve!

    // clang-format off
    const auto Q = make_matrix<scalar_t>(5,
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0);

    const auto c0 = make_vector<scalar_t>(0, 0, 0, 0, 625.0);
    const auto G0 = make_matrix<scalar_t>(3,
        -0.00014353301163777320648, -8.2849464293226782207e-05, 0.00015109222548321000752, 3.7447177848252078335e-05,
-1, -2.5466140562675764974e-06, -1.4699448434609828959e-06, 2.680732333404427623e-06, 6.6440122998724819222e-07, -1,
        -1.0682720741105252288e-06, -6.1662308934775684817e-07, 1.1245329785614741871e-06, 2.78707830993565414e-07, -1);
    const auto h0 = make_vector<scalar_t>(
        4.8529937564564530991e-06,
        3.2332731948977429066e-09,
        0);

    const auto c1 = make_vector<scalar_t>(0, 0, 0, 0, 6550.5901686479783166);
    const auto G1 = make_matrix<scalar_t>(2,
        9.8983231668534294088e-09, 7.3781561103856015495e-07, -2.457903178239621485e-06, 1.2768656355090551211e-06, -1,
        3.0291065158146719688e-09, 2.2578794783653675608e-07, -7.5217290918316140752e-07, 3.9074921591692585722e-07,
-1); const auto h1 = make_vector<scalar_t>( 3.2748492557082926398e-09, 0);

    const auto c2 = make_vector<scalar_t>(0, 0, 0, 0, 100.0);
    const auto G2 = make_matrix<scalar_t>(2,
        0.012945828710536660261, 0.012945828710536658526, 0.01294582871053666373, 0.0129458287105366672, -1,
        -999999.9926269260468, -999999.9926269260468, -999999.9926269260468, -999999.9926269260468, -1);
    const auto h2 = make_vector<scalar_t>(
        0,
        6.2111205068049457623e-11);

    const auto c3 = make_vector<scalar_t>(0, 0, 0, 0, 1.1891869117837732522e-08);
    const auto G3 = make_matrix<scalar_t>(3,
        -1186635.9607374120969, 1034615.2219196240185, -1037609.0778432264924, 1074810.3402491894085, -1,
        -1034787.050578787108, -1012282.5866353140445, 1009671.8421346789692, -1006163.5156296341447, -1,
        1452009.5900903099682, 1015214.1063532972476, -1016596.6736856097123, 1011673.505992121296, -1);
    const auto h3 = make_vector<scalar_t>(
        73260.788020616397262,
        0,
        245177.65755747375078);

    const auto c4 = make_vector<scalar_t>(0, 0, 0, 0, 4.5047256130523651577e-06);
    const auto G4 = make_matrix<scalar_t>(6,
        103.18255965398196849, -104.68626533140547963, -101.40250221092107097, -100.32209609517937565, -1,
        -101.26673725830967498, 102.72551340085949789, -101.37037438044797, -102.41852928317091198, -1,
        -100.03644882491296642, -100.07627390238310738, 100.07856682263320636, 100.15955025015588831, -1,
        100.31673282845804351, 100.2726224557913639, -99.991043735714484342, 100.23360423845207379, -1,
        -100.02722612715126616, -100.06864756887910062, -100.10942081806923909, -99.988238095707103525, -1,
        100.01861621482468934, -100.0228178758262203, -100.02844709757481212, 100.02737081239307315, -1);
    const auto h4 = make_vector<scalar_t>(
        0.19769215956472097062,
        0.19666148807061800685,
        0.094300662946758873062,
        0.072627651761172584699,
        0.040246099662433532096,
        0);

    // clang-format on

    for (const auto& [c, G, h] : {std::make_tuple(c0, G0, h0), std::make_tuple(c1, G1, h1), std::make_tuple(c2, G2, h2),
                                  std::make_tuple(c3, G3, h3), std::make_tuple(c4, G4, h4)})
    {
        static auto index    = 0;
        auto        function = quadratic_program_t{scat("qp-bundle-case", index++), Q, c};
        UTEST_REQUIRE(G * function.variable() <= h);

        check_function(function);
        check_minimize(make_solvers(), function);
    }
}

UTEST_CASE(bundle_cases_with_level)
{
    // NB: quadratic programs generated by bundle methods with additional level constraint,
    //     that are badly conditioned and hard to solve!

    // clang-format off
    const auto Q1 = make_matrix<scalar_t>(5,
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0);
    const auto c1 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto G1 = make_matrix<scalar_t>(1,
        5.0712765698903083944, 10.060198726343752895, 13.516963149558435475, 15.89047373074238223, -1);
    const auto h1 = make_vector<scalar_t>(-4.4192633716574407643);
    const auto w1 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto l1 = 4.5;

    const auto Q2 = make_matrix<scalar_t>(5,
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0);
    const auto c2 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto G2 = make_matrix<scalar_t>(1,
        -9.583957630012883, -24.76184946994476, -34.83609461286843, -46.83363016420555, -1);
    const auto h2 = make_vector<scalar_t>(-13.15910824050277);
    const auto w2 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto l2 = 20.5;

    const auto Q3 = make_matrix<scalar_t>(5,
        1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 0);
    const auto c3 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto G3 = make_matrix<scalar_t>(1,
        5911.054370218572, 2029.910659278819, 1956.336168695624, 7424.608906364497, -1);
    const auto h3 = make_vector<scalar_t>(-4900.490855535085);
    const auto w3 = make_vector<scalar_t>(0, 0, 0, 0, 1);
    const auto l3 = 900.0981711070167;
    // clang-format on

    for (const auto& [Q, c, G, h, w, l] :
         {std::make_tuple(Q1, c1, G1, h1, w1, l1), std::make_tuple(Q2, c2, G2, h2, w2, l2),
          std::make_tuple(Q3, c3, G3, h3, w3, l3)})
    {
        static auto index    = 0;
        auto        function = quadratic_program_t{scat("qp-bundle-level-case", index++), Q, c};
        UTEST_REQUIRE(G * function.variable() <= h);
        UTEST_REQUIRE(w * function.variable() <= l);

        check_function(function);
        check_minimize(make_solvers(), function);

        if (index == 1)
        {
            const auto x0 = make_vector<scalar_t>(0.3720481659153125, 0.8746846640195591, -0.5662730895148255,
                                                  0.08623337083228955, 0.9036251941038855);
            check_minimize(make_solvers(), function, x0);
        }
    }
}

UTEST_CASE(factory)
{
    for (const auto& function : function_t::make({2, 16, function_type::quadratic_program}))
    {
        check_function(*function);
        check_minimize(make_solvers(), *function);
    }
}

UTEST_CASE(regression1)
{
    const auto function =
        make_function("osqp2", "function::seed", 266, "function::osqp2::neqs", 0.9, "function::osqp2::alpha", 1e-2);

    const auto x0 =
        make_vector<scalar_t>(0.9758460027831883, -0.6445622522158582, -0.3456228489243688, 0.5118898178500717,
                              0.4041865905542412, 0.07801044900083798, -0.4686052325467003, 0.9663423575357488,
                              0.5703720988434413, 0.6801780981467593, -0.1765819566623654, -0.306732374952134,
                              0.8287095558315107, -0.9020445986675114, 0.2456373809086276, -0.5490883124330687);

    check_minimize(make_solvers(), *(function->make(16)), x0);
}

UTEST_CASE(regression2)
{
    const auto function =
        make_function("osqp1", "function::seed", 3849, "function::osqp1::nineqs", 20, "function::osqp1::alpha", 1e-2);

    const auto x0 =
        make_vector<scalar_t>(0.668425765576367, 0.2905673792850159, -0.7024952794498051, 0.3922410585264389,
                              0.4161605003320679, -0.7584247358478521, -0.3616094192973631, 0.3970577010445298,
                              -0.9379886737694388, -0.5653949271948902, -0.6765821552442182, 0.8572741454280879,
                              0.3979742434610472, 0.7835358960256542, 0.7904647349393179, -0.9306335171900869);

    check_minimize(make_solvers(), *(function->make(16)), x0);
}

UTEST_CASE(regression3)
{
    const auto function =
        make_function("osqp1", "function::seed", 8663, "function::osqp1::nineqs", 20, "function::osqp1::alpha", 1e-2);

    const auto x0 =
        make_vector<scalar_t>(-0.5946316878653621, 0.3467531147620639, -0.6216705703153838, 0.2428857778681006,
                              -0.6594793298540012, 0.2422560983532791, -0.360785629815527, -0.2751850675456408,
                              -0.01142828063545787, 0.1682555665468959, -0.2834998925973646, 0.7312248223012749,
                              -0.8557768372632004, -0.850251526566133, 0.1442914878897066, -0.3039492051089099);

    check_minimize(make_solvers(), *(function->make(16)), x0);
}

UTEST_CASE(regression4)
{
    const auto function =
        make_function("osqp1", "function::seed", 1268, "function::osqp1::nineqs", 10, "function::osqp1::alpha", 1e-2);

    // NOLINTBEGIN(modernize-use-std-numbers)
    const auto x0 =
        make_vector<scalar_t>(0.5295057438431254, 0.6935502761504575, 0.2589574398151886, 0.6372639138602401);
    // NOLINTEND(modernize-use-std-numbers)

    check_minimize(make_solvers(), *(function->make(4)), x0);
}

UTEST_END_MODULE()
