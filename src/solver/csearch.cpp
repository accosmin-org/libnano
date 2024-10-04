#include <solver/bundle.h>
#include <solver/csearch.h>

using namespace nano;

template <>
nano::enum_map_t<csearch_status> nano::enum_string()
{
    return {
        {            csearch_status::failed,             "failed"},
        {         csearch_status::max_iters,          "max_iters"},
        {         csearch_status::converged,          "converged"},
        {         csearch_status::null_step,          "null step"},
        {      csearch_status::descent_step,       "descent step"},
        {csearch_status::cutting_plane_step, "cutting plane step"}
    };
}

csearch_t::point_t::point_t(const tensor_size_t dims)
    : m_y(dims)
    , m_gy(dims)
{
}

csearch_t::csearch_t(const function_t& function, const scalar_t m1, const scalar_t m2, const scalar_t m3,
                     const scalar_t m4, const scalar_t interpol, const scalar_t extrapol)
    : m_function(function)
    , m_m1(m1)
    , m_m2(m2)
    , m_m3(m3)
    , m_m4(m4)
    , m_interpol(interpol)
    , m_extrapol(extrapol)
    , m_point(function.size())
{
}

const csearch_t::point_t& csearch_t::search(bundle_t& bundle, const scalar_t miu, const tensor_size_t max_evals,
                                            const scalar_t epsilon, const logger_t& logger)
{
    constexpr auto level         = std::numeric_limits<scalar_t>::quiet_NaN();
    const auto     epsil_epsilon = epsilon * std::sqrt(static_cast<scalar_t>(m_point.m_y.size()));
    const auto     gnorm_epsilon = epsilon * std::sqrt(static_cast<scalar_t>(m_point.m_y.size()));

    auto& t = m_point.m_t;
    t       = 1.0;
    auto tL = 0.0;
    auto tR = std::numeric_limits<scalar_t>::infinity();

    const auto new_trial = [&]()
    {
        if (std::isfinite(tR))
        {
            return (1.0 - m_interpol) * tL + m_interpol * tR;
        }
        else
        {
            return t * m_extrapol;
        }
    };

    while (m_function.fcalls() + m_function.gcalls() < max_evals)
    {
        auto&       status = m_point.m_status;
        auto&       y      = m_point.m_y;
        auto&       gy     = m_point.m_gy;
        auto&       fy     = m_point.m_fy;
        const auto& x      = bundle.x();
        const auto  fx     = bundle.fx();

        // step (1) - get proximal point, compute statistics
        const auto& proxim = bundle.solve(t / miu, level, logger);

        y  = proxim.m_x;
        fy = m_function.vgrad(y, gy);

        const auto ghat  = (miu / t) * (x - y);
        const auto delta = fx - proxim.m_fhat + 0.5 * ghat.dot(y - x);
        const auto error = fx - fy + gy.dot(y - x);
        const auto epsil = fx - proxim.m_fhat + ghat.dot(y - x);
        const auto gnorm = ghat.lpNorm<2>();
        const auto econv = epsil <= epsil_epsilon;
        const auto gconv = gnorm <= gnorm_epsilon;

        logger.info("[csearch]: calls=", m_function.fcalls(), "|", m_function.gcalls(), ",fx=", fx, ",fy=", fy,
                    ",delta=", delta, ",error=", error, ",epsil=", epsil, ",gnorm=", gnorm, ",bsize=", bundle.size(),
                    ",miu=", miu, ",t=", t, "[", tL, ",", tR, "]\n");

        assert(proxim.m_fhat <= fx);
        assert(delta + epsilon1<scalar_t>() >= 0.0);
        assert(error + epsilon1<scalar_t>() >= 0.0);
        assert(epsil + epsilon1<scalar_t>() >= 0.0);

        // compute tests...
        const auto test_converged     = econv && gconv;                              // stopping criterion (35)
        const auto test_descent       = fy <= fx - m_m1 * delta;                     // descent test (31)
        const auto test_null_step     = error <= m_m3 * delta;                       // null-step test (33)
        const auto test_cutting_plane = gconv || (ghat.dot(y - x) >= -m_m4 * epsil); // cutting-plane test (36)
        const auto test_sufficient    = gy.dot(y - x) >= -m_m2 * delta;              // test (34)

        // step (1...) - curve search
        if (const auto failed = !std::isfinite(fy); failed)
        {
            status = csearch_status::failed;
            break;
        }
        else if (test_converged)
        {
            status = csearch_status::converged;
            break;
        }
        else if (test_descent)
        {
            // step (2)
            tL = t;
            // step (4)
            if (test_sufficient)
            {
                status = csearch_status::descent_step;
                break;
            }
            // step (5)
            if (!std::isfinite(tR) && test_cutting_plane)
            {
                status = csearch_status::cutting_plane_step;
                break;
            }
            // step (6)
            t = new_trial();
        }
        else
        {
            // step (3)
            tR = t;
            if (tL < epsilon0<scalar_t>() && test_null_step)
            {
                status = csearch_status::null_step;
                break;
            }
            // step (6)
            t = new_trial();
        }
    }

    logger.info("[csearch]: calls=", m_function.fcalls(), "|", m_function.gcalls(), ",fy=", m_point.m_fy,
                ",t=", m_point.m_t, ",status=", m_point.m_status, "\n");

    return m_point;
}

void csearch_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::m3"), 0, LT, 1.0, LT, 1e+6));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::m4"), 0, LT, 1.0, LT, 1e+6));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::interpol"), 0.0, LT, 0.3, LT, 1.0));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::extrapol"), 1.0, LT, 5.0, LT, 1e+2));
    c.register_parameter(parameter_t::make_scalar_pair(scat(prefix, "::csearch::m1m2"), 0, LT, 0.5, LT, 0.9, LT, 1.0));
}

csearch_t csearch_t::make(const function_t& function, const configurable_t& c, const string_t& prefix)
{
    const auto [m1, m2] = c.parameter(scat(prefix, "::csearch::m1m2")).value_pair<scalar_t>();
    const auto m3       = c.parameter(scat(prefix, "::csearch::m3")).value<scalar_t>();
    const auto m4       = c.parameter(scat(prefix, "::csearch::m4")).value<scalar_t>();
    const auto interpol = c.parameter(scat(prefix, "::csearch::interpol")).value<scalar_t>();
    const auto extrapol = c.parameter(scat(prefix, "::csearch::extrapol")).value<scalar_t>();

    return {function, m1, m2, m3, m4, interpol, extrapol};
}
