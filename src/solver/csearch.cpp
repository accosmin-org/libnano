#include <nano/solver/bundle.h>
#include <nano/solver/csearch.h>

#include <iomanip>
#include <iostream>

using namespace nano;

namespace
{
template <typename tvector>
bool converged(const scalar_t delta, const scalar_t error, const tvector& sgrad, const scalar_t etol,
               const scalar_t stol)
{
    return std::max({delta, error}) < etol && sgrad.template lpNorm<2>() < stol;
}
} // namespace

csearch_t::csearch_t(const function_t& function, const scalar_t m1, const scalar_t m2, const scalar_t m3,
                     const scalar_t m4)
    : m_function(function)
    , m_m1(m1)
    , m_m2(m2)
    , m_m3(m3)
    , m_m4(m4)
{
    m_point.m_y.resize(function.size());
    m_point.m_gy.resize(function.size());
}

const csearch_t::point_t& csearch_t::search(bundle_t& bundle, const scalar_t miu, const tensor_size_t max_evals,
                                            const scalar_t epsilon)
{
    // FIXME: the references do not specify how to choose these thresholds
    const auto etol = epsilon * std::sqrt(static_cast<scalar_t>(m_function.size()));
    const auto stol = epsilon * std::sqrt(static_cast<scalar_t>(m_function.size())) * 1e+2;

    auto& t = m_point.m_t;
    t       = 1.0;
    auto tL = 0.0;
    auto tR = std::numeric_limits<scalar_t>::infinity();

    const auto new_trial = [&]()
    {
        if (std::isfinite(tR))
        {
            return 0.5 * (tL + tR); // TODO: better than bisection?!
        }
        else
        {
            return t * 10.0; // TODO: customizable
        }
    };

    while (m_function.fcalls() + m_function.gcalls() < max_evals)
    {
        auto& status = m_point.m_status;
        auto& y      = m_point.m_y;
        auto& gy     = m_point.m_gy;
        auto& fy     = m_point.m_fy;

        // estimate proximal point
        bundle.solve(miu / t);

        y  = bundle.proximal(miu / t);
        fy = m_function.vgrad(y, gy);

        const auto& x     = bundle.x();
        const auto  fx    = bundle.fx();
        const auto  e     = bundle.smeared_e();
        const auto  s     = bundle.smeared_s();
        const auto  delta = bundle.delta(miu / t);

        std::cout << std::fixed << std::setprecision(9) << "calls=" << m_function.fcalls() << "|" << m_function.gcalls()
                  << ",fx=" << fx << ",fy=" << fy << ",de=" << e << ",ds=" << s.lpNorm<2>() << ",dd=" << delta
                  << ",bsize=" << bundle.size() << ",miu=" << miu << ",t=" << t << "[" << tL << "," << tR << "]"
                  << std::endl;

        if (const auto failed = !std::isfinite(fy); failed)
        {
            status = csearch_status::failed;
            break;
        }
        else if (const auto converged = ::converged(delta, e, s, etol, stol); converged)
        {
            status = csearch_status::converged;
            break;
        }
        else if (const auto descent = fx - fy >= m_m1 * delta; descent)
        {
            tL = t;
        }
        else
        {
            tR = t;
            if (tL < epsilon0<scalar_t>() && e <= m_m3 * delta)
            {
                status = csearch_status::null_step;
                break;
            }
            else
            {
                t = new_trial();
                continue;
            }
        }

        if (gy.dot(y - x) >= -m_m2 * delta)
        {
            status = csearch_status::descent_step;
            break;
        }
        else if (!std::isfinite(tR) && (s.lpNorm<2>() <= stol || s.dot(y - x) >= -m_m4 * delta))
        {
            status = csearch_status::cutting_plane_step;
            break;
        }
        else
        {
            t = new_trial();
        }
    }

    return m_point;
}

void csearch_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::m3"), 0, LT, 1.0, LT, 1e+6));
    c.register_parameter(parameter_t::make_scalar(scat(prefix, "::csearch::m4"), 0, LT, 1.0, LT, 1e+6));
    c.register_parameter(parameter_t::make_scalar_pair(scat(prefix, "::csearch::m1m2"), 0, LT, 0.5, LT, 0.9, LT, 1.0));
}

csearch_t csearch_t::make(const function_t& function, const configurable_t& c, const string_t& prefix)
{
    const auto [m1, m2] = c.parameter(scat(prefix, "::csearch::m1m2")).value_pair<scalar_t>();
    const auto m3       = c.parameter(scat(prefix, "::csearch::m3")).value<scalar_t>();
    const auto m4       = c.parameter(scat(prefix, "::csearch::m4")).value<scalar_t>();

    return {function, m1, m2, m3, m4};
}
