#include <cmath>
#include <cassert>
#include <ostream>
#include <nano/mlearn/train.h>

using namespace nano;

train_point_t::train_point_t(scalar_t tr_value, scalar_t tr_error, scalar_t vd_error) :
    m_tr_value(tr_value),
    m_tr_error(tr_error),
    m_vd_error(vd_error)
{
}

bool train_point_t::valid() const
{
    return  std::isfinite(m_tr_value) &&
            std::isfinite(m_tr_error) &&
            std::isfinite(m_vd_error);
}

train_curve_t::train_curve_t(std::unordered_map<string_t, scalar_t> params) :
    m_params(std::move(params))
{
}

void train_curve_t::add(scalar_t tr_value, scalar_t tr_error, scalar_t vd_error)
{
    m_points.emplace_back(tr_value, tr_error, vd_error);
}

train_status train_curve_t::check(size_t patience) const
{
    if (!m_points.empty() && !m_points.rbegin()->valid())
    {
        return train_status::diverged;
    }
    else if (m_points.size() < 2)
    {
        return train_status::better;
    }
    else if (optindex() + patience < m_points.size())
    {
        return train_status::overfit;
    }
    else
    {
        const auto size = m_points.size();
        return (m_points[size - 2] < m_points[size - 1]) ? train_status::worse : train_status::better;
    }
}

size_t train_curve_t::optindex() const
{
    const auto it = std::min_element(m_points.begin(), m_points.end());
    return static_cast<size_t>(std::distance(m_points.begin(), it));
}

train_point_t train_curve_t::optimum() const
{
    const auto it = std::min_element(m_points.begin(), m_points.end());
    assert(it != m_points.end());
    return *it;
}

std::ostream& train_curve_t::save(std::ostream& stream, const char delim, const bool header) const
{
    if (header)
    {
        stream << "step" << delim << "tr_value" << delim << "tr_error" << delim << "vd_error" << "\n";
    }

    for (size_t i = 0U, size = m_points.size(); static_cast<bool>(stream) && (i < size); ++ i)
    {
        const auto& point = m_points[i];
        stream << i << delim << point.tr_value() << delim << point.tr_error() << delim << point.vd_error() << "\n";
    }

    if (!stream)
    {
        stream.setstate(std::ios_base::failbit);
    }
    return stream;
}

train_curve_t& train_fold_t::add(const std::unordered_map<string_t, scalar_t>& params)
{
    string_t name;
    for (const auto& param : params)
    {
        name += scat(param.first, "=", param.second, ";");
    }

    const auto it = m_curves.emplace(name, params);
    return it.first->second;
}

std::pair<string_t, const train_curve_t> train_fold_t::optimum() const
{
    assert(!m_curves.empty());
    const auto it = std::min_element(m_curves.begin(), m_curves.end(), [] (const auto& lhs, const auto& rhs)
    {
        return lhs.second < rhs.second;
    });
    return *it;
}

scalar_t train_fold_t::tr_value() const
{
    return m_curves.empty() ? std::numeric_limits<scalar_t>::quiet_NaN() : optimum().second.optimum().tr_value();
}

scalar_t train_fold_t::tr_error() const
{
    return m_curves.empty() ? std::numeric_limits<scalar_t>::quiet_NaN() : optimum().second.optimum().tr_error();
}

scalar_t train_fold_t::vd_error() const
{
    return m_curves.empty() ? std::numeric_limits<scalar_t>::quiet_NaN() : optimum().second.optimum().vd_error();
}

std::ostream& train_result_t::save(std::ostream& stream, const char delim, const bool header) const
{
    if (header)
    {
        stream << "fold" << delim  << "tr_error" << delim << "vd_error" << delim << "te_error" << "\n";
    }

    for (size_t i = 0U, size = m_folds.size(); static_cast<bool>(stream) && (i < size); ++ i)
    {
        const auto& fold = m_folds[i];
        stream << i << delim << fold.tr_error() << delim << fold.vd_error() << delim << fold.te_error() << "\n";
    }

    if (!stream)
    {
        stream.setstate(std::ios_base::failbit);
    }
    return stream;
}
