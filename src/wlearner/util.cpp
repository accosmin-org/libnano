#include <nano/wlearner/util.h>

using namespace nano;

void nano::wlearner::scale(tensor4d_t& tables, const vector_t& scale)
{
    assert(scale.size() == 1 || scale.size() == tables.size<0>());
    assert(scale.minCoeff() >= 0);

    for (tensor_size_t i = 0; i < tables.size<0>(); ++i)
    {
        tables.array(i) *= scale(std::min(i, scale.size() - 1));
    }
}

rwlearners_t nano::wlearner::clone(const rwlearners_t& wlearners)
{
    auto clones = rwlearners_t{};
    clones.reserve(wlearners.size());
    for (const auto& wlearner : wlearners)
    {
        assert(wlearner.get() != nullptr);
        clones.emplace_back(wlearner->clone());
    }

    return clones;
}

void nano::wlearner::merge(rwlearners_t& wlearners)
{
    for (size_t i = 0U; i < wlearners.size(); ++i)
    {
        if (!wlearners[i])
        {
            continue;
        }

        auto merged = false;
        for (size_t j = i + 1U; j < wlearners.size(); ++j)
        {
            if (wlearners[i]->try_merge(wlearners[j]))
            {
                wlearners[j] = nullptr;
                merged       = true;
            }
        }

        if (!merged)
        {
            break;
        }
    }

    // NB: remove merged weak learners!
    wlearners.erase(std::remove_if(wlearners.begin(), wlearners.end(),
                                   [](const auto& wlearner) { return !static_cast<bool>(wlearner); }),
                    wlearners.end());
}
