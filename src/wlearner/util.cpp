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
