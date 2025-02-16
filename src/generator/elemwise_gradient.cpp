#include <nano/generator/elemwise_gradient.h>
#include <nano/generator/select.h>

using namespace nano;

feature_mapping_t elemwise_gradient_t::do_fit()
{
    const auto mapping = select_struct(datasource(), original_features());

    tensor_size_t count = 0;
    for (tensor_size_t i = 0; i < mapping.size<0>(); ++i)
    {
        if (mapping(i, 3) >= 3 && mapping(i, 4) >= 3)
        {
            const auto channels = mapping(i, 2);
            count += channels * 4; // NB: input channels * gradient features!
        }
    }

    auto feature_mapping = feature_mapping_t{count, 7};
    for (tensor_size_t i = 0, k = 0; i < mapping.size<0>(); ++i)
    {
        if (mapping(i, 3) >= 3 && mapping(i, 4) >= 3)
        {
            for (tensor_size_t channel = 0, channels = mapping(i, 2); channel < channels; ++channel)
            {
                for (tensor_size_t type = 0; type < 4; ++type)
                {
                    feature_mapping.vector(k).segment(0, 5) = mapping.vector(i).segment(0, 5);
                    feature_mapping(k, 2)                   = 1; // one channel filtered at a time
                    feature_mapping(k, 3) -= 2;                  // rows after filtering with a 3x3 kernel
                    feature_mapping(k, 4) -= 2;                  // columns after filtering with a 3x3 kernel
                    feature_mapping(k, 5)   = channel;
                    feature_mapping(k++, 6) = type;
                }
            }
        }
    }

    return feature_mapping;
}

feature_t elemwise_gradient_t::feature(const tensor_size_t ifeature) const
{
    const auto original = mapped_original(ifeature);
    const auto dims     = mapped_dims(ifeature);

    const auto& feature = datasource().feature(original);

    auto suffix = scat(m_type);
    switch (mapped_mode(ifeature))
    {
    case gradient3x3_mode::gradx:
        suffix += "::gx";
        break;
    case gradient3x3_mode::grady:
        suffix += "::gy";
        break;
    case gradient3x3_mode::magnitude:
        suffix += "::gg";
        break;
    default:
        suffix += "::theta";
        break;
    }

    const auto channel = mapped_channel(ifeature);

    return feature_t{scat(suffix, "(", feature.name(), "[channel::", channel, "])")}.scalar(feature_type::float64,
                                                                                            dims);
}

tensor_size_t elemwise_gradient_t::mapped_channel(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return mapping()(ifeature, 5);
}

gradient3x3_mode elemwise_gradient_t::mapped_mode(tensor_size_t ifeature) const
{
    assert(ifeature >= 0 && ifeature < features());
    return static_cast<gradient3x3_mode>(mapping()(ifeature, 6));
}

template class nano::elemwise_generator_t<elemwise_gradient_t>;
