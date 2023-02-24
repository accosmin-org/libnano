#include "fixture/datasource/hits.h"
#include "fixture/datasource/random.h"

///
/// \brief synthetic dataset:
///     the targets is a random affine transformation of the flatten input features.
///
/// NB: uniformly-distributed noise is added to targets if noise() > 0.
/// NB: every column % modulo() is not taken into account.
///
class linear_datasource_t final : public random_datasource_t
{
public:
    linear_datasource_t(const tensor_size_t samples, const tensor_size_t features, const tensor_size_t targets)
        : random_datasource_t(samples, make_features(features, targets), static_cast<size_t>(features),
                              make_all_hits(samples, features + 1))
    {
    }

    rdatasource_t clone() const override { return std::make_unique<linear_datasource_t>(*this); }

    void noise(scalar_t noise) { m_noise = noise; }

    void modulo(tensor_size_t modulo) { m_modulo = modulo; }

    auto noise() const { return m_noise; }

    const auto& bias() const { return m_bias; }

    const auto& weights() const { return m_weights; }

private:
    static features_t make_features(const tensor_size_t n_features, const tensor_size_t n_targets)
    {
        features_t features;
        for (tensor_size_t ifeature = 0; ifeature < n_features; ++ifeature)
        {
            feature_t feature;
            switch (ifeature % 4)
            {
            case 0: feature = feature_t{scat("scalar", ifeature)}.scalar(); break;
            case 1: feature = feature_t{scat("sclass", ifeature)}.sclass(3U); break;
            case 2: feature = feature_t{scat("mclass", ifeature)}.mclass(4U); break;
            default:
                feature = feature_t{scat("struct", ifeature)}.scalar(feature_type::float64, make_dims(2, 1, 3));
                break;
            }
            features.push_back(feature);
        }
        features.push_back(feature_t{"Wx+b+eps"}.scalar(feature_type::float64, make_dims(n_targets, 1, 1)));

        return features;
    }

    void do_load() override
    {
        random_datasource_t::do_load();

        // create samples: target = weights * input + bias + noise
        const auto dataset = make_dataset(*this);
        const auto itarget = features();
        const auto targets = ::nano::size(dataset.target_dims());
        const auto samples = dataset.samples();

        m_bias.resize(targets);
        m_weights.resize(targets, dataset.columns());

        m_bias.random();
        m_weights.random();
        for (tensor_size_t column = m_modulo, columns = dataset.columns(); column < columns; column += m_modulo)
        {
            m_weights.matrix().row(column).setConstant(0.0);
        }

        auto iterator = flatten_iterator_t{dataset, arange(0, samples)};
        iterator.loop(
            [&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
            {
                auto target  = tensor1d_t{targets};
                auto weights = m_weights.matrix();

                for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
                {
                    target.vector() = weights * inputs.vector(i) + m_bias.vector();
                    target.vector() += m_noise * make_random_vector<scalar_t>(m_bias.size());
                    set(i + range.begin(), static_cast<tensor_size_t>(itarget), target);
                }
            });
    }

    // attributes
    scalar_t      m_noise{0};  ///< noise level (relative to the [-1,+1] uniform distribution)
    tensor_size_t m_modulo{3}; ///< modulo columns to exclude from creating the targets
    tensor2d_t    m_weights;   ///< 2D weight matrix that maps the input to the output
    tensor1d_t    m_bias;      ///< 1D bias vector that offsets the output
};
