#include <nano/core/numeric.h>
#include <nano/core/random.h>
#include <nano/splitter/random.h>

using namespace nano;

random_splitter_t::random_splitter_t()
    : splitter_t("random")
{
    register_parameter(parameter_t::make_integer("splitter::random::train_per", 10, LE, 80, LE, 90));
}

splitter_t::splits_t random_splitter_t::split(indices_t samples) const
{
    const auto seed  = parameter("splitter::seed").value<uint64_t>();
    const auto folds = parameter("splitter::folds").value<tensor_size_t>();

    const auto train_perc = parameter("splitter::random::train_per").value<tensor_size_t>();
    const auto train_size = idiv(train_perc * samples.size(), 100);
    const auto valid_size = samples.size() - train_size;

    auto rng = make_rng(seed);

    splits_t splits;
    splits.reserve(static_cast<size_t>(folds));

    for (tensor_size_t fold = 0; fold < folds; ++fold)
    {
        std::shuffle(std::begin(samples), std::end(samples), rng);

        indices_t valid(valid_size);
        indices_t train(train_size);

        train.vector() = samples.vector().segment(0, train_size);
        valid.vector() = samples.vector().segment(train_size, valid_size);

        // NB: sorting samples by index may increase speed!
        std::sort(std::begin(train), std::end(train));
        std::sort(std::begin(valid), std::end(valid));

        splits.emplace_back(std::move(train), std::move(valid));
    }

    return splits;
} // LCOV_EXCL_LINE

rsplitter_t random_splitter_t::clone() const
{
    return std::make_unique<random_splitter_t>(*this);
}
