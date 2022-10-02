#include <nano/core/random.h>
#include <nano/splitter/kfold.h>

using namespace nano;

kfold_splitter_t::kfold_splitter_t()
    : splitter_t("k-fold")
{
}

splitter_t::splits_t kfold_splitter_t::split(indices_t samples) const
{
    const auto seed  = parameter("splitter::seed").value<uint64_t>();
    const auto folds = parameter("splitter::folds").value<tensor_size_t>();

    std::shuffle(begin(samples), end(samples), make_rng(seed));

    splits_t splits;
    splits.reserve(static_cast<size_t>(folds));

    for (tensor_size_t fold = 0; fold < folds; ++fold)
    {
        const auto chunk       = samples.size() / folds;
        const auto valid_begin = fold * chunk;
        const auto valid_end   = (fold + 1 < folds) ? (valid_begin + chunk) : samples.size();

        indices_t valid(valid_end - valid_begin);
        indices_t train(samples.size() - valid.size());

        const auto world = samples.vector();

        valid.vector()                         = world.segment(valid_begin, valid.size());
        train.vector().segment(0, valid_begin) = world.segment(0, valid_begin);
        train.vector().segment(valid_begin, train.size() - valid_begin) =
            world.segment(valid_end, world.size() - valid_end);

        // NB: sorting samples by index may increase speed!
        std::sort(begin(train), end(train));
        std::sort(begin(valid), end(valid));

        splits.emplace_back(std::move(train), std::move(valid));
    }

    return splits;
}

rsplitter_t kfold_splitter_t::clone() const
{
    return std::make_unique<kfold_splitter_t>(*this);
}
