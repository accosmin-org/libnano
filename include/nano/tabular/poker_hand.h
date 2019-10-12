#pragma once

#include <nano/tabular.h>

namespace nano
{
    ///
    /// \brief Poker hand dataset: http://archive.ics.uci.edu/ml/datasets/Poker+Hand
    ///
    class poker_hand_dataset_t final : public tabular_dataset_t
    {
    public:

        poker_hand_dataset_t();
        split_t make_split() const override;
    };
}
