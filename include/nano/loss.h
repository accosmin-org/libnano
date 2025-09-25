#pragma once

#include <nano/configurable.h>
#include <nano/factory.h>
#include <nano/tensor.h>

namespace nano
{
class loss_t;
using rloss_t = std::unique_ptr<loss_t>;

///
/// \brief generic multivariate loss function of two parameters:
///     - the target value to predict (ground truth, annotation) and
///     - the machine learning model's output (prediction).
///
/// NB: usually the loss function upper-bounds or
///     approximates the true (usually non-smooth) error function to minimize.
///
class NANO_PUBLIC loss_t : public typed_t, public configurable_t, public clonable_t<loss_t>
{
public:
    ///
    /// \brief default constructor
    ///
    explicit loss_t(string_t id);

    ///
    /// \brief returns the available implementations
    ///
    static factory_t<loss_t>& all();

    ///
    /// \brief compute the error value and the loss (value, gradient and hessian) for each sample
    ///     given the targets (the ground truth) and the outputs (the predictions).
    ///
    /// NB: the targets and the outputs are given as 4D tensors,
    ///     where the first index is the sample index.
    ///
    /// NB: the gradients and the hessians keep the same shape as the targets and the outputs.
    ///
    void error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const;
    void value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const;
    void vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t) const;
    void vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_map_t) const;

    ///
    /// \brief overloads to simplify usage.
    ///
    /// NB: the output tensors are allocated accordingly.
    ///
    void error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& errors) const;
    void value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& values) const;
    void vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_t& vgrads) const;
    void vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_t& vhesss) const;

    ///
    /// \brief returns whether the loss function is convex.
    ///
    bool convex() const { return m_convex; }

    ///
    /// \brief returns whether the loss function is smooth.
    ///
    /// NB: if not, then only sub-gradients are available.
    ///
    bool smooth() const { return m_smooth; }

    ///
    /// \brief returns the expected dimensions of the cross-samples hessians given
    ///     the number of samples and the target dimensions.
    ///
    static tensor7d_dims_t make_hess_dims(tensor4d_cmap_t targets);
    static tensor7d_dims_t make_hess_dims(tensor4d_dims_t targets_dims);
    static tensor7d_dims_t make_hess_dims(tensor_size_t samples, tensor3d_dims_t target_dims);

protected:
    void convex(bool);
    void smooth(bool);

    virtual void do_error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const = 0;
    virtual void do_value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const = 0;
    virtual void do_vgrad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t) const = 0;
    virtual void do_vhess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor7d_map_t) const = 0;

private:
    // attributes
    bool m_convex{false}; ///< whether the loss function is convex
    bool m_smooth{false}; ///< whether the loss function is smooth (otherwise subgradients should be used)
};
} // namespace nano
