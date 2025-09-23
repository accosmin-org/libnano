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
    /// \brief compute the error value, the loss value, the loss' gradient and the loss' hessian
    ///     wrt the output for the given samples.
    ///
    /// NB: the targets and the outputs are given as 4D tensors,
    ///     where the first index is the sample index.
    ///
    /// NB: the gradients keep the same shape as the targets and the outputs.
    /// NB: the hessians are flatten across the target dimensions.
    ///
    virtual void error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const = 0;
    virtual void value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_map_t) const = 0;
    virtual void grad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_map_t) const  = 0;
    virtual void hess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor3d_map_t) const  = 0;

    ///
    /// \brief overloads to simplify usage.
    ///
    /// NB: the output tensors are allocated accordingly.
    ///
    void error(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& errors) const
    {
        errors.resize(targets.size<0>());
        error(targets, outputs, errors.tensor());
    }

    void value(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor1d_t& values) const
    {
        values.resize(targets.size<0>());
        value(targets, outputs, values.tensor());
    }

    void grad(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor4d_t& grads) const
    {
        grads.resize(targets.dims());
        grad(targets, outputs, grads.tensor());
    }

    void hess(tensor4d_cmap_t targets, tensor4d_cmap_t outputs, tensor3d_t& vhess) const
    {
        const auto dims = targets.size<1>() * targets.size<2>() * targets.size<3>();
        vhess.resize(targets.size<0>(), dims, dims);
        hess(targets, outputs, vhess.tensor());
    }

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

protected:
    void convex(bool);
    void smooth(bool);

private:
    // attributes
    bool m_convex{false}; ///< whether the loss function is convex
    bool m_smooth{false}; ///< whether the loss function is smooth (otherwise subgradients should be used)
};
} // namespace nano
