#include <Eigen/Dense>
#include <nano/program/util.h>
#include <nano/tensor/stack.h>

using namespace nano;

namespace
{
void reduce(matrix_t& A)
{
    // independant linear constraints
    const auto dd = A.transpose().fullPivLu();
    if (dd.rank() == A.rows())
    {
        return;
    }

    // dependant linear constraints, use decomposition to formulate equivalent linear equality constraints
    const auto& P  = dd.permutationP();
    const auto& LU = dd.matrixLU();

    const auto n = std::min(A.rows(), A.cols());
    const auto L = LU.leftCols(n).triangularView<Eigen::UnitLower>().toDenseMatrix();
    const auto U = LU.topRows(n).triangularView<Eigen::Upper>().toDenseMatrix();

    A = U.transpose().block(0, 0, dd.rank(), U.rows()) * L.transpose() * P;
}
} // namespace

bool nano::program::is_psd(matrix_cmap_t tQ)
{
    const auto Q = tQ.matrix();
    if (!Q.isApprox(Q.transpose()))
    {
        return false;
    }

    const auto ldlt = Q.selfadjointView<Eigen::Upper>().ldlt();
    return ldlt.info() != Eigen::NumericalIssue && ldlt.isPositive();
}

bool nano::program::reduce(matrix_t& A, vector_t& b)
{
    assert(A.rows() == b.size());

    if (A.rows() == 0)
    {
        return false;
    }

    // NB: need to reduce [A|b] altogether!
    auto Ab = ::nano::stack<scalar_t>(A.rows(), A.cols() + 1, A.matrix(), b.vector());
    ::reduce(Ab);

    A = Ab.block(0, 0, Ab.rows(), Ab.cols() - 1);
    b = Ab.matrix().col(Ab.cols() - 1);
    return true;
}
