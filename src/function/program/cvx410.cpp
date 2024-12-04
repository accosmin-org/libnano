#include <Eigen/Dense>
#include <function/program/cvx410.h>
#include <nano/core/scat.h>

using namespace nano;

linear_program_cvx410_t::linear_program_cvx48f_t(const tensor_size_t dims, const bool feasible)
    : linear_program_t(scat("cvx410-[", feasible ? "feasible" : "unfeasible", "]"), dims)
{
    const auto D = make_random_matrix<scalar_t>(dims, dims);
    const auto A = D.transpose() * D + matrix_t::identity(dims, dims);
    const auto c = make_random_vector<scalar_t>(dims);

    if (feasible)
    {
        // the solution is feasible
        const auto x = make_random_vector<scalar_t>(dims, +1.0, +2.0);
        const auto b = A * x;

        reset(c);
        optimum(x);

        A* variable() == b;
        variable() >= 0.0;
    }
    else
    {
        // the solution is not feasible
        const auto x = make_random_vector<scalar_t>(dims, -2.0, -1.0);
        const auto b = A * x;

        reset(c);
        optimum(x);

        A* variable() == b;
        variable() >= 0.0;

    TODO:
        expected.status(solver_status::unfeasible);
    }
}

rfunction_t linear_program_cvx410_t::clone() const
{
    return std::make_unique<linear_program_cvx410_t>(*this);
}

rfunction_t linear_program_cvx410_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx410_t>(dims);
}
