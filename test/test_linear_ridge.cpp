#include <fixture/datasource/linear.h>
#include <fixture/linear.h>
#include <fixture/loss.h>
#include <fixture/solver.h>

using namespace nano;
using namespace nano::ml;

UTEST_BEGIN_MODULE(test_linear_ridge)

UTEST_CASE(ridge)
{
    const auto datasource = make_linear_datasource(100, 1, 4, "datasource::linear::relevant", 70);
    const auto dataset    = make_dataset(datasource);
    const auto samples    = arange(0, dataset.samples());
    const auto model      = make_model("ridge", scaling_type::mean, 100);

    const auto param_names = strings_t{"l2reg"};
    for (const auto& loss_id : strings_t{"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss       = make_loss(loss_id);
        const auto solver     = loss_id == "mse" ? make_solver("lbfgs") : make_solver("rqb");
        const auto fit_params = make_fit_params(solver);
        const auto result     = model->fit(dataset, samples, *loss, fit_params);
        const auto epsilon    = 1e-6;

        check_result(result, param_names, 2, epsilon);
        check_model(*model, dataset, samples, epsilon);
        check_importance(*model, dataset, datasource.relevant_feature_mask());
    }
}

UTEST_END_MODULE()