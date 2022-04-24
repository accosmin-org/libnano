#include "fixture/loss.h"
#include "fixture/linear.h"
#include <nano/linear/regularization.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_linear_model)

UTEST_CASE(regularization_variance)
{
    const auto dataset = make_dataset(100, 1, 4);
    const auto generator = make_generator(dataset);
    const auto samples = arange(0, dataset.samples());

    auto model = make_model();
    model.parameter("model::linear::scaling") = scaling_type::minmax;
    model.parameter("model::linear::regularization") = linear::regularization_type::variance;

    const auto param_names = strings_t{"vAreg"};
    for (const auto* const loss_id : {"squared", "absolute"})
    {
        [[maybe_unused]] const auto _ = utest_test_name_t{loss_id};

        const auto loss = make_loss(loss_id);
        const auto solver = make_solver(loss_id);
        const auto result = model.fit(generator, samples, *loss, *solver);
        const auto epsilon = string_t{loss_id} == "squared" ? 1e-6 : 1e-3;

        check_result(result, param_names, 6U, epsilon);
        check_model(model, generator, samples, epsilon);
    }
}

UTEST_END_MODULE()
