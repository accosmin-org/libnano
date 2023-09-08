# Machine learning module


#### Introduction

Libnano implements machine learning models to address the following types of supervised machine learning (ML) problems:
* supervised univariate and multivariate `regression`,
* supervised `binary classification` and
* single-label or multi-label `multi-class classification`.

The most important concepts related to training and evaluating ML models are mapped to easy-to-use and extensible interfaces. The builtin implementations are available in the associated factories. The next sections explain how these interfaces work:
* [datasource_t](../include/nano/datasource.h) - in-memory collection of samples consisting of input features and optional targets.
* [generator_t](../include/nano/generator.h) - generate features on the fly from a given dataset.
* [dataset_t](../include/nano/dataset.h) - machine learning dataset consisting of samples and feature generators.
* [loss_t](../include/nano/loss.h) - loss function measuring how well the model's predictions match the target.
* [tuner_t](../include/nano/tuner.h) - strategy to optimize hyper-parameters of models.
* [splitter_t](../include/nano/splitter.h) - strategy to split samples into training and validation (testing).

The implementation follows an optimization approach to training machine learning models. As such the loss function and the optional regularization terms form a function of the model's parameters to be minimized. The library uses a `solver_t` instance (see the [nonlinear numerical optimization module](nonlinear.md))  to minimize such functions and to yield the optimum parameters of the machine learning model of interest.


#### Dataset

A **data source** is a collection of samples described by features (e.g. images, measurements, time series) useful for training and evaluating ML models. Libnano has built-in support for several flavors of well-known datasets. Use the following command to download and uncompress locally these datasets to ```${HOME}/libnano/datasets```:
```
bash scripts/download_datasets.sh --all
```
so that they are available to the library.

The `datasource_t` object is used for storing efficiently in-memory many kinds of samples. It supports categorical (single-class and multi-class), continuous (of various storage types, like int32, float, double) and structured (of various storage types to represent for example images) input features and targets. Missing input features are supported.

The builtin standard ML data sources are available from their associated factory with:
```
std::cout << make_table("datasource", datasource_t::all());
// prints something like...
|----------------|---------------------------------------------------------------------------------------------------|
| datasource     | description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| abalone        | predict the age of abalone from physical measurements (Waugh, 1995)                               |
| adult          | predict if a person makes more than 50K per year (Kohavi & Becker, 1994)                          |
| bank-marketing | predict if a client has subscribed a term deposit (Moro, Laureano & Cortez, 2011)                 |
| breast-cancer  | diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992) |
| cifar10        | classify 3x32x32 color images (CIFAR-10)                                                          |
| cifar100c      | classify 3x32x32 color images (CIFAR-100 with 20 coarse labels)                                   |
| cifar100f      | classify 3x32x32 color images (CIFAR-100 with 100 fine labels)                                    |
| fashion-mnist  | classify 28x28 grayscale images of fashion articles (Fashion-MNIST)                               |
| forest-fires   | predict the burned area of the forest (Cortez & Morais, 2007)                                     |
| iris           | classify flowers from physical measurements of the sepal and petal (Fisher, 1936)                 |
| mnist          | classify 28x28 grayscale images of hand-written digits (MNIST)                                    |
| wine           | predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)                   |
|----------------|---------------------------------------------------------------------------------------------------|
```


#### Feature generation

It is often useful to either represent the feature values of a dataset differently (e.g. flattening them for training and evaluating linear models) or to generate new feature values (e.g. quadratic terms of the original feature values). Libnano provides the `generator_t` interface to support these cases. The generated features can be categorical (single-class and multi-class), continuous or structured and take into account if the original feature values are missing.

Multiple generators can be combined on a given dataset using a `dataset_t` object. Then an iterator-like wrapper simplifies the usage for particular ML models as following:

* `flatten_iterator_t` - useful for flattening the generated features to train and evaluate dense models like linear models or feed-forward neural networks. This iterator supports caching of the inputs and the targets, access by grouping samples in fixed-size batches and access in both the single-threaded and the multi-threaded cases.

* `select_iterator_t` - useful for train and evaluating models that perform explicit feature selection like gradient boosting. This iterator supports grouping features by type (e.g. categorical, continuous or categorical) and access in both the single-threaded and the multi-threaded cases.

The builtin feature generators are available from their associated factory with:
```
std::cout << make_table("generator", generator_t::all());
// prints something like...
|-----------------|---------------------------------------------------------------------------------------------------|
| generator       | description                                                                                       |
|-----------------|---------------------------------------------------------------------------------------------------|
| gradient        | gradient-like features (e.g. edge orientation & magnitude) from structured features (e.g. images) |
| identity-mclass | identity transformation, forward the multi-label features                                         |
| identity-scalar | identity transformation, forward the scalar features                                              |
| identity-sclass | identity transformation, forward the single-label features                                        |
| identity-struct | identity transformation, forward the structured features (e.g. images)                            |
| product         | product of scalar features to generate quadratic terms                                            |
|-----------------|---------------------------------------------------------------------------------------------------|
```


#### Loss functions

The `loss_t` interface is used for measuring how well the predictions of machine learning model match the associated targets. This requires two components:

* the `error function` - the lower, the more accurate the model. Note that most of such errors are not differentiable and as such cannot be easily used during training. Examples: the 0-1 loss for classification or the mean absolute error for regression problems.

* the `loss function` - a typically smooth upper bound or approximation of the error function that is easier to minimize during training. Examples: the logistic loss for classification or the mean squared error for regression problems. Note that the gradient with respect to the predictions is required to train the supported ML models.

The builtin loss functions are available from their associated factory with:
```
std::cout << make_table("loss", loss_t::all());
// prints something like...
|-----------------|------------------------------------------------------------|
| loss            | description                                                |
|-----------------|------------------------------------------------------------|
| absolute        | absolute error (multivariate regression)                   |
| cauchy          | cauchy loss (multivariate regression)                      |
| m-exponential   | exponential loss (multi-label classification)              |
| m-hinge         | hinge loss (multi-label classification)                    |
| m-logistic      | logistic loss (multi-label classification)                 |
| m-savage        | savage loss (multi-label classification)                   |
| m-squared-hinge | squared hinge loss (multi-label classification)            |
| m-tangent       | tangent loss (multi-label classification)                  |
| s-classnll      | class negative log likehoold (single-label classification) |
| s-exponential   | exponential loss (single-label classification)             |
| s-hinge         | hinge loss (single-label classification)                   |
| s-logistic      | logistic loss (single-label classification)                |
| s-savage        | savage loss (single-label classification)                  |
| s-squared-hinge | squared hinge loss (single-label classification)           |
| s-tangent       | tangent loss (single-label classification)                 |
| squared         | squared error (multivariate regression)                    |
|-----------------|------------------------------------------------------------|
```

The predictions and the targets are represented as 3D tensors for greater flexibility. In the majority of cases these are flatten for typical regression and classification loss functions. But this design allows custom loss functions for structured targets to be usable within the library (e.g. reconstructing color images).



#### Machine learning models

The `model_t` interface represents a ML model with the following most important functionalities:
* training with a given dataset, loss function and solver.
* automatic tuning of any hyper-parameters using the given tuning strategy (e.g. by fitting a quadratic surrogate function).
* customizable strategy for splitting samples into training (to optimize parameters) and validation (to evaluate and tune hyper-parameters, e.g. regularization parameters).
* evaluation on a given dataset and loss function.
* serialization to and from binary streams, e.g. file.

To properly train and evaluate ML models it is advised to divide samples using a nested cross-validation-like strategy in the following categories:

* *training* samples - to fit the ML model's parameters (e.g. coefficients, selected features) with the regularization hyper-parameters kept fixed.
* *validation* samples - to evaluate and tune the hyper-parameters (e.g. regularization strength).
* *testing* samples - to evaluate the final ML model refited on the union of the training and validation samples with the optimum hyper-parameters.

Note that this process should be repeated multiple times to get an estimation of the performance distribution as ML is a stochastic process.


The following ML models are builtin:
* [linear models](../docs/linear.md)
* [gradient boosting models](../docs/gboost.md)
