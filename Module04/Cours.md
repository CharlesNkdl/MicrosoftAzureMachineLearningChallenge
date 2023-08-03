# Module 04 : Use Automated Machine Learning in Azure Machine Learning #

- [Module 04 : Use Automated Machine Learning in Azure Machine Learning](#module-04--use-automated-machine-learning-in-azure-machine-learning)
	- [Introduction](#introduction)
	- [What is ML](#what-is-ml)
	- [What is Azure Machine Learning studio?](#what-is-azure-machine-learning-studio)
	- [What is Azure Automated ML](#what-is-azure-automated-ml)

## Introduction ##

In this module we'll :

- Identify the ML process
- Understand MSAz Learning capabilities
- Use automated ML in MSAz to train and deploy a predictive model

## What is ML ##

ML uses mathematics and statistics to create a model that can predict unknown values.

There is two general approches to ML, supervised and unsupervised ML.

Supervised requires a dataset with known label values, includes two types :

- Regression : Used to predict a continues value; like a price, a sales total or some other measures
- Classification: Used to determine a class label, example of a binary class, a patient and whether he have diabetes or not

Unsupervised ML starts with a dataset without known label values, one example is clustering :

- Clustering : Used to determine labels, by grouping similar informations into label groups.

For example, a bike rental agency.
The model want to predicts the number of bikes rental per day.
The label (the thing we want to predict) = y = Bike rented
The features (the thing that will predict the label) = x = the weather, the day ....

Process : We need the data, called data ingestion
Then there is the data pre-processing, we'll chose which feature will help to predict the model, exclude outlier, and fill-in or opt-out missing values

## What is Azure Machine Learning studio? ##

Is a cloud based service that helps simplify some of the tasks to prepare data, train a model and deploy a predictive service.

It increases efficiency since it automates many of the time consuming tasks associated with training models.
It also enables cloud based compute resources that scale effectively to handle large volumes of data.

To use Azure ML, you need to create a workspace, then use it to manage data, compute resourcesm clode, models, and other artifacts

There is also the Azure ML studio, which is a web portal for ML solutions. It includes a wide range of features and capabilities to prepare data, train models, publish predictive services, and monitor their usage.

We can create 4 kinds of compute :

- Compute instances : Development workstations to work with data and models
- Compute clusters : Scalable clusters of VM for on demand processing of experiment code
- Inference clusters : Deployment targets for predictive services that use your trained model
- Attached compute : links to existing Azure compute resources, such as VM or Azure Databricks

## What is Azure Automated ML ##

It includes an automated ML capabolity. It tries multiple pre processing techniques and model-training algorithms in parallel.
It can allow us to train models even without extensive DS knowledge.

We can setup operations aka Jobs with multiple settings before starting ML.

## Understand the AutoML process ##

The steps are the following :
- Prepare data: Identify features and label. pre process, clean and transform the data.
- Train model : split the data into two groups, a training and validation set. Train it then validate it.
- Evaluate performance : compare how close the model's predictions are
- Deploy a predictive service : After you train a ML model, you can deploy it

We can use the automated ML for :

- Classification = Predicting categories or classes
- Regression = Predicting numeric values
- Time series forecasting = Predicting numeric values at a future point in time

### Evaluate Performance ###

The best model is identified based on the evaluation metric you specified, Normalized root mean squared error.

A technique called cross-validation is used to calculate the evaluation metric. After the model is trained using a portion of the data, the remaining portion is used to iteratively test, or cross-validate, the trained model. The metric is calculated by comparing the predicted value from the test with the actual known value, or label.

The difference between the predicted and actual value, known as the residuals, indicates the amount of error in the model. The performance metric root mean squared error (RMSE), is calculated by squaring the errors across all of the test cases, finding the mean of these squares, and then taking the square root. What all of this means is that smaller this value is, the more accurate the model's predictions. The normalized root mean squared error (NRMSE) standardizes the RMSE metric so it can be used for comparison between models which have variables on different scales.

The Residual Histogram shows the frequency of residual value ranges. Residuals represent variance between predicted and true values that can't be explained by the model, in other words, errors. You should hope to see the most frequently occurring residual values clustered around zero. You want small errors with fewer errors at the extreme ends of the scale.

### Exercise - Explore Automated ML in Azure ML ### 

Dataset : A CSV file with 732 lines and 13 columns about bike rental agency predictions on rentals

Results : With Max Abs Scaler and Light GBM 

Explained variance = Explained variance is a statistical measure that quantifies the proportion of the total variance in a dataset that is explained by a statistical model. It is used to measure the discrepancy between a model and actual data, which is the part of the model's total variance that is explained by factors that are actually present and isn't due to error variance.

0.83255

Mean absolute error = Mean Absolute Error (MAE) is a metric used to evaluate the performance of a regression model. It is the average of all absolute errors in a collection of predictions, without taking their direction into account. MAE is a linear score, meaning all individual differences contribute equally to the mean.

181.05

Mean absolute percentage error = Mean Absolute Percentage Error (MAPE) is a metric used to measure the accuracy of a forecasting method. It represents the average of the absolute percentage errors of each entry in a dataset, showing how accurate the forecasted quantities were in comparison with the actual quantities.

50.661

Median absolute error = In statistics, the median absolute deviation (MAD) is a robust measure of the variability of a univariate sample of quantitative data.

112.90

Normalized mean absolute error = measure of errors between paired observations expressing the same phenomenon.

0.053126

Normalized median absolute error =

0.033129

Normalized root mean squared error = root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed. 

0.080489

Normalized root mean squared log error

0.065856

R2 score = Coefficient of determination, indicator for how well data points fit a line or curve

0.83075

Root mean squared error

274.31

Root mean squared log error

0.46337

Spearman correlation = The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables; while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not). If there are no repeated data values, a perfect Spearman correlation of +1 or −1 occurs when each of the variables is a perfect monotone function of the other.

0.92297

When we explore the processed data, we can see the most important features for predicting the label is the workingday, the temperature and year.
We can also visualize the python file generated for the model training, which is put in here.



