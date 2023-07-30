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


