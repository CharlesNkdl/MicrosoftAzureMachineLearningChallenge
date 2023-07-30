# Module 03 : Introduction to Machine Learning #

## Introduction ##

Machine-learning models are computer algorithms that use data to make estimations (educated guesses) or decisions. Machine-learning models differ from traditional algorithms in how they are designed. When normal computer software needs to be improved, people edit it. By contrast, a machine-learning algorithm uses data to get better at a specific task.

For example, spam filters use machine learning. 20 years ago, spam filters did not have many examples to learn from and were not good at identifying what is and isn’t spam. As more spam has arrived and been labeled as junk by human users, the machine-learning algorithms have gained more experience and become better at their job.

During all this module, we'll be using an example scenario to explain key concepts.

I own a shop for harnesses for avalanche-rescur dogs.
I recently expanded to also sell doggy boots.
Customers always get the correct harness size, but never the correct doggy boots size.
I have an idea : Maybe there is a way to approximate the boots size depending on the harness

## What are machine learning models? ##

The model is the core component of machine learning, and ultimately what we are trying to build.

Models can be built in many ways. For example, a traditional model that simulates how an airplane flies is built by people, using knowledge of physics and engineering. Machine-learning models are special; rather than being edited by people so that they work well, machine learning models are shaped by data. They learn from experience.

### How to think about models ###

You can think of a model as a function that accepts data as an input and produces an output. More specifically, a model uses input data to estimate something else. For example, in our scenario, we want to build a model that is given a harness size and estimates boot size

Models are often not meaningfully different from simple functions you're already familiar with. Like other code, they contain logic and parameters. For example, the logic might be “multiply the harness size by parameter_1”.

### Select a model ###

There are many model types, some simple and some complex.

Like all code, simpler models are often the most reliable and easy to understand, while complex models can potentially perform impressive feats. Which kind of model you should choose depends on your goal. For example, medical scientists often work with models that are relatively simple, because they are reliable and intuitive. By contrast, AI-based robots typically rely on very complex models.

The first step in machine learning is selecting the kind of model that you'd like to use. This means we're choosing a model based on its internal logic. For example, we might select a two-parameter model to estimate dog boot size from harness size:

Input = Harness_size = 4  => Model = Harness_size * P1 + P2  => Dog Boot size

Notice how we selected a model based on how it works logically, but not based on its parameter values. In fact, at this point the parameters have not yet been set to any particular value.

### Parameters are discovered during training ###

The human designer doesn't select parameter values. Instead, parameter values are set to an initial guess, then adjusted during an automated learning process called training.

Given our selection of a two-parameter model (above), we'll now provide random guesses for our parameters:

Input = Harness_size = 4  => Model = Harness_size * P1(0,2) + P2(1,2)  => Dog Boot size = 10  WRONG

These random parameters will mean the model isn’t good at estimating boot size, so we'll perform training. During training, these parameters are automatically changed to two new values that give better results:

Input = Harness_size = 4  => Model = Harness_size * P1(1,5) + P2(4)  => Dog Boot size = 10  CORRECT

## Exercise : Create a machine learning model ##

We have a datasets of 50 boot_size in relation with 50 harness_size.
We load a python library with models :

``` python

import statsmodels.formula.api as smf

```

We use a simple model called OLS :
" Ordinary least squares (OLS) is a linear least squares method used in statistics to choose unknown parameters in a linear regression model. It involves minimizing the sum of the squares of the differences between the observed dependent variable in the input dataset and the output of the linear function of the independent variable. OLS is considered the most useful optimization strategy for linear regression models as it helps find unbiased real value estimates for alpha and beta. It minimizes the sum of the squares of the differences between the observed values and the predicted values. "

They take two parameters : A slope and a offset. But since this is machinea learning, we don't know these values yet, we need to train it.

``` python

# Load some libraries to do the hard work for us
import graphing

# Train (fit) the model so that it creates a line that
# fits our data. This method does the hard work for
# us. We will look at how this method works in a later unit.
fitted_model = model.fit()

```

We do get the slope and the offset corresponding to the dataset we provided.
Now we can run prediction to find the right bootsize !

## What are inputs and outputs? ##

The goal of training is to improve a model so that it can make high-quality estimations or predictions. Once trained, you can use a model in the real world like normal software.

Models don’t train themselves. They're trained using data plus two pieces of code, the objective function and the optimizer. Let’s explore how these components work together to train a model to work well.

### The Objective ###

The objective is what we want the model to be able to do. For example, the objective of our scenario is to be able to estimate a dog’s boot size based on their harness size.

So that a computer can understand our objective, we need to provide our goal as code snippet called an objective function (also known as cost function). Objective functions judge whether the model is doing a good job (estimating boot size well) or bad job (estimating boot size badly).

### The data ###

Data refers to the information that we provide to the model (also known as inputs). In our scenario, this is harness size.

Data also refers to information that the objective function might need. For example, if our objective function reports whether the model guessed the boot size correctly, it will need to know the correct boot size! This is why in our previous exercise, we provided both harness sizes and the correct answers to the training code.

### The Optimizer ###

During training, the model makes a prediction, and the objective function calculates how well it performed. The optimizer is code that then changes the model’s parameters so the model will do a better job next time.

How an optimizer does this is complex, and something we'll cover in later material. Don’t be intimidated, though; we don’t normally write our own optimizers, we use open-source frameworks where the hard work has been done for us.

It's important to keep in mind that the objective, data, and optimizer are simply a means to train the model. They are not needed once training is complete. It's also important to remember that training only changes the parameter values inside of a model; it doesn't change what kind of model is used.

## Exercice : Datasets in Python ##

We'll load data from a file, filter it, and graph it. Doing so is a very important first step in order to build proper models, or to understand their limitations.

``` python

import pandas
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv

# Read the text file containing data using pandas
dataset = pandas.read_csv('doggy-boot-harness.csv')

# Print the data
# Because there are a lot of data, use head() to only print the first few rows
dataset.head()

```

Data is easy to filter by columns. We can either type this directly, like dataset.my_column_name, or like so: dataset["my_column_name"].

We can use this to either extract data, or to delete data.

Lets take a look at the harness sizes, and delete the sex and age_years columns.

``` python

# Look at the harness sizes
print("Harness sizes")
print(dataset.harness_size)

# Remove the sex and age-in-years columns.
del dataset["sex"]
del dataset["age_years"]

# Print the column names
print("\nAvailable columns after deleting sex and age information:")
print(dataset.columns.values)

```

We can access the top with head() , the bottom with tail().
We can put "mask" on each value of the dataset when iterating so we know when we need to print them.
This is more python-ish, but ok, loop is bad in python due to performance

``` python

# Determine whether each avalanche dog's harness size is < 55
# This creates a True or False value for each row where True means
# they are smaller than 55
is_small = dataset.harness_size < 55
print("\nWhether the dog's harness was smaller than size 55:")
print(is_small)

# Now apply this 'mask' to our data to keep the smaller dogs
data_from_small_dogs = dataset[is_small]
print("\nData for dogs with harness smaller than size 55:")
print(data_from_small_dogs)

# Print the number of small dogs
print(f"\nNumber of dogs with harness size less than 55: {len(data_from_small_dogs)}")

```

To be more compressive, we can copy the dataset by putting a parameter to check the condition
We can use the copy() to copy it.

``` python

data_smaller_paws = dataset[dataset.boot_size < 40].copy()

```

We'll begin to graph the data.
In this course, they use plotly.express, and a custom file named graphing.py.

## How to use a Model ##

It's important to make a distinction between training and using a model.

Using a model means providing inputs and receiving an estimation or prediction. We do this both when we're training our model and when we or our customers use it in the real world. Using a model normally takes less than a few seconds.

By contrast, training a model is the process of improving how well a model works. Training requires that we use the model, as well as the objective function and optimizer, in a special loop. This can take minutes or days to complete. Usually, we only train a model once. Once it's trained, we can use it as many times as we like without making further changes.

For example, in our avalanche-rescue dog store scenario, we want to train a model using a public dataset, which will change the model so that it can predict a dog’s boot size based on its harness size. Once our model is trained, we'll use the model as part of our online store to make sure customers are buying doggy boots that will fit their dogs.

When we use our model, we only need the column(s) of data that the model accepts as input. These columns are called features. In our scenario, if our model accepts harness size and estimates boot size, then our feature is harness size.

During training, the objective function usually needs to know both the model’s output and what the correct answer was. These are called labels. In our scenario, if our model predicts boot size, boot size is our label.

So when a model finished training, we just need the Features

## Exercise - Use Machine Learning models ##

We are gonna take a model which is already trained.
Because as said before, once the model is trained oncem we just want to load the model and use it
So the plan is to create the model, save it, load it, use it to make predictions

``` python

import pandas
!pip install statsmodels
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py
!wget https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv

# Load a file containing dog's boot and harness sizes
data = pandas.read_csv('doggy-boot-harness.csv')

```

As before, we'll train the model, still using the statsmodels.formula.api, and the Ols formula, Ordinary Least Squares.

``` python

model = smf.ols(formula = "boot_size ~ harness_size", data = data).fit()

```

Then we'll use a library called joblib, which is lightweight pipelining in python

``` python

import joblib

model_filename = './avalanche_dog_boot_model.pkl'
joblib.dump(model, model_filename)

```

Then we just have to load it, also with joblib to handle the pipelining.

``` python

model_loaded = joblib.load(model_filename)

```

Now we just need to create a function to calculate the bootsize

``` python

def load_predict(harness_size)
    model = joblib.load(model_filename)
    inputs = {"harness_size" : [harness_size]}
    predicted = loaded_model.predict(inputs)[0]
    return predicted

prediction = load_predict(45)
print("prediction :", prediction)

```

Now that we have a model which is trained, we need to put it in our webstore to warn users about wrong sized boots
For this we'll make a function which take in parameters the harness size, the boot size, and display an error message if this is far off the prediction

``` python

def check_size(harness_size, boot_size)
    estimated_boot_size = int(round(load_predict(harness_size)))
    if estimated_boot_size == boot_size
        return f"ok"
    if selected_boot_size < estimated_boot_size:
        # Selected boots might be too small
        return "The boots you have selected might be TOO SMALL for a dog as "\
               f"big as yours. We recommend a doggy boots size of {estimated_boot_size}."

    if selected_boot_size > estimated_boot_size:
        # Selected boots might be too big
        return "The boots you have selected might be TOO BIG for a dog as "\
               f"small as yours. We recommend a doggy boots size of {estimated_boot_size}."

check_size(55, 39)

```
