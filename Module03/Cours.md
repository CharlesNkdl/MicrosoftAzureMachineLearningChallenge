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
