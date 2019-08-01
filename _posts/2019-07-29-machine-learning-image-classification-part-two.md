---
layout: post
title:  "Machine Learning Image Classification Part Two"
tags: [ Machine Learning, Python ]
featured_image_thumbnail: assets/images/posts/2019/machine-learning-image-classification-part-two_thumbnail.png
featured_image: assets/images/posts/2019/machine-learning-image-classification-part-two_title.png
featured: false
hidden: false
---

The overall goal of this series of write-ups is to explore a number of models performing binary classification on a given set of images.  This is the second write-up in the series where we advance from utilizing a **_logistic regression model with gradient descent_** to a **_shallow neural network_**.  Our hope is that for each of the write-ups we'll advance to utilizing a more sophisticated model and achieve better and better classification results.

<!--more-->

The model created in this write-up be to perform binary classification on a set of images utilizing a shallow neural network, predict (hopefully correctly) if a given image is a cat or not (hence the binary nature of the model), and if possible to perform the classification task better than the previously developed system.  

Comparisons between the models will be achieved through accuracy ratings on the test set, as well as the model's F-Score.

For reference here are links to the previous entries in this series:
* [Machine Learning Image Classification Part One]({% post_url 2019-07-28-machine-learning-image-classification-part-one %})

So, let's get started!

# Notes on the How this Write-up Evolved

When I first started this write-up I was mostly focused on the mechanics of building the model.  I spend about an hour writing the code, and then fed in my data set.  The accuracy was poor and the cost function didn't monotonically decrease along with other issues.  I adjusted, adjusted, and then adjusted the model some more, and its performance still fell short or barely exceeded what the previous logistic regression model was able to achieve.

__And so, at that time, the real focus of this write-up become apparent:  The mechanics of building the model were easy, it was the *TUNING* of the model that was would require the real effort.__

And so that realization is what fueled most of the work done during this write-up.  I needed a way to quickly generate and adjust all the hyperparameters which previously I didn't have to contend with, record the results, and then compare the outputs to select the best model.

This in turn required I modify the model's code, write a better model execution utility, deal with how to display the outputs of multiple models, manage the display of many cost graphs, and so on and so forth.  So below are the fruits of those efforts, and as we continue to delve into additional models and optimizations I have no doubts we'll continue to evolve our utility set.

# Model Code Development

## Import libraries and data sets


```python
%matplotlib inline

# autoreload reloads modules automatically before entering the execution of code typed at the IPython prompt.
%load_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings('ignore')
```


```python
from os import path
from utils import *
import pandas as pd
from IPython.display import display, HTML
import numpy as np
from matplotlib import pyplot as plt
import inspect
import time
import copy

import random
random.seed(10)
np.random.seed(10)
```


```python
# Examine the data used for training the model
imageData = path.join("datasets", "imageData500_64pixels.hdf5")
validateArchive(imageData)
```

    *** KEYS
    HDF5 container keys: ['testData', 'testLabels', 'trainData', 'trainLabels']

    *** LABELS
    Total number of training labels: 800
    Number of cat labels: 396
    Number of object labels: 404
    First 10 training labels: [0 0 1 1 1 1 1 1 0]


    Total number of testing labels: 200
    Number of cat labels: 104
    Number of object labels: 96
    First 10 testing labels: [1 1 0 0 1 1 0 0 0]


    *** IMAGE DATA
    Image data shape in archive: (800, 64, 64, 3)


    First HDF5 container dataSet item shape: (64, 64, 3)
    Image data shape after flattening: (192, 64)
    First 10 dataSet item matrix values: []


    Recreating and showing first 20 images from flattened matrix values:

![png](assets/images/posts/2019/output_9_1.png)
![png](assets/images/posts/2019/output_9_2.png)
![png](assets/images/posts/2019/output_9_3.png)
![png](assets/images/posts/2019/output_9_4.png)

```python
# Load, shape, and normalize the data used for training the model
with h5py.File(imageData, "r") as archive:   
    trainingData = np.squeeze(archive["trainData"][:])
    testData = np.squeeze(archive["testData"][:])
    trainingLabels = np.array(archive["trainLabels"][:])
    testLabels = np.array(archive["testLabels"][:])
    archive.close()

print("Archive trainingData.shape:    ", trainingData.shape)
print("Archive trainingLabels.shape:  ", trainingLabels.shape)
print("Archive testData.shape:        ", testData.shape)
print("Archive testLabels.shape:      ", testLabels.shape)
print("\n")

# Reshape the training and test data and label matrices
trainingData = trainingData.reshape(trainingData.shape[0], -1).T
testData = testData.reshape(testData.shape[0], -1).T

print ("Flattened, normalized trainingData shape:  " + str(trainingData.shape))
print ("Flattened, normalized testData shape:      " + str(testData.shape))

# Normalization
trainingData = trainingData/255.
testData = testData/255.
```

    Archive trainingData.shape:     (800, 64, 64, 3)
    Archive trainingLabels.shape:   (1, 800)
    Archive testData.shape:         (200, 64, 64, 3)
    Archive testLabels.shape:       (1, 200)


    Flattened, normalized trainingData shape:  (12288, 800)
    Flattened, normalized testData shape:      (12288, 200)


## Write utility functions

```python
# Great reference:  https://www.python-course.eu/matplotlib_multiple_figures.php

# Write a function to show multiple graphs in the same figure
def printCostGraphs(costs, keys, cols, fsize = (15,6)):
    # Figure out how many rows and columns we need
    counter = 0
    rows = np.ceil(len(costs) / cols)
    fig = plt.figure(figsize = fsize)

    # Add each of the cost graphs to the figure
    for key in keys:
        c = np.squeeze(costs[key])
        sub = fig.add_subplot(rows, cols, counter + 1)
        sub.set_title('Epoch ' + str(key))
        sub.plot(c)
        counter = counter + 1

    # Draw the figure on the page
    plt.plot()
    plt.tight_layout()
    plt.show()
```


```python
# Randomize values for hyperparameters based on a given key:value dictionary
class HPicker:

    def pick(self, ranges):
        hParams = {}

        # For each parameter key:val
        for key, value in ranges.items():
            if isinstance(value, list):
                start, stop, step = value
                vals = []

                # Create a range of possible values
                while (start < stop):
                    start = round(start + step, len(str(step)))
                    vals.append(start)

                # Pick one of the possible values randomly    
                hParams[key] = random.choice(vals)
            else:
                hParams[key] = value

        return hParams     
```


```python
# Create instances of each of the activations we might utilize in the model

class AbstractActivation(object):
    def activate(self, z):
        raise NotImplementedError("Requires implementation by inheriting class.")

class Sigmoid(AbstractActivation):
    def activate(z):
        return 1 / (1 + np.exp(-(z)))

class Relu(AbstractActivation):
    def activate(z):
        return  z * (z > 0)
```


```python
# Create a pandas dataframe with labeled columns to record model training results
def getResultsDF(hRanges):
    columns = list(hRanges.keys())
    df = pd.DataFrame(columns = columns)

    return(df)
```


```python
# Do all the heavy lifting required when running N number of models with various hyperparameter configurations
def runModels(hRanges, epochs, silent = False):

    # Var inits
    picker = HPicker()
    resultsDF = getResultsDF(hRanges)
    costs = {}
    params = {}
    epoch = 0

    print("\n*** Starting model training")

    while (epoch < epochs):

        # Get the random hyperparam values
        hparams = picker.pick(hRanges)
        hparams["Epoch"] = epoch

        # Print a summary of the model about to be trained and its params to the user
        if silent is not True:
            print("Training epoch", epoch, "with params:  LR", hparams["Learning_Rate"],
                  ", iterations", hparams["Iterations"], ", HL units", hparams["HL_Units"],
                  ", lambda", hparams["Lambda"], ", and init. multi.", hparams["Weight_Multi"])

        # Train the model its given hyperparams and record the results
        params[epoch], costs[epoch],  hparams["Descending_Graph"] = model(
            trainingData, trainingLabels, hparams["HL_Units"], hparams["Iterations"],
            hparams["Learning_Rate"], hparams["Lambda"], hparams["Weight_Multi"], False)

        # Make predictions based on the model
        trainingPreds = predict(trainingData, params[epoch], trainingLabels)
        testPreds = predict(testData, params[epoch], testLabels)

        # Record prediction results
        hparams["Train_Acc"] = trainingPreds["accuracy"]
        hparams["Test_Acc"] = testPreds["accuracy"]
        hparams["Final_Cost"] = costs[epoch][-1]

        # Add model results to the pandas dataframe
        resultsDF.loc[epoch] = list(hparams.values())
        epoch = epoch + 1

    print("*** Done!\n")

    # Sort the dataframe so it's easier to find the results we are interested in
    resultsDF = resultsDF.sort_values(by = ['Descending_Graph', 'Test_Acc'], ascending = False)

    return resultsDF, params, costs
```

## Write the shallow neural network model code


```python
# Define the dimensions of the model
defineDimensions(data, labels, layerSize):
    nnDims = {}

    nnDims["numberInputs"] = data.shape[0]
    nnDims["hiddenLayerSize"] = layerSize
    nnDims["numberOutputs"] = labels.shape[0]

    return nnDims;
```


```python
#  Initialize model params (i.e. W and b)
def initilizeParameters(dimensionDict, multiplier):

    np.random.seed(10)  # Yes, this has to be done every time...  :(
    w1 = np.random.randn(dimensionDict["hiddenLayerSize"], dimensionDict["numberInputs"]) * multiplier

    np.random.seed(10)  # Yes, this has to be done every time...  :(
    w2 = np.random.randn(dimensionDict["numberOutputs"], dimensionDict["hiddenLayerSize"]) * multiplier

    b1 = np.zeros((dimensionDict["hiddenLayerSize"], 1))
    b2 = np.zeros((dimensionDict["numberOutputs"], 1))

    params = {
        "w1" : w1,
        "w2" : w2,
        "b1" : b1,
        "b2" : b2}

    return params

```


```python
# Perform forward propogation
def forwardPropagation(data, params, activation = Sigmoid()):
    w1 = params["w1"]
    b1 = params["b1"]
    w2 = params["w2"]
    b2 = params["b2"]

    z1 = np.dot(w1, data) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = Sigmoid.activate(z2)

    # Sanity check the dimensions
    assert(a2.shape == (1, data.shape[1]))

    cache = {"z1": z1,
             "a1": a1,
             "z2": z2,
             "a2": a2}

    return cache
```


```python
# Calculate the cost of the model (includes L2 regularization)
def calculateCost(labels, params, cache, lamb):
    # Define vars to make reading and writing the formulas easier below...
    m = labels.shape[1]
    a2 = cache["a2"]
    w1 = params["w1"]
    w2 = params["w2"]

    # Perform cost and regularization calculations
    crossEntropyCost = (-1/m) * np.sum( labels*np.log(a2) + (1-labels)*np.log(1-a2) )
    l2RegularizationCost = (1/m) * (lamb/2) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    finalCost = crossEntropyCost + l2RegularizationCost

    return finalCost
```


```python
# Perform backward propogation
def backwardPropagation(data, labels, params, cache, lamb):
    # Define and populate variables
    m = data.shape[1]
    w1 = params["w1"]
    w2 = params["w2"]
    a1 = cache["a1"]
    a2 = cache["a2"]

    # Calculate gradients
    dz2 = a2 - labels
    dw2 = (1/m) * np.dot(dz2, a1.T) + (lamb/m) * w2
    db2 = (1/m) * np.sum(dz2, axis = 1, keepdims = True)
    dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
    dw1 = (1/m) * np.dot(dz1, data.T) +  (lamb/m) * w1
    db1 = (1/m) * np.sum(dz1, axis = 1, keepdims = True)

    # Store in the gradients cache
    gradients = {"dw1": dw1,
             "db1": db1,
             "dw2": dw2,
             "db2": db2}

    return gradients
```


```python
# Update the model params based on the results of the backward propogation calculations
def updateParams(params, gradients, learningRate):
    params["w1"] = params["w1"] - learningRate * gradients["dw1"]
    params["b1"] = params["b1"] - learningRate * gradients["db1"]
    params["w2"] = params["w2"] - learningRate * gradients["dw2"]
    params["b2"] = params["b2"] - learningRate * gradients["db2"]

    return params
```


```python
# Define the actual neural network classification model
def model(data, labels, layerSize, numIterations, learningRate, lamb, initializeMultiplier, printCost = False, showGraph = False):

    # Init vars
    dims = defineDimensions(data, labels, layerSize)
    params = initilizeParameters(dims, initializeMultiplier)
    costs = []
    descendingGraph = True

    # For each training iteration
    for i in range(0, numIterations + 1):

        # Forward propagation
        cache = forwardPropagation(data, params)

        # Cost function
        cost = calculateCost(labels, params, cache, lamb)

        # Backward  propagation
        grads = backwardPropagation(data, labels, params, cache, lamb)

        # Gradient descent parameter update
        params = updateParams(params, grads, learningRate)

        # Print the cost every N number of iterations
        if printCost and i % 500 == 0:
            print ("Cost after iteration", str(i), "is", str(cost))

        # Record the cost every N number of iterations
        if i % 500 == 0:
            if (len(costs) != 0) and (cost > costs[-1]):
                descendingGraph = False
            costs.append(cost)

    # Print the model training cost graph
    if showGraph:
        _costs = np.squeeze(costs)
        plt.plot(_costs)
        plt.ylabel('Cost')
        plt.xlabel('Iterations (every 100)')
        plt.title("Learning rate =" + str(learningRate))
        plt.show()

    return params, costs, descendingGraph
```


```python
# Utilize the model's trained params to make predictions
def predict(data, params, trueLabels):
    # Apply the training weights and the sigmoid activation to the inputs
    cache = forwardPropagation(data, params)

    # Classify anything with a probability of greater than 0.5 to a 1 (i.e. cat) classification
    predictions = (cache["a2"] > 0.5)
    accuracy = 100 - np.mean(np.abs(predictions - trueLabels)) * 100

    preds = {"predictions" : predictions, "accuracy": accuracy}

    return preds
```

# Model Training with Variable Hyperparameters

It's finally time to test the model with a number of hyperparameter configurations, and we'll see if we can find a combination of hyperparameters that optimizes and improves on the classification prediction rate.  

For reference here is what we achieved without model tuning:

```bash
Train accuracy: 90.625
Test accuracy: 62.0
```

We'll start with 80 models utilizing a smaller learning rate with more and less iterations, and then we'll take a look at another 80 models having a larger learning rate again with more and less iterations.  Hopefully one or more of the 160 models will have good results, and we can then look at its F-Score for comparison to the logistic regression model we generated in the [last write-up]({% post_url 2019-07-28-machine-learning-image-classification-part-one %}).

For each model the hyperparameters will be generated randomly from a defined range.  This should help us to quickly explore a number of combinations without having to hand-craft each one.

## Smaller learning rate training


```python
h1 = {
    "Epoch": None,
    "HL_Units": [2, 8, 1],
    "HL_Size": 1,
    "Iterations": [750, 2000, 50],
    "Learning_Rate": [0, .01, .001],
    "Lambda": [0, 2, .1],
    "Weight_Multi": [0, 0.001, .0001],
    "Train_Acc": None,
    "Test_Acc": None,
    "Final_Cost": None,
    "Descending_Graph": True
}

r1, p1, c1 = runModels(h1, 40, True)
display(HTML(r1.to_html()))
```


    *** Starting model training
    *** Done!




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epoch</th>
      <th>HL_Units</th>
      <th>HL_Size</th>
      <th>Iterations</th>
      <th>Learning_Rate</th>
      <th>Lambda</th>
      <th>Weight_Multi</th>
      <th>Train_Acc</th>
      <th>Test_Acc</th>
      <th>Final_Cost</th>
      <th>Descending_Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>8</td>
      <td>1</td>
      <td>1150</td>
      <td>0.009</td>
      <td>0.9</td>
      <td>0.0009</td>
      <td>77.750</td>
      <td>69.5</td>
      <td>0.567576</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>5</td>
      <td>1</td>
      <td>1650</td>
      <td>0.006</td>
      <td>0.8</td>
      <td>0.0006</td>
      <td>71.875</td>
      <td>68.0</td>
      <td>0.510398</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>4</td>
      <td>1</td>
      <td>1500</td>
      <td>0.007</td>
      <td>1.7</td>
      <td>0.0008</td>
      <td>72.250</td>
      <td>68.0</td>
      <td>0.531841</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>8</td>
      <td>1</td>
      <td>1100</td>
      <td>0.010</td>
      <td>1.9</td>
      <td>0.0008</td>
      <td>78.125</td>
      <td>68.0</td>
      <td>0.595439</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>6</td>
      <td>1</td>
      <td>1650</td>
      <td>0.010</td>
      <td>1.2</td>
      <td>0.0009</td>
      <td>85.500</td>
      <td>67.0</td>
      <td>0.515924</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>1150</td>
      <td>0.007</td>
      <td>0.8</td>
      <td>0.0001</td>
      <td>73.625</td>
      <td>66.5</td>
      <td>0.608160</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>6</td>
      <td>1</td>
      <td>2000</td>
      <td>0.006</td>
      <td>1.1</td>
      <td>0.0007</td>
      <td>74.500</td>
      <td>66.5</td>
      <td>0.488919</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>4</td>
      <td>1</td>
      <td>800</td>
      <td>0.009</td>
      <td>0.2</td>
      <td>0.0006</td>
      <td>72.875</td>
      <td>66.0</td>
      <td>0.658947</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1100</td>
      <td>0.008</td>
      <td>1.6</td>
      <td>0.0005</td>
      <td>70.000</td>
      <td>65.5</td>
      <td>0.564497</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>8</td>
      <td>1</td>
      <td>1250</td>
      <td>0.006</td>
      <td>0.5</td>
      <td>0.0008</td>
      <td>74.500</td>
      <td>65.5</td>
      <td>0.594659</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>3</td>
      <td>1</td>
      <td>1550</td>
      <td>0.005</td>
      <td>2.0</td>
      <td>0.0002</td>
      <td>73.250</td>
      <td>65.5</td>
      <td>0.592611</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>1500</td>
      <td>0.010</td>
      <td>1.3</td>
      <td>0.0001</td>
      <td>75.625</td>
      <td>65.0</td>
      <td>0.563984</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>1650</td>
      <td>0.008</td>
      <td>1.4</td>
      <td>0.0008</td>
      <td>82.000</td>
      <td>64.5</td>
      <td>0.552842</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>5</td>
      <td>1</td>
      <td>1950</td>
      <td>0.005</td>
      <td>1.5</td>
      <td>0.0007</td>
      <td>73.000</td>
      <td>64.5</td>
      <td>0.559365</td>
      <td>True</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>850</td>
      <td>0.007</td>
      <td>1.6</td>
      <td>0.0010</td>
      <td>70.625</td>
      <td>64.0</td>
      <td>0.670679</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>6</td>
      <td>1</td>
      <td>1250</td>
      <td>0.005</td>
      <td>1.5</td>
      <td>0.0003</td>
      <td>69.750</td>
      <td>64.0</td>
      <td>0.647818</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>1800</td>
      <td>0.010</td>
      <td>1.1</td>
      <td>0.0009</td>
      <td>81.875</td>
      <td>64.0</td>
      <td>0.533753</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>6</td>
      <td>1</td>
      <td>1050</td>
      <td>0.004</td>
      <td>1.2</td>
      <td>0.0007</td>
      <td>65.250</td>
      <td>64.0</td>
      <td>0.665525</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>4</td>
      <td>1</td>
      <td>950</td>
      <td>0.010</td>
      <td>1.5</td>
      <td>0.0008</td>
      <td>74.375</td>
      <td>64.0</td>
      <td>0.642413</td>
      <td>True</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>8</td>
      <td>1</td>
      <td>1150</td>
      <td>0.006</td>
      <td>2.0</td>
      <td>0.0001</td>
      <td>70.250</td>
      <td>64.0</td>
      <td>0.624042</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>7</td>
      <td>1</td>
      <td>2000</td>
      <td>0.002</td>
      <td>1.0</td>
      <td>0.0003</td>
      <td>62.875</td>
      <td>64.0</td>
      <td>0.670351</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>6</td>
      <td>1</td>
      <td>1500</td>
      <td>0.004</td>
      <td>0.9</td>
      <td>0.0003</td>
      <td>69.125</td>
      <td>63.5</td>
      <td>0.615967</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>7</td>
      <td>1</td>
      <td>1600</td>
      <td>0.003</td>
      <td>0.4</td>
      <td>0.0005</td>
      <td>66.500</td>
      <td>63.5</td>
      <td>0.651325</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>6</td>
      <td>1</td>
      <td>1700</td>
      <td>0.003</td>
      <td>1.3</td>
      <td>0.0008</td>
      <td>67.875</td>
      <td>63.5</td>
      <td>0.650475</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>3</td>
      <td>1</td>
      <td>950</td>
      <td>0.005</td>
      <td>1.4</td>
      <td>0.0004</td>
      <td>65.500</td>
      <td>63.5</td>
      <td>0.690524</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>5</td>
      <td>1</td>
      <td>1750</td>
      <td>0.003</td>
      <td>0.4</td>
      <td>0.0002</td>
      <td>66.500</td>
      <td>62.5</td>
      <td>0.667028</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>6</td>
      <td>1</td>
      <td>1550</td>
      <td>0.003</td>
      <td>1.8</td>
      <td>0.0008</td>
      <td>66.750</td>
      <td>62.5</td>
      <td>0.650610</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>1000</td>
      <td>0.010</td>
      <td>1.2</td>
      <td>0.0007</td>
      <td>75.250</td>
      <td>61.5</td>
      <td>0.513657</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>6</td>
      <td>1</td>
      <td>1250</td>
      <td>0.003</td>
      <td>0.6</td>
      <td>0.0003</td>
      <td>61.500</td>
      <td>59.5</td>
      <td>0.687296</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>6</td>
      <td>1</td>
      <td>1750</td>
      <td>0.008</td>
      <td>0.3</td>
      <td>0.0003</td>
      <td>63.750</td>
      <td>59.5</td>
      <td>0.472106</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>6</td>
      <td>1</td>
      <td>1850</td>
      <td>0.008</td>
      <td>1.3</td>
      <td>0.0003</td>
      <td>64.875</td>
      <td>59.0</td>
      <td>0.468910</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>900</td>
      <td>0.004</td>
      <td>1.2</td>
      <td>0.0001</td>
      <td>54.000</td>
      <td>55.0</td>
      <td>0.692530</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>7</td>
      <td>1</td>
      <td>800</td>
      <td>0.004</td>
      <td>0.5</td>
      <td>0.0004</td>
      <td>56.875</td>
      <td>54.5</td>
      <td>0.691068</td>
      <td>True</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>8</td>
      <td>1</td>
      <td>1250</td>
      <td>0.002</td>
      <td>0.2</td>
      <td>0.0010</td>
      <td>53.250</td>
      <td>54.5</td>
      <td>0.689499</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>7</td>
      <td>1</td>
      <td>900</td>
      <td>0.003</td>
      <td>1.3</td>
      <td>0.0010</td>
      <td>55.875</td>
      <td>54.0</td>
      <td>0.691434</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>1050</td>
      <td>0.001</td>
      <td>1.7</td>
      <td>0.0008</td>
      <td>50.500</td>
      <td>48.0</td>
      <td>0.692637</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>5</td>
      <td>1</td>
      <td>1150</td>
      <td>0.002</td>
      <td>0.9</td>
      <td>0.0008</td>
      <td>51.250</td>
      <td>48.0</td>
      <td>0.690641</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>3</td>
      <td>1</td>
      <td>1450</td>
      <td>0.001</td>
      <td>0.1</td>
      <td>0.0008</td>
      <td>50.500</td>
      <td>48.0</td>
      <td>0.692755</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>7</td>
      <td>1</td>
      <td>900</td>
      <td>0.001</td>
      <td>1.2</td>
      <td>0.0002</td>
      <td>50.500</td>
      <td>48.0</td>
      <td>0.693133</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>5</td>
      <td>1</td>
      <td>1800</td>
      <td>0.009</td>
      <td>1.7</td>
      <td>0.0003</td>
      <td>84.750</td>
      <td>65.5</td>
      <td>0.625315</td>
      <td>False</td>
    </tr>
  </tbody>
</table>



```python
printCostGraphs(c1, list(r1.iloc[0:8, 0]), 4, (10,20))
```


![png](assets/images/posts/2019/output_30_0.png)


## Smaller learning rate; more iterations


```python
h3 = {
    "Epoch": None,
    "HL_Units": [2, 8, 1],
    "HL_Size": 1,
    "Iterations": [2000, 10000, 500],
    "Learning_Rate": [0, .01, .001],
    "Lambda": [0, 2, .1],
    "Weight_Multi": [0, 0.001, .0001],
    "Train_Acc": None,
    "Test_Acc": None,
    "Final_Cost": None,
    "Descending_Graph": True
}

r3, p3, c3 = runModels(h3, 40, True)
display(HTML(r3.to_html()))
```


    *** Starting model training
    *** Done!




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epoch</th>
      <th>HL_Units</th>
      <th>HL_Size</th>
      <th>Iterations</th>
      <th>Learning_Rate</th>
      <th>Lambda</th>
      <th>Weight_Multi</th>
      <th>Train_Acc</th>
      <th>Test_Acc</th>
      <th>Final_Cost</th>
      <th>Descending_Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>7</td>
      <td>1</td>
      <td>3000</td>
      <td>0.007</td>
      <td>2.0</td>
      <td>0.0002</td>
      <td>90.250</td>
      <td>70.0</td>
      <td>0.349669</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>4500</td>
      <td>0.005</td>
      <td>0.7</td>
      <td>0.0004</td>
      <td>86.875</td>
      <td>66.5</td>
      <td>0.314388</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>8</td>
      <td>1</td>
      <td>3500</td>
      <td>0.002</td>
      <td>0.3</td>
      <td>0.0006</td>
      <td>72.625</td>
      <td>66.5</td>
      <td>0.564334</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>6</td>
      <td>1</td>
      <td>2500</td>
      <td>0.003</td>
      <td>0.9</td>
      <td>0.0002</td>
      <td>72.625</td>
      <td>66.0</td>
      <td>0.572031</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>8</td>
      <td>1</td>
      <td>3000</td>
      <td>0.003</td>
      <td>0.2</td>
      <td>0.0001</td>
      <td>77.250</td>
      <td>66.0</td>
      <td>0.521390</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>5500</td>
      <td>0.005</td>
      <td>1.2</td>
      <td>0.0002</td>
      <td>92.625</td>
      <td>64.0</td>
      <td>0.237177</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>4</td>
      <td>1</td>
      <td>2500</td>
      <td>0.002</td>
      <td>1.3</td>
      <td>0.0001</td>
      <td>65.000</td>
      <td>64.0</td>
      <td>0.664944</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>6500</td>
      <td>0.005</td>
      <td>0.6</td>
      <td>0.0002</td>
      <td>95.125</td>
      <td>64.0</td>
      <td>0.189869</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>7</td>
      <td>1</td>
      <td>5000</td>
      <td>0.001</td>
      <td>1.3</td>
      <td>0.0010</td>
      <td>67.625</td>
      <td>63.5</td>
      <td>0.626675</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>3</td>
      <td>1</td>
      <td>4000</td>
      <td>0.001</td>
      <td>0.2</td>
      <td>0.0003</td>
      <td>63.125</td>
      <td>63.0</td>
      <td>0.677799</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>4</td>
      <td>1</td>
      <td>3000</td>
      <td>0.001</td>
      <td>2.0</td>
      <td>0.0001</td>
      <td>50.625</td>
      <td>48.0</td>
      <td>0.690490</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>5</td>
      <td>1</td>
      <td>4000</td>
      <td>0.008</td>
      <td>1.0</td>
      <td>0.0008</td>
      <td>95.875</td>
      <td>69.5</td>
      <td>0.195421</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>5000</td>
      <td>0.010</td>
      <td>1.4</td>
      <td>0.0003</td>
      <td>97.625</td>
      <td>68.5</td>
      <td>0.151293</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>7</td>
      <td>1</td>
      <td>4500</td>
      <td>0.008</td>
      <td>0.6</td>
      <td>0.0006</td>
      <td>98.125</td>
      <td>68.5</td>
      <td>0.148196</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>4</td>
      <td>1</td>
      <td>3000</td>
      <td>0.008</td>
      <td>0.2</td>
      <td>0.0006</td>
      <td>89.000</td>
      <td>68.5</td>
      <td>0.535870</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>6000</td>
      <td>0.009</td>
      <td>1.0</td>
      <td>0.0001</td>
      <td>99.625</td>
      <td>67.5</td>
      <td>0.081284</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>5000</td>
      <td>0.007</td>
      <td>0.8</td>
      <td>0.0002</td>
      <td>96.000</td>
      <td>67.5</td>
      <td>0.176135</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>8</td>
      <td>1</td>
      <td>5000</td>
      <td>0.010</td>
      <td>1.4</td>
      <td>0.0001</td>
      <td>97.000</td>
      <td>67.0</td>
      <td>0.181041</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>6</td>
      <td>1</td>
      <td>9500</td>
      <td>0.008</td>
      <td>1.2</td>
      <td>0.0002</td>
      <td>100.000</td>
      <td>66.5</td>
      <td>0.047833</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>8</td>
      <td>1</td>
      <td>5500</td>
      <td>0.009</td>
      <td>0.1</td>
      <td>0.0004</td>
      <td>99.750</td>
      <td>66.5</td>
      <td>0.065625</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>7</td>
      <td>1</td>
      <td>10000</td>
      <td>0.003</td>
      <td>0.2</td>
      <td>0.0008</td>
      <td>99.125</td>
      <td>66.5</td>
      <td>0.101235</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>4</td>
      <td>1</td>
      <td>10000</td>
      <td>0.009</td>
      <td>1.2</td>
      <td>0.0010</td>
      <td>99.625</td>
      <td>66.5</td>
      <td>0.054953</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>7</td>
      <td>1</td>
      <td>8500</td>
      <td>0.003</td>
      <td>1.1</td>
      <td>0.0006</td>
      <td>97.250</td>
      <td>66.0</td>
      <td>0.156339</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>4000</td>
      <td>0.007</td>
      <td>1.0</td>
      <td>0.0005</td>
      <td>93.250</td>
      <td>65.5</td>
      <td>0.226588</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>7</td>
      <td>1</td>
      <td>10000</td>
      <td>0.002</td>
      <td>1.9</td>
      <td>0.0007</td>
      <td>91.500</td>
      <td>65.5</td>
      <td>0.243441</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>3</td>
      <td>1</td>
      <td>8500</td>
      <td>0.002</td>
      <td>0.1</td>
      <td>0.0007</td>
      <td>88.500</td>
      <td>65.5</td>
      <td>0.317150</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>7</td>
      <td>1</td>
      <td>6500</td>
      <td>0.009</td>
      <td>1.4</td>
      <td>0.0009</td>
      <td>99.750</td>
      <td>65.5</td>
      <td>0.070478</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>5</td>
      <td>1</td>
      <td>9000</td>
      <td>0.004</td>
      <td>1.6</td>
      <td>0.0010</td>
      <td>99.125</td>
      <td>65.5</td>
      <td>0.103912</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>4000</td>
      <td>0.006</td>
      <td>1.7</td>
      <td>0.0010</td>
      <td>90.125</td>
      <td>65.0</td>
      <td>0.248698</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>5</td>
      <td>1</td>
      <td>7500</td>
      <td>0.008</td>
      <td>0.6</td>
      <td>0.0001</td>
      <td>99.875</td>
      <td>65.0</td>
      <td>0.059867</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>4</td>
      <td>1</td>
      <td>8500</td>
      <td>0.010</td>
      <td>1.8</td>
      <td>0.0010</td>
      <td>97.625</td>
      <td>65.0</td>
      <td>0.160802</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>7</td>
      <td>1</td>
      <td>9000</td>
      <td>0.009</td>
      <td>0.1</td>
      <td>0.0003</td>
      <td>99.750</td>
      <td>65.0</td>
      <td>0.025635</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>6</td>
      <td>1</td>
      <td>9500</td>
      <td>0.006</td>
      <td>0.6</td>
      <td>0.0001</td>
      <td>100.000</td>
      <td>65.0</td>
      <td>0.046617</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>7</td>
      <td>1</td>
      <td>7000</td>
      <td>0.003</td>
      <td>0.4</td>
      <td>0.0009</td>
      <td>90.250</td>
      <td>65.0</td>
      <td>0.270268</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>3</td>
      <td>1</td>
      <td>7000</td>
      <td>0.010</td>
      <td>1.1</td>
      <td>0.0001</td>
      <td>89.125</td>
      <td>64.5</td>
      <td>0.233201</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>3</td>
      <td>1</td>
      <td>9000</td>
      <td>0.006</td>
      <td>0.6</td>
      <td>0.0001</td>
      <td>99.750</td>
      <td>64.0</td>
      <td>0.072092</td>
      <td>False</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>7</td>
      <td>1</td>
      <td>8500</td>
      <td>0.003</td>
      <td>1.6</td>
      <td>0.0005</td>
      <td>96.250</td>
      <td>64.0</td>
      <td>0.169106</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>6</td>
      <td>1</td>
      <td>5000</td>
      <td>0.003</td>
      <td>0.1</td>
      <td>0.0007</td>
      <td>83.875</td>
      <td>62.5</td>
      <td>0.390400</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>8500</td>
      <td>0.002</td>
      <td>1.3</td>
      <td>0.0002</td>
      <td>88.000</td>
      <td>61.0</td>
      <td>0.320435</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>4</td>
      <td>1</td>
      <td>7000</td>
      <td>0.007</td>
      <td>0.5</td>
      <td>0.0001</td>
      <td>82.250</td>
      <td>58.0</td>
      <td>0.186820</td>
      <td>False</td>
    </tr>
  </tbody>
</table>



```python
printCostGraphs(c3, list(r3.iloc[0:8, 0]), 4, (10,20))
```


![png](assets/images/posts/2019/output_33_0.png)


## Larger learning rate training


```python
h2 = {
    "Epoch": None,
    "HL_Units": [2, 8, 1],
    "HL_Size": 1,
    "Iterations": [750, 2000, 50],
    "Learning_Rate": [0, .1, .01],
    "Lambda": [0, 2, .1],
    "Weight_Multi": [0, 0.001, .0001],
    "Train_Acc": None,
    "Test_Acc": None,
    "Final_Cost": None,
    "Descending_Graph": True
}

r2, p2, c2 = runModels(h2, 40, True)
display(HTML(r2.to_html()))
```


    *** Starting model training
    *** Done!




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epoch</th>
      <th>HL_Units</th>
      <th>HL_Size</th>
      <th>Iterations</th>
      <th>Learning_Rate</th>
      <th>Lambda</th>
      <th>Weight_Multi</th>
      <th>Train_Acc</th>
      <th>Test_Acc</th>
      <th>Final_Cost</th>
      <th>Descending_Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1100</td>
      <td>0.06</td>
      <td>0.6</td>
      <td>0.0002</td>
      <td>83.375</td>
      <td>73.0</td>
      <td>0.350286</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>7</td>
      <td>1</td>
      <td>1450</td>
      <td>0.06</td>
      <td>1.8</td>
      <td>0.0007</td>
      <td>85.375</td>
      <td>72.5</td>
      <td>0.412003</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>1250</td>
      <td>0.08</td>
      <td>1.7</td>
      <td>0.0007</td>
      <td>82.625</td>
      <td>71.0</td>
      <td>0.478089</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>5</td>
      <td>1</td>
      <td>1200</td>
      <td>0.06</td>
      <td>1.5</td>
      <td>0.0007</td>
      <td>79.250</td>
      <td>70.5</td>
      <td>0.392045</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>7</td>
      <td>1</td>
      <td>1950</td>
      <td>0.05</td>
      <td>0.1</td>
      <td>0.0010</td>
      <td>90.500</td>
      <td>70.5</td>
      <td>0.311047</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>5</td>
      <td>1</td>
      <td>1800</td>
      <td>0.06</td>
      <td>0.3</td>
      <td>0.0007</td>
      <td>86.500</td>
      <td>70.5</td>
      <td>0.433791</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>5</td>
      <td>1</td>
      <td>900</td>
      <td>0.07</td>
      <td>1.5</td>
      <td>0.0010</td>
      <td>79.375</td>
      <td>70.5</td>
      <td>0.569526</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>8</td>
      <td>1</td>
      <td>1700</td>
      <td>0.06</td>
      <td>0.6</td>
      <td>0.0005</td>
      <td>91.000</td>
      <td>70.5</td>
      <td>0.327656</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>7</td>
      <td>1</td>
      <td>1350</td>
      <td>0.05</td>
      <td>2.0</td>
      <td>0.0003</td>
      <td>86.375</td>
      <td>70.0</td>
      <td>0.392760</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>3</td>
      <td>1</td>
      <td>1150</td>
      <td>0.10</td>
      <td>0.3</td>
      <td>0.0007</td>
      <td>82.000</td>
      <td>69.5</td>
      <td>0.447228</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>8</td>
      <td>1</td>
      <td>1900</td>
      <td>0.10</td>
      <td>0.2</td>
      <td>0.0001</td>
      <td>86.625</td>
      <td>69.0</td>
      <td>0.380536</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>7</td>
      <td>1</td>
      <td>1350</td>
      <td>0.07</td>
      <td>1.1</td>
      <td>0.0010</td>
      <td>85.750</td>
      <td>69.0</td>
      <td>0.426419</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>8</td>
      <td>1</td>
      <td>1000</td>
      <td>0.01</td>
      <td>0.6</td>
      <td>0.0004</td>
      <td>71.125</td>
      <td>68.0</td>
      <td>0.516636</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>1100</td>
      <td>0.07</td>
      <td>0.7</td>
      <td>0.0002</td>
      <td>87.125</td>
      <td>67.5</td>
      <td>0.505792</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>1</td>
      <td>1850</td>
      <td>0.09</td>
      <td>0.7</td>
      <td>0.0005</td>
      <td>77.500</td>
      <td>66.5</td>
      <td>0.455560</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>5</td>
      <td>1</td>
      <td>850</td>
      <td>0.03</td>
      <td>1.5</td>
      <td>0.0008</td>
      <td>72.750</td>
      <td>66.0</td>
      <td>0.534921</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>1750</td>
      <td>0.07</td>
      <td>1.3</td>
      <td>0.0004</td>
      <td>71.000</td>
      <td>65.0</td>
      <td>0.375173</td>
      <td>True</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>4</td>
      <td>1</td>
      <td>850</td>
      <td>0.10</td>
      <td>0.6</td>
      <td>0.0002</td>
      <td>69.750</td>
      <td>63.5</td>
      <td>0.585656</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>4</td>
      <td>1</td>
      <td>1050</td>
      <td>0.05</td>
      <td>0.4</td>
      <td>0.0009</td>
      <td>73.875</td>
      <td>63.0</td>
      <td>0.485734</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>5</td>
      <td>1</td>
      <td>1050</td>
      <td>0.01</td>
      <td>1.8</td>
      <td>0.0009</td>
      <td>62.625</td>
      <td>61.0</td>
      <td>0.523185</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>3</td>
      <td>1</td>
      <td>1750</td>
      <td>0.01</td>
      <td>0.6</td>
      <td>0.0005</td>
      <td>71.125</td>
      <td>61.0</td>
      <td>0.527912</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>1350</td>
      <td>0.02</td>
      <td>0.1</td>
      <td>0.0004</td>
      <td>66.375</td>
      <td>60.0</td>
      <td>0.435040</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>6</td>
      <td>1</td>
      <td>1800</td>
      <td>0.06</td>
      <td>1.2</td>
      <td>0.0005</td>
      <td>76.000</td>
      <td>59.5</td>
      <td>0.384182</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>3</td>
      <td>1</td>
      <td>1400</td>
      <td>0.10</td>
      <td>1.4</td>
      <td>0.0002</td>
      <td>58.875</td>
      <td>59.5</td>
      <td>0.495893</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>4</td>
      <td>1</td>
      <td>1600</td>
      <td>0.10</td>
      <td>1.1</td>
      <td>0.0001</td>
      <td>54.375</td>
      <td>55.0</td>
      <td>0.573383</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>6</td>
      <td>1</td>
      <td>1400</td>
      <td>0.04</td>
      <td>0.4</td>
      <td>0.0007</td>
      <td>86.875</td>
      <td>74.0</td>
      <td>0.550969</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>8</td>
      <td>1</td>
      <td>1050</td>
      <td>0.03</td>
      <td>0.4</td>
      <td>0.0010</td>
      <td>83.875</td>
      <td>72.5</td>
      <td>0.628924</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>7</td>
      <td>1</td>
      <td>1500</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.0003</td>
      <td>80.375</td>
      <td>72.5</td>
      <td>0.447564</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>1550</td>
      <td>0.03</td>
      <td>1.2</td>
      <td>0.0010</td>
      <td>89.250</td>
      <td>71.5</td>
      <td>0.477405</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>4</td>
      <td>1</td>
      <td>1800</td>
      <td>0.05</td>
      <td>1.5</td>
      <td>0.0005</td>
      <td>89.375</td>
      <td>71.0</td>
      <td>0.349060</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>2000</td>
      <td>0.07</td>
      <td>0.3</td>
      <td>0.0001</td>
      <td>86.875</td>
      <td>69.5</td>
      <td>0.448897</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>1400</td>
      <td>0.04</td>
      <td>1.1</td>
      <td>0.0010</td>
      <td>86.750</td>
      <td>69.0</td>
      <td>0.564314</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>1500</td>
      <td>0.08</td>
      <td>1.9</td>
      <td>0.0008</td>
      <td>76.625</td>
      <td>68.5</td>
      <td>0.612455</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>4</td>
      <td>1</td>
      <td>1400</td>
      <td>0.08</td>
      <td>1.1</td>
      <td>0.0001</td>
      <td>75.500</td>
      <td>68.0</td>
      <td>0.578355</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>5</td>
      <td>1</td>
      <td>2000</td>
      <td>0.04</td>
      <td>1.6</td>
      <td>0.0003</td>
      <td>86.500</td>
      <td>67.5</td>
      <td>0.372239</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>7</td>
      <td>1</td>
      <td>1550</td>
      <td>0.09</td>
      <td>0.9</td>
      <td>0.0007</td>
      <td>86.875</td>
      <td>66.0</td>
      <td>0.422013</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>4</td>
      <td>1</td>
      <td>1750</td>
      <td>0.07</td>
      <td>1.9</td>
      <td>0.0002</td>
      <td>84.000</td>
      <td>65.5</td>
      <td>0.470529</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>3</td>
      <td>1</td>
      <td>1900</td>
      <td>0.09</td>
      <td>1.2</td>
      <td>0.0007</td>
      <td>67.250</td>
      <td>62.0</td>
      <td>0.455809</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>5</td>
      <td>1</td>
      <td>1450</td>
      <td>0.06</td>
      <td>1.8</td>
      <td>0.0009</td>
      <td>63.250</td>
      <td>60.0</td>
      <td>0.496687</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>4</td>
      <td>1</td>
      <td>1700</td>
      <td>0.10</td>
      <td>1.5</td>
      <td>0.0010</td>
      <td>65.750</td>
      <td>59.0</td>
      <td>0.600045</td>
      <td>False</td>
    </tr>
  </tbody>
</table>



```python
printCostGraphs(c2, list(r2.iloc[0:8, 0]), 4, (10,20))
```


![png](assets/images/posts/2019/output_36_0.png)


## Larger learning rate; more iterations


```python
h4 = {
    "Epoch": None,
    "HL_Units": [2, 8, 1],
    "HL_Size": 1,
    "Iterations": [2000, 10000, 500],
    "Learning_Rate": [0, .1, .01],
    "Lambda": [0, 2, .1],
    "Weight_Multi": [0, 0.001, .0001],
    "Train_Acc": None,
    "Test_Acc": None,
    "Final_Cost": None,
    "Descending_Graph": True
}

r4, p4, c4 = runModels(h4, 40, True)
display(HTML(r4.to_html()))
```


    *** Starting model training
    *** Done!




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Epoch</th>
      <th>HL_Units</th>
      <th>HL_Size</th>
      <th>Iterations</th>
      <th>Learning_Rate</th>
      <th>Lambda</th>
      <th>Weight_Multi</th>
      <th>Train_Acc</th>
      <th>Test_Acc</th>
      <th>Final_Cost</th>
      <th>Descending_Graph</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>32</td>
      <td>6</td>
      <td>1</td>
      <td>3000</td>
      <td>0.02</td>
      <td>1.6</td>
      <td>0.0006</td>
      <td>98.250</td>
      <td>68.5</td>
      <td>0.151224</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>2500</td>
      <td>0.08</td>
      <td>0.6</td>
      <td>0.0003</td>
      <td>79.500</td>
      <td>67.0</td>
      <td>0.402815</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>8</td>
      <td>1</td>
      <td>4500</td>
      <td>0.08</td>
      <td>0.6</td>
      <td>0.0004</td>
      <td>97.750</td>
      <td>71.5</td>
      <td>0.146351</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>7</td>
      <td>1</td>
      <td>3500</td>
      <td>0.08</td>
      <td>0.2</td>
      <td>0.0009</td>
      <td>85.750</td>
      <td>70.5</td>
      <td>0.410727</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>8500</td>
      <td>0.09</td>
      <td>1.8</td>
      <td>0.0004</td>
      <td>93.625</td>
      <td>70.0</td>
      <td>0.413673</td>
      <td>False</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>3</td>
      <td>1</td>
      <td>5000</td>
      <td>0.07</td>
      <td>1.0</td>
      <td>0.0003</td>
      <td>96.875</td>
      <td>69.5</td>
      <td>0.193465</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>5</td>
      <td>1</td>
      <td>3000</td>
      <td>0.03</td>
      <td>0.8</td>
      <td>0.0002</td>
      <td>96.000</td>
      <td>69.5</td>
      <td>0.213647</td>
      <td>False</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>6</td>
      <td>1</td>
      <td>9000</td>
      <td>0.08</td>
      <td>1.2</td>
      <td>0.0002</td>
      <td>95.250</td>
      <td>69.5</td>
      <td>0.237999</td>
      <td>False</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>4</td>
      <td>1</td>
      <td>2500</td>
      <td>0.03</td>
      <td>1.5</td>
      <td>0.0006</td>
      <td>90.000</td>
      <td>69.0</td>
      <td>0.264472</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>3</td>
      <td>1</td>
      <td>6000</td>
      <td>0.01</td>
      <td>0.1</td>
      <td>0.0003</td>
      <td>95.875</td>
      <td>69.0</td>
      <td>0.238603</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>6</td>
      <td>1</td>
      <td>2500</td>
      <td>0.08</td>
      <td>1.5</td>
      <td>0.0010</td>
      <td>89.625</td>
      <td>68.5</td>
      <td>0.367724</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>39</td>
      <td>5</td>
      <td>1</td>
      <td>10000</td>
      <td>0.02</td>
      <td>2.0</td>
      <td>0.0002</td>
      <td>98.500</td>
      <td>68.5</td>
      <td>0.127511</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>8</td>
      <td>1</td>
      <td>7000</td>
      <td>0.09</td>
      <td>1.9</td>
      <td>0.0001</td>
      <td>82.125</td>
      <td>68.0</td>
      <td>0.376518</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>6</td>
      <td>1</td>
      <td>10000</td>
      <td>0.10</td>
      <td>1.5</td>
      <td>0.0007</td>
      <td>88.000</td>
      <td>68.0</td>
      <td>0.465391</td>
      <td>False</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>7</td>
      <td>1</td>
      <td>2500</td>
      <td>0.09</td>
      <td>1.9</td>
      <td>0.0001</td>
      <td>84.875</td>
      <td>68.0</td>
      <td>0.469647</td>
      <td>False</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35</td>
      <td>4</td>
      <td>1</td>
      <td>2500</td>
      <td>0.10</td>
      <td>1.4</td>
      <td>0.0001</td>
      <td>86.250</td>
      <td>68.0</td>
      <td>0.342725</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>6000</td>
      <td>0.03</td>
      <td>1.3</td>
      <td>0.0004</td>
      <td>99.250</td>
      <td>67.5</td>
      <td>0.069514</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>3500</td>
      <td>0.07</td>
      <td>1.5</td>
      <td>0.0002</td>
      <td>81.375</td>
      <td>67.0</td>
      <td>0.603899</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>5</td>
      <td>1</td>
      <td>6500</td>
      <td>0.03</td>
      <td>0.2</td>
      <td>0.0008</td>
      <td>98.250</td>
      <td>67.0</td>
      <td>0.098654</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>6</td>
      <td>1</td>
      <td>8000</td>
      <td>0.05</td>
      <td>0.1</td>
      <td>0.0003</td>
      <td>96.000</td>
      <td>66.5</td>
      <td>0.175305</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37</td>
      <td>6</td>
      <td>1</td>
      <td>4000</td>
      <td>0.07</td>
      <td>0.8</td>
      <td>0.0004</td>
      <td>86.250</td>
      <td>66.0</td>
      <td>0.554717</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>6</td>
      <td>1</td>
      <td>8000</td>
      <td>0.06</td>
      <td>0.1</td>
      <td>0.0006</td>
      <td>98.125</td>
      <td>65.0</td>
      <td>0.075860</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7</td>
      <td>1</td>
      <td>4500</td>
      <td>0.10</td>
      <td>1.9</td>
      <td>0.0009</td>
      <td>84.875</td>
      <td>64.5</td>
      <td>0.514395</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>3</td>
      <td>1</td>
      <td>6500</td>
      <td>0.10</td>
      <td>1.6</td>
      <td>0.0002</td>
      <td>82.000</td>
      <td>64.5</td>
      <td>0.460273</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>9000</td>
      <td>0.02</td>
      <td>0.4</td>
      <td>0.0008</td>
      <td>99.875</td>
      <td>64.0</td>
      <td>0.032819</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30</td>
      <td>7</td>
      <td>1</td>
      <td>10000</td>
      <td>0.05</td>
      <td>1.2</td>
      <td>0.0006</td>
      <td>87.750</td>
      <td>64.0</td>
      <td>0.369827</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>8</td>
      <td>1</td>
      <td>5000</td>
      <td>0.03</td>
      <td>1.6</td>
      <td>0.0006</td>
      <td>87.250</td>
      <td>64.0</td>
      <td>0.413589</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34</td>
      <td>4</td>
      <td>1</td>
      <td>5500</td>
      <td>0.09</td>
      <td>1.1</td>
      <td>0.0005</td>
      <td>80.125</td>
      <td>62.0</td>
      <td>0.415828</td>
      <td>False</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>9000</td>
      <td>0.05</td>
      <td>0.6</td>
      <td>0.0003</td>
      <td>82.375</td>
      <td>60.5</td>
      <td>0.529366</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>4</td>
      <td>1</td>
      <td>3500</td>
      <td>0.07</td>
      <td>1.8</td>
      <td>0.0002</td>
      <td>75.000</td>
      <td>60.5</td>
      <td>0.481597</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>10000</td>
      <td>0.07</td>
      <td>0.7</td>
      <td>0.0002</td>
      <td>59.750</td>
      <td>59.5</td>
      <td>0.698914</td>
      <td>False</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31</td>
      <td>7</td>
      <td>1</td>
      <td>9000</td>
      <td>0.06</td>
      <td>2.0</td>
      <td>0.0005</td>
      <td>79.375</td>
      <td>59.0</td>
      <td>0.538237</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>6</td>
      <td>1</td>
      <td>4000</td>
      <td>0.07</td>
      <td>1.6</td>
      <td>0.0004</td>
      <td>58.250</td>
      <td>58.5</td>
      <td>0.774961</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>6</td>
      <td>1</td>
      <td>7000</td>
      <td>0.08</td>
      <td>0.6</td>
      <td>0.0002</td>
      <td>70.250</td>
      <td>58.0</td>
      <td>0.627695</td>
      <td>False</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>7</td>
      <td>1</td>
      <td>8000</td>
      <td>0.08</td>
      <td>1.0</td>
      <td>0.0003</td>
      <td>69.750</td>
      <td>58.0</td>
      <td>0.640571</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>8500</td>
      <td>0.04</td>
      <td>0.8</td>
      <td>0.0006</td>
      <td>56.250</td>
      <td>55.5</td>
      <td>0.662464</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>5</td>
      <td>1</td>
      <td>9500</td>
      <td>0.10</td>
      <td>0.1</td>
      <td>0.0009</td>
      <td>57.125</td>
      <td>54.0</td>
      <td>0.661862</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>8</td>
      <td>1</td>
      <td>10000</td>
      <td>0.07</td>
      <td>1.0</td>
      <td>0.0009</td>
      <td>52.875</td>
      <td>53.5</td>
      <td>0.553552</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>8</td>
      <td>1</td>
      <td>10000</td>
      <td>0.08</td>
      <td>1.2</td>
      <td>0.0001</td>
      <td>52.625</td>
      <td>53.0</td>
      <td>0.750212</td>
      <td>False</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>4</td>
      <td>1</td>
      <td>6000</td>
      <td>0.06</td>
      <td>2.0</td>
      <td>0.0003</td>
      <td>54.000</td>
      <td>49.5</td>
      <td>0.885316</td>
      <td>False</td>
    </tr>
  </tbody>
</table>



```python
printCostGraphs(c4, list(r4.iloc[0:8, 0]), 4, (10,20))
```


![png](assets/images/posts/2019/output_39_0.png)


# Verify Top Model


```python
%%html
<style>
  table {margin-left: 0 !important;}
</style>
```


<style>
  table {margin-left: 0 !important;}
</style>


If we examine the training results for the 160 models we find that the one with the best performance is the Epoch 1 model from the larger learning rate training batch.

It had the following hyperparameters:

|Hyperparameter     |Value   |
|-------------------|--------|
|Hidden Layer Units |6 	     |
|Training Iterations|1100    |
|Learning Rate:     |0.06    |
|L2 Lambda:         |0.6     |
|Weight Multiplier: |0.0002  |


The test and training accuracy were:

|Dataset|Value  |
|-------|-------|
|Train  |83.4%  |
|Test   |73.0%  |


Let's train the model again, ensure we can reproduce the cost graph and test set accuracy rate, and then generate the F-Score for comparison.

## Execute model with optimal hyperparameters


```python
# Epoch 	HL_Units 	HL_Size 	Iterations 	Learning_Rate 	Lambda 	Weight_Multi 	Train_Acc 	 Test_Acc 	Final_Cost 	Descending_Graph
# 1	6	1	1100	0.06	0.6	0.0002	83.375	73.0	0.350286	True

pLarge, cLarge, gLarge = model(trainingData, trainingLabels, 6, 1100, .06, .6, .0002, False, True)

trainingPredsLarge = predict(trainingData, pLarge, trainingLabels)
testPredsLarge = predict(testData, pLarge, testLabels)
print("\nTrain accuracy:", trainingPredsLarge["accuracy"])
print("Test accuracy:", testPredsLarge["accuracy"])
```


![png](assets/images/posts/2019/output_44_0.png)



    Train accuracy: 83.375
    Test accuracy: 73.0


## Generate F-Score


```python
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(np.squeeze(testLabels), np.squeeze(testPredsLarge["predictions"]))

print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F-Score: {}'.format(fscore))
```

    Precision: [0.69090909 0.77777778]
    Recall: [0.79166667 0.67307692]
    F-Score: [0.73786408 0.72164948]


# Model Summary

So far in this series of write-ups we have the following results:

<table>
    <tr>
        <th>Model Type</th>
        <th>Test Set Accuracy</th>
        <th>F-Score</th>
    </tr>
    <tr>
        <td>Linear regression</td>
        <td>65.5%</td>
        <td>[0.64974619 0.66009852]</td>
    </tr>
    <tr>
        <td>Shallow neural network</td>
        <td>73.0%</td>
        <td>[0.73786408 0.72164948]</td>
    </tr>
</table>

<p>As we continue to explore other models and optimization we will hopefully see these metrics continue to improve.</p>

---
