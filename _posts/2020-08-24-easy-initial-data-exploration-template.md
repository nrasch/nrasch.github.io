---
layout: post
title:  "Easy Initial Data Exploration Template"
tags: [ Python, Data Analysis]
featured_image_thumbnail: assets/images/posts/2020/easy-initial-data-exploration-template_thumbnail.jpg
featured_image: assets/images/posts/2020/easy-initial-data-exploration-template_title.jpg
featured: false
hidden: false
---

In this article we explore what it means to create a [Jupyter Notebook](https://jupyter.org/) template for use in initial data exploration. The resulting familiarity with the data can then be utilized as the basis for further detailed analysis, narrative development, feature engineering, etc.

<!--more-->

## Introduction

When leading teams or managing projects it is often that one will encounter data in different forms such as user feedback, project management metrics, system usage statistics, or performance metrics.

In each of these cases we'd like to be able to analyze the data, develop a narrative describing what the data is telling us, and then create one or more actionable recommendations and/or insights.

However, the first step is usually becoming familiar with the general 'shape' of the data, and then use that as a guide on how to proceed next. We aren't performing deep dive as of yet; our first goal is simply to become generally familiar with the data, and then allow this to guide us onto the next, more detailed steps.


![png](assets/images/posts/2020/data-analysis-process.png)

For example, let us assume we have a corpus of unstructured user feedback on a web application we manage. In this case we might want to understand the most commonly used language in the texts, frequency counts, and review lengths. Based on what we uncover perhaps we'll notice a high number of negative terms in the texts.

This in turn might lead us to start thinking about how we might more deeply analyze the seemingly negative comments:

* For example, should we categorize the reviews in a way that maps to various areas of the application?  Is the data structured in such a way that this is possible?
* If there isn't a clear categorical delineation should we instead apply an unsupervised [K-means](https://en.wikipedia.org/wiki/K-means_clustering) clustering algorithm to programmatically create groupings for further investigation?

Based on the groupings we find we could then potentially develop a narrative, a set of actionable recommendations to address the root causes for each category, and Key Performance Indicators (KPIs) to measure progress.

So, in the example above–and indeed the subject of this article–we use initial data exploration to shape and guide our deeper, secondary analysis, narrative development, and recommendations/insights.

## Assumptions

Before we begin a few assumptions:

### Your data is clean.  

Data cleaning is a subject unto itself, and a whole separate article could be written on the topic. But for the purposes of this write up; however, we will assume that your data is mostly clean, and that any issues with missing values, incorrect data types etc has been handled and dealt with before the initial exploration.

So, while we will take a quick look later on at dealing with a few messy records, this article will not explore the subject in detail.

[Here](https://www.justintodata.com/data-cleaning-python-ultimate-guide/) is a good resource on the subject if you want to learn more.

### You have some basic Python programming skills and access to a Jupyter Notebook environment.  

If you use the template from this article most of the programming work has been done for you. You'll simply need to adjust the template as you see fit to meet your objectives. If on the other hand you need help setting up your Jupyter Notebook environment you can refer to a previous article I wrote on this subject [here](https://nrasch.github.io/setup-a-machine-learning-python-development-environment-in-windows).

Also note if you need a hand let me know; I'll be happy to help out as I'm able.  :)

## Resources

We'll discuss each the following resources in more detail later on, but we include them here for easy reference:

* You can find the template discussed in this article [here](/assets/html/Easy-Initial-Data-Exploration-Notebook.ipynb)
* You can find the modified `Iris Data Set` utilized in the template [here](/assets/html/datasets/iris.data.csv)
* You can find the modified `IMDB Movie Reviews Set` text utilized in the template [here](/assets/html/datasets/reviews.obj)

## Let's Get Started!

So, having said all that, let's get started!

### Sample Data

The first thing we need to do; however, is obtain example data to execute the template against to illustrate how it works.

To do this we'll be utilizing two freely and commonly referred to example data sets:

* [The Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris/)
* [The IMDB Movie Reviews Set](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Note that we've made two modifications to these data sets:

* For the `Iris Data Set` we've added two new records to the set to simulate non-numeric and NaN issues.  You can find our modified version [here](assets/html/datasets/iris.data.csv).

* For the `IMDB Movie Reviews Set` we already cleaned and processed the review text to use it directly in the template.  You can find our modified version [here](assets/html/datasets/reviews.obj).

### Jupyter Notebook Template

Once we have access to the data sets we can utilize the Jupyter Notebook Template (found [here](assets/html/Easy-Initial-Data-Exploration-Notebook.ipynb)) to gain the initial idea of what the data looks like and plan our next actions.

Let's break down what the template is doing section-by-section:

# Load libraries and data

## Libraries

We start by importing the Python libraries we'll need for our initial data exploration:

```python
# Configure the notebook
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
```


```python
# Load libraries
import numpy as np

import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import DataFrame

from matplotlib import pyplot

from wordcloud import WordCloud, STOPWORDS

import pickle

import nltk
from nltk.probability import FreqDist

import os
```

## Data

Note we are loading a modified Iris data set and a IMDB Movie Reviews data set.  You can download these files [here](assets/html/datasets/iris.data.csv) and [here](assets/html/datasets/reviews.obj), and we assume you've placed the data sets in a directory called `datasets` in the same root folder as this template like so:

```
├── datasets
│   ├── iris.data.csv
│   └── reviews.obj
```

Let's go ahead and load the data files:

```python
dataFile = os.path.join('.', 'datasets', 'iris.data.csv')
data = read_csv(dataFile, header = 0)
```


```python
filehandler = open(os.path.join('.', 'datasets', 'reviews.obj'), 'rb')
reviews = pickle.load(filehandler)
```

# Helper functions

Next we want to write a quick helper function, `corrTableColors()`. This function allows us to set rules on when to apply colors to the `correlation` table values which we'll create below.

Note that the threshold values in the function can be altered to meet your needs as required. For example, we are highlighting any correlation values greater or less than 0.7, but these can easily be altered for your project data.

```python
def corrTableColors(value):
    # Default color
    color = 'black'

    if value == 1:
        color = 'white'
    elif value < -0.7:
        color = 'red'
    elif value > 0.7:
        color = 'green'

    return 'color: %s' % color
```

# Inspect and visualize the data (Iris Data Set)

Now we are ready for the fun part:  Inspecting the data and learning more about how it is structured!  

We'll utilize the `Iris Data Set` for the steps below, so that we can see how the commands work on an actual data set.

## Non-Visual Exploration

We start with a non-visual exploration where the properties of the data are examined such as the number of records, the data types, descriptive statistics, correlations, etc.

### Shape

The command below will return the number of rows by number of columns which are also referred to as `observations` and `variables`:


```python
print("data.shape = ", data.shape)
```

    data.shape =  (152, 5)


Thus we have 152 rows/observations and 5 columns/variables.

### First 10 rows

Using the `head` command we can examine the first 10 records:


```python
print(data.head(10))
```

      sepalLength  sepalWidth  petalLength  petalWidth        class
    0         5.1         3.5          1.4         0.2  Iris-setosa
    1         4.9         3.0          1.4         0.2  Iris-setosa
    2         4.7         3.2          1.3         0.2  Iris-setosa
    3         4.6         3.1          1.5         0.2  Iris-setosa
    4         5.0         3.6          1.4         0.2  Iris-setosa
    5         5.4         3.9          1.7         0.4  Iris-setosa
    6         4.6         3.4          1.4         0.3  Iris-setosa
    7         5.0         3.4          1.5         0.2  Iris-setosa
    8         4.4         2.9          1.4         0.2  Iris-setosa
    9         4.9         3.1          1.5         0.1  Iris-setosa


We can clearly see that the first four columns are numeric data, and the last column appears to be classification [labels](https://en.wikipedia.org/wiki/Labeled_data).  

We aren't sure at this point; however, what the ranges of the numeric data are or how many labels we are dealing with.  We'll explore this later on.

### Data Types

Let's confirm our assumption that the first four columns are numeric.  To do this we examine the output of the `dtypes` command:


```python
print(data.dtypes)
```

    sepalLength     object
    sepalWidth     float64
    petalLength    float64
    petalWidth     float64
    class           object
    dtype: object


OK, this is interesting!  The `dtypes` command appears to think we have some non-numeric data in the `sepalLength` column, and we know this because it classifies the column as the `object` type.

**Spoiler alert**:  Remember when I said above that we are using a modified version of the `Iris Data Set`? This is the reason: I intentionally placed two problematic records into the data to simulate what might happen if a few slipped through our initial cleaning. Below we'll take a few basic actions to mitigate this…

If we sort of the data in descending order the non-numeric values should float to the top and give us a better idea of what is occurring:


```python
data['sepalLength'].sort_values(ascending = False).head(5)
```

    151    abc
    131    7.9
    135    7.7
    118    7.7
    117    7.7
    Name: sepalLength, dtype: object


Ahah!  It appears that record `151` has a typo:  Someone (i.e. me) put the value `abc` into the field instead of an actual numeric value.

Let's drop the record and recast the column into a numeric data type.  

Note that in this case it is perfectly fine for us to drop the record, because we are working on initial data exploration.  At this point we simply want to get a sense of the story the data is telling us, and so removing a few problematic records shouldn't affect this outcome.


```python
# Drop the erroneous record; likely a data entry error
data.drop(151, inplace = True)

# Recast the column into numeric values
data['sepalLength'] = pd.to_numeric(data['sepalLength'])

# Confirm our work
print(data.dtypes)
```

    sepalLength    float64
    sepalWidth     float64
    petalLength    float64
    petalWidth     float64
    class           object
    dtype: object


We now see that all the columns are numeric except the last column which we assumed contained classification label data.

Another item we should check for is `NULL` values.  Python does not use the `NULL` keyword, and instead uses `NaN` and `None` to represent a `NULL`.

We can quickly check for this as follows:


```python
# isnull() will find both NaN and None values in a dataframe
# Let's see how many we have and count them up
data.isnull().sum()
```




    sepalLength    0
    sepalWidth     1
    petalLength    0
    petalWidth     0
    class          0
    dtype: int64



It appears we have one NaN/None value in our data set.  Let's go ahead and drop it, so we can continue with the exploration:


```python
#Drop any NaN/None values from our data set and then confirm we've removed them all:
data.dropna(inplace = True)

# Confirm our work
data.isna().sum()
```




    sepalLength    0
    sepalWidth     0
    petalLength    0
    petalWidth     0
    class          0
    dtype: int64



As a reminder: Data cleaning is a whole subject unto itself, which could have an entire article dedicated to this subject alone. For this write up; however, we are assuming your data is mostly clean, and we've just covered two basic commands to deal with a few records that might have slipped through.

### Descriptive stats

Next we can take a look at the descriptive statistics for the numeric data in our set:


```python
# Expand the width of the displayed table
set_option('display.width', 200)

# Set the decimal place at two places
set_option('precision', 2)

# Output descriptive statistics
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepalLength</th>
      <th>sepalWidth</th>
      <th>petalLength</th>
      <th>petalWidth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.00</td>
      <td>150.00</td>
      <td>150.00</td>
      <td>150.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.84</td>
      <td>3.05</td>
      <td>3.76</td>
      <td>1.20</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.83</td>
      <td>0.43</td>
      <td>1.76</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.30</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.10</td>
      <td>2.80</td>
      <td>1.60</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.80</td>
      <td>3.00</td>
      <td>4.35</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.40</td>
      <td>3.30</td>
      <td>5.10</td>
      <td>1.80</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.90</td>
      <td>4.40</td>
      <td>6.90</td>
      <td>2.50</td>
    </tr>
  </tbody>
</table>
</div>



We can see for example that the `sepalLength` values fluctuate between 4.3 and 7.9, and that the standard deviation (`std`) is relatively small for this data set.  This implies that most of the data is close to the mean, and that we don't have a wide range of values to consider.

We can also note that the numeric values seem to be on similar scale.  

Note that scale is important, because if you were planning on using an algorithm such as [linear regression](https://en.wikipedia.org/wiki/Linear_regression), [neural networks](https://en.wikipedia.org/wiki/Neural_network), [K-means](https://en.wikipedia.org/wiki/K-means_clustering), etc. as part of your narrative/analysis you'd need to take into scaling into account.

For example, when dealing with data such as housing you might value such as $450,000 for the house price and the value 3 for the number of bathrooms.  In these cases you'd need to apply `normalization` and/or `standardization` techniques to mitigate the variances between the value scales.  

For example, we wouldn't want the computer to become confused and think that the number of bathrooms was so small in comparison to the house price that it had no effect, which is quite the opposite!

While these topics are out-of-scope for this article, if you want to learn more you can read more about them [here](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/).

### Class distribution

Let's examine the last column which appeared to be categorization labels for each row.  How many different classes/labels are we dealing with?


```python
data.groupby('class').size()
```




    class
    Iris-setosa        50
    Iris-versicolor    50
    Iris-virginica     50
    dtype: int64



We note that we have an even distribution of classes/labels among the samples.  

If you were planning on using data modeling this would be important, because you'd have to account for a disparity in the number of classes within the observations if an imbalance existed.

For example, if we have one thousand positive comments and four negative ones, it would be hard to draw convincing conclusions, identify trends, or make recommendations on how we could improve on the four negative samples alone.

### Correlation

Next we examine the data for [correlations](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) using the `corr()` command.  We also utilize the helper function, `corrTableColors()`, we wrote above to make reading the table easier.

As stated before, if you want to change the color thresholds to fit your needs this should be easy to do.


```python
corr = data.corr(method = "pearson")
corr.style.applymap(corrTableColors)
```

<style  type="text/css" >
#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col0,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col1,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col2,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col3{
            color:  white;
        }#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col1,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col0,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col2,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col3,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col1,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col1{
            color:  black;
        }#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col2,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col3,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col0,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col3,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col0,#T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col2{
            color:  green;
        }</style><table id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sepalLength</th>        <th class="col_heading level0 col1" >sepalWidth</th>        <th class="col_heading level0 col2" >petalLength</th>        <th class="col_heading level0 col3" >petalWidth</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88level0_row0" class="row_heading level0 row0" >sepalLength</th>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col0" class="data row0 col0" >1.00</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col1" class="data row0 col1" >-0.11</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col2" class="data row0 col2" >0.87</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row0_col3" class="data row0 col3" >0.82</td>
            </tr>
            <tr>
                        <th id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88level0_row1" class="row_heading level0 row1" >sepalWidth</th>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col0" class="data row1 col0" >-0.11</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col1" class="data row1 col1" >1.00</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col2" class="data row1 col2" >-0.42</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row1_col3" class="data row1 col3" >-0.36</td>
            </tr>
            <tr>
                        <th id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88level0_row2" class="row_heading level0 row2" >petalLength</th>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col0" class="data row2 col0" >0.87</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col1" class="data row2 col1" >-0.42</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col2" class="data row2 col2" >1.00</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row2_col3" class="data row2 col3" >0.96</td>
            </tr>
            <tr>
                        <th id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88level0_row3" class="row_heading level0 row3" >petalWidth</th>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col0" class="data row3 col0" >0.82</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col1" class="data row3 col1" >-0.36</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col2" class="data row3 col2" >0.96</td>
                        <td id="T_36ee5286_e292_11ea_bfd9_c1ca50cf1e88row3_col3" class="data row3 col3" >1.00</td>
            </tr>
    </tbody></table>



Unsurprisingly the data set has one or more positive correlations present, which our intuition would have told us would be the case in data concerning plants.

In a real-world data set this would potentially provide us areas to investigate further.  For example, assume we had metrics on the number of clicks required to perform some task, and metrics on the time taken for a web page to load.  We could examine the correlation between these two metrics and the reported user satisfaction rating for the system in order to identify potential positive or negative relationships.

We would expect to find that requiring the user to click excessively together with long loading times would increase dissatisfaction, and the correlation table could help us empirically identify this linkage.   

### Skew

[Skewness](https://en.wikipedia.org/wiki/Skewness) is a measure of how long the 'tails' are in the data.  Here is an image to illustrate:

![png](assets/images/posts/2020/skewness.png)

In this image the 'tail' of the data drifts off to the right, and this is called `positive skewness`.  If on the other hand the tail would have drifted off to the left we would have said the data had `negative skewness`.

Skewness might interest you if you are concerned about data that doesn't follow the [bell curve](https://en.wikipedia.org/wiki/Normal_distribution) (also known as a `normal distribution`).

For example, let us assume we are examining data from a project concerning customer hold times.  If we saw `positive skewness` in the data we might be concerned, because the long right-side tail would imply some customers are having potentially excessive wait times which in turn could negatively impact customer satisfaction.

Let's see how we can view a numeric representation of skewness for our sample data:


```python
data.skew()
```




    sepalLength    0.31
    sepalWidth     0.33
    petalLength   -0.27
    petalWidth    -0.10
    dtype: float64



For a `normal distribution` the skew would be zero.  Our data set has skewness values close to zero, so there does not appear to be any significant positive or negative skew.

We'll also be able to visually confirm this in the next section.

## Visualizations

Now that we've finished with the non-visual data exploration let's employ some graphical techniques and see what we can find. :)

### Histograms

Creating histograms for our sample data set is extremely easy as show in the code below:


```python
# Control how big the histogram images are
data.hist(figsize = (8,8))

# Draw the histograms
pyplot.show()
```

![png](assets/images/posts/2020/output_47_0.png)


The histograms confirm what we learned earlier about `skewness` in the data, and both `sepalLength` and `sepalWidth` appear to `normal distributions` (i.e. follow the `bell curve`).  

`petalLength` and `petalWidth` on the other hand appear to be [bimodal](https://www.statology.org/bimodal-distribution/) distritubtions.  Or that is to say they appear to have two 'peaks' in their value distributions.

A `bimodal` (or `multimodal` if there are more than two peaks) might be of interest to you if you were considering scheduling or queue analysis data for example.  

Let's assume that we were working on a project to deploy an online restaurant point of sale (POS) system, and we wanted to ensure the system was responsive during peak hours (i.e. non-functional requirements). Examining the sales data for the restaurant hour-by-hour would likely reveal a bimodal distribution with two peaks: One at breakfast and another at lunch.

We could use these peaks to calculate the capacity and scalability requirements for the system, recommend on-demand load balancing for the back end POS servers during peak hours, etc.


### Density plots

`Density plots` can be thought of as plots of smoothed histograms. This assists in examining the 'shape' of the data, and are easier to read–for me anyhow–when considering distribution and multimodal distribution types.

Let's take a look:


```python
# Create the density plots in a 2x2 layout
data.plot(
    kind = 'density',
    subplots = True,
    layout = (2,2),
    sharex = False,
    figsize = (8,8)
)

# Draw the plots
pyplot.show()
```


![png](assets/images/posts/2020/output_50_0.png)



These plots make it very easy to gauge the distribution of the data as well as the presence of a multimodal distribution.

So for example we can clearly see that the `petalLength` and `sepalWidth` follow a `bell curve`, and that the `petalLength` and `petalWidth` values are indeed `bimodal`.

We also have an additional confirmation regarding the lack of skewness of the data as covered earlier.

### Box and whisker plots

`Box and whisker` plots are fantastic data visualization tools--although they can be non-intuitive to decipher at first.  [Here](https://flowingdata.com/2008/02/15/how-to-read-and-use-a-box-and-whisker-plot/) is a quick two-minute article on how to interpret them, and here is an info graphic explanation from that same source:


![png](assets/images/posts/2020/box-and-whisker-plot.png)

Creating them in our notebook is easy too:


```python
# Create the box-and-whisker plots in a 2x2 layout
data.plot(
    kind="box",
    subplots = True,
    layout = (2,2),
    sharex = False,
    sharey = False,
    figsize = (8,8)
)

# Draw the graphs
pyplot.show()
```

![png](assets/images/posts/2020/output_55_0.png)


We can note that the box-and-whisker plots contain a lot of information about our data set in a compact form:

* By examining the Y-axis values and 'whiskers' it is clear the data has a value spread of about 1 to 8 centimeters
* Other than the `sepalWidth` variable, the data does not contain outliers
* The interquartile range (IQR) also appears to have a narrow spread with the greatest amount of variation occurring in the `petalLengh` values
* We can quickly identify the median value for each variable by examining the value of the green bar.  For example, `sepalWidth` appears to have a median of about 5.75.

### Scatter plot matrix

We next consider a scatter plot matrix, which allows us to visually estimate correlations similar to how we did this above using quantitative means:


```python
scatter_matrix(data, figsize=(10, 10), s = 200)
pyplot.show()
```

![png](assets/images/posts/2020/output_58_0.png)


The scatter plot matrix reinforces visually what we saw above in the correlations table.

For example, we see that data points for `petalLength` vs. `petalWidth` increase steadily towards the upper right side of the graph in a diagonal line.  If we consult the correlations table we created earlier we find that these two variables are highly correlated at 0.96.

On the other hand, if we saw data points on the graph decreasing towards the lower right side of the graph we would acknowledge that there was negative correlation.

`sepalWidth` and `sepalLength` on the other hand don't seem to have much correlation present since there doesn't appear to be a clear pattern in the plotted data points.  The correlation table confirms with a recorded value of -0.11 for these two variables.

Here is a visual explanation of these concepts:


![png](assets/images/posts/2020/correlation-graphs.png)

[Source](https://sphweb.bumc.bu.edu/otlt/MPH-Modules/BS/BS704_Correlation-Regression/BS704_Correlation-Regression_print.html)

## Unstructured Text (IMDB Reviews)

At this point we are staring to have a good initial understanding of what our data looks like. However, what if we have unstructured natural language data such as customer reviews as part of our collection?

In this section we'll cover three easy techniques to analyze and assess unstructured textual data.

For our example data set we are going to use the `IMDB Movie Reviews` data set we loaded at the beginning of this notebook.

Note that we've already cleaned up the text by removing punctuation and other non-alphabetic values, set all the text to lower case, removed stop words such as 'a', 'and', 'the', etc., and each review is on a separate line.

Let's take a quick peek at one of the records we'll be working with:


```python
reviews[110]
```




    'truly dreadful know initially kiwi effort soon started realize characters speaking hardly disguised kiwi accents fake american ones need set n america anyway could set nz actors could used normal voices surely someone production team could hear dreadful attempts speaking american accents bad bad surprised lasted long make seemed like poor attempt segal willis type action man flick total waste money taxpayer money piece trash would leading revolution money put back treasury still reeling get pun reeling absolute garbage seen continue watch well fanatic cant help'



From all the negative words I think it's pretty clear someone didn't like this movie!

### World Clouds

The first technique we'll cover is a `word cloud`.

Google says a word cloud is "an image composed of words used in a particular text or subject, in which the size of each word indicates its frequency or importance."

Thus, the more often a specific word appears in the text, the bigger and bolder those words will appear in the word cloud.

So, for example if we were to utilize a word cloud on a corpus of text related to customer support tickets it might help us to gain an initial understanding of the most common issues. We might see the word 'slow' or 'missing' occur for a system that had latency and data errors.

Let's create one from our data, and see what it looks like:



```python
# Generate wordcloud
wordcloud = WordCloud(
    random_state=1,
    background_color='black',
    colormap='Set2',
    collocations=False,
    stopwords = STOPWORDS).generate(' '.join(reviews))

wordcloud.to_image()

```


![png](assets/images/posts/2020/output_66_0.png)



We quickly have a powerful graphical analysis tool with almost no code!

The words 'good' and 'time' have the 2nd and 3rd largest font size respectively.  We also see the word 'bad' in the word cloud, but its font size is much smaller.

From this we can form an initial assumption that our sample of movie reviews contains more positive reviews than negative.  Now of course we'd need to empirically validate this with further analysis, but for an initial data exploration it serves the purpose of helping us form a story in our mind of what the data is telling us and how to proceed next.

### Numerical Word Frequency Counts

If after creating the word cloud we want to know the exact word count frequencies for words appearing the greatest and least amount of times we can do that like so:


```python
# Count the number of times each word appears in the text(s)
data_analysis = nltk.FreqDist(' '.join(reviews).split())
```

Now we can view the top 10 most frequent words in the data set:


```python
data_analysis.most_common(10)
```




    [('one', 26788),
     ('like', 20274),
     ('good', 15140),
     ('time', 12724),
     ('even', 12646),
     ('would', 12436),
     ('story', 11983),
     ('really', 11736),
     ('see', 11475),
     ('well', 10662)]



And what if we wanted to know the top 10 least frequent words?

Easy enough:


```python
data_analysis.most_common()[-10:]
```




    [('riiiight', 1),
     ('imy', 1),
     ('jayden', 1),
     ('concider', 1),
     ('acids', 1),
     ('crahan', 1),
     ('substitutions', 1),
     ('daeseleire', 1),
     ('shortsightedness', 1),
     ('unfairness', 1)]



### Sentence Lengths

Next we'll examine the length of each record in the reviews.

First let's calculate the number of words per review, and store them in their own column:


```python
reviewsDF = pd.DataFrame(reviews, columns = ['Review Text'])
reviewsDF['Word Count'] = [len(x.split()) for x in reviewsDF['Review Text']]
```

We can examine how many words are in the first five reviews with the `head()` command:


```python
reviewsDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review Text</th>
      <th>Word Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>stuff going moment mj started listening music ...</td>
      <td>214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>classic war worlds timothy hines entertaining ...</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>starts manager nicholas bell giving welcome in...</td>
      <td>238</td>
    </tr>
    <tr>
      <th>3</th>
      <td>must assumed praised greatest filmed opera eve...</td>
      <td>184</td>
    </tr>
    <tr>
      <th>4</th>
      <td>superbly trashy wondrously unpretentious explo...</td>
      <td>208</td>
    </tr>
  </tbody>
</table>
</div>



Next we can use the `describe()` function as we did on the Iris data to create descriptive statistics for the number of words in each review:


```python
reviewsDF['Word Count'].describe()
```




    count    25000.00
    mean       116.16
    std         89.16
    min          3.00
    25%         61.00
    50%         86.00
    75%        141.00
    max       1416.00
    Name: Word Count, dtype: float64



What is really interesting here is the spread of the values.  The `std` statistic alone gives us an indication there is a lot of variance at `89.16`, and if we next examine the `min` and `max` entries we see values of `3` and `1,416` respectively.

(For reference 1,416 words is about three pages!)

Clearly we had two individuals with very different levels of motivation when it came to authoring movie reviews...

We can also depict this visually via a histogram:


```python
ax = reviewsDF['Word Count'].hist();
ax.set_xlabel("Number of words")
ax.set_ylabel("Number of records")
```




    Text(0, 0.5, 'Number of records')


![png](assets/images/posts/2020/output_81_1.png)


Whether or not the variance in record length matters will be up to you and your situation.  

For example, if this was unstructured text containing user comments on application performance perhaps I would remove records with less than twenty words and more than five hundred.  I might not feel that a record of less than twenty words or more than five hundred would have the concise details I needed to perform a proper root cause analysis.

Again; however, this will be up to you and your domain knowledge of your data.

As a final note, if we wanted to review the five records with the least amount of words we could do this:


```python
reviewsDF.sort_values(by = ['Word Count'], ascending = True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review Text</th>
      <th>Word Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20276</th>
      <td>terrible good effects</td>
      <td>3</td>
    </tr>
    <tr>
      <th>22258</th>
      <td>ming merciless little bardwork foul</td>
      <td>5</td>
    </tr>
    <tr>
      <th>9245</th>
      <td>better choose paul verhoeven even watched</td>
      <td>6</td>
    </tr>
    <tr>
      <th>23558</th>
      <td>rent one even dollar rental night</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4048</th>
      <td>adrian pasdar excellent makes fascinating woman</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



This not only gives us the word count for each record, but also the content and the row index where the record can be found in the data set (i.e. the values in the table's first column).

If you wanted to see the records with the largest number of words you could replace `ascending = True` with `ascending = False` like so:


```python
reviewsDF.sort_values(by = ['Word Count'], ascending = False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review Text</th>
      <th>Word Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3485</th>
      <td>match tag team table match bubba ray spike dud...</td>
      <td>1416</td>
    </tr>
    <tr>
      <th>20982</th>
      <td>titanic directed james cameron presents fictio...</td>
      <td>903</td>
    </tr>
    <tr>
      <th>20347</th>
      <td>attention spoilers first let say rob roy one b...</td>
      <td>892</td>
    </tr>
    <tr>
      <th>10503</th>
      <td>probably heard bit new disney dub miyazaki cla...</td>
      <td>853</td>
    </tr>
    <tr>
      <th>6864</th>
      <td>praised atlantis lost empire disney adventure ...</td>
      <td>805</td>
    </tr>
  </tbody>
</table>
</div>



## Wrapping Up

In this article we covered how to create a Jupyter Notebook template for initial data exploration. We then explored how to load the data, perform some simple cleaning operations, explore the data using numerical and graphic techniques, and we also examined three ways to initially analyze unstructured natural language data.

On the basis of this template we have gained a deeper understanding of the story the data had to tell us, and next we could take a deeper dive into narrative development, actionable recommendations, or KPIs and baselines to measure future data sets against.

If you have any methods, strategies, or experiences building out your own initial data exploration process and would like to share I would love to hear about them! Please let me know either in the comments or via email, and I will add them to this article (giving you full credit of course!).

I'd also be more than happy to answer any questions you may have after reading this, or if you have any problems setting up your own Jupyter Notebook environment and need a hand.

Thank you.
