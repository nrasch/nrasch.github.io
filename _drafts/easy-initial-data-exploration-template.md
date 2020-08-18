---
layout: post
title:  "Easy Initial Data Exploration Template"
tags: [ Software Development, Management]
featured_image_thumbnail: assets/images/posts/2020/incrementaly-modernizing-your-software-development-processes_thumbnail.jpg
featured_image: assets/images/posts/2020/incrementaly-modernizing-your-software-development-processes_title.jpg
featured: false
hidden: false
---

In this article we explore a Jupyter Notebook template for assisting in initial data exploration.  The resulting familiarity with the data can then be utilized as the basis for further detailed analysis, narrative development, feature engineering, etc.

<!--more-->

## Introduction

Often when leading teams or managing projects we’ll be confronted with various data sets.  Examples might be user feedback on a new feature or system, project management metrics, system usage statistics, or performance metrics.

In each of these cases we’d like to be able to analyze the data, develop a narrative describing what the data is telling us, and then create one or more actionable recommendations.

However, the first step is usually becoming familiar with the general shape of the data, and then using that as a guide on how to proceed next.  We aren’t performing a deep dive; we just want to become generally familiar with the data, and then let that guide us onto the next, more detailed steps.

[insert flow diagram here]

For example, let us assume we have a set of unstructured textual data gathered from user feedback on a web application we manage.  In this case we might want to understand the most commonly used language in the text, frequency counts, and review length.  Based on what we see perhaps we’ll notice a high number of negative terms used in the review.  This will help us to start thinking about how to more deeply analyze the negative reviews by perhaps sorting them into categories that map to various areas of the application, or if there isn’t a clear delineation perhaps we’d like to apply an unsupervised  K-means clustering algorithm to the data to programmatically create groupings for further investigation.

Based on the groupings we find we could then develop a narrative explaining what the data is telling us, a set of actional recommendations to address the root causes for each category, and Key Performance Indicators (KPIs) to measure progress.

## Assumptions


Before we begin a few assumptions:

### Your data is clean.  

Data cleaning is a whole subject unto itself, and we could write an entire separate article on the matter.  For the purposes of this writeup; however, we’ll assume your data is mostly clean, and you’ve dealt with any missing values, incorrect data types, etc. before the initial exploration.

We’ll take a quick look into dealing with a few messy records, but we won’t take a deep dive into the subject in this article.

### You have some basic Python programming skills and access to a Jupyter Notebook environment.  

If you use the template from this article most of the programming work has been done for you.  You’ll simply need to adjust the template as you see fit to meet your objectives.  If on the other hand you need help setting up your Jupyter Notebook environment you can refer to a previous article I wrote on this subject [here].

## Resources

We’ll discuss each the following resources in more detail later on, and we include them here for easy reference:

* You can find the template discussed in this article [here]()
* You can find the modified Iris data utilized in the template [here]()
* You can find the modified IMDB movie review text utilized in the template [here]()

## Let’s Get Started!

### Sample Data

With the introduction and assumptions out of the way we can get started.  First; however, we need some data to execute the template against and show how it works.

We’ll be utilizing two freely available data sets that are commonly used throughout the data science community:

* [The Iris Data Set](http://archive.ics.uci.edu/ml/datasets/Iris/)
* [The IMDB Movie Reviews Set](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Note that we’ve made two modifications to these data sets:

For the `Iris Data Set` we’ve added two new records to the set to simulate non-numeric and NaN issues.  You can find our modified version [here]().

For the `IMDB Movie Reviews Set` we already cleaned and processed the review text, so that we can utilize it directly in the template.  You can find our modified version [here]().

### Jupyter Notebook Template

Once we have the cleaned data in hand we can begin utilizing the Jupyter Notebook Template (found [here]()) to gain an initial idea of what the data looks like and plan our next actions.  

Let's break down what the template is doing section-by-section:
