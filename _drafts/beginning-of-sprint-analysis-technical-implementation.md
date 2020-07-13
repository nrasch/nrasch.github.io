---
layout: post
title:  "Beginning of Sprint Analysis - Technical Implementation"
tags: [ Agile, Scrum, Project Management]
featured_image_thumbnail: assets/images/posts/2020/beginning-of-sprint-analysis_thumbnail.jpg
featured_image: assets/images/posts/2020/beginning-of-sprint-analysis_title.jpg
featured: false
hidden: false
---
In this post we'll exercise our data science chops to describe how we created the report and analysis document we examined the narrative for in the [previous article]().  

Our goal for this article is to create a [Jupyter Notebook](https://jupyter.org/) that can be applied against data extracted at the beginning of a new sprint from [JIRA](https://www.atlassian.com/software/jira), and then have the notebook process the data and output reporting assets.  

This in turn is the basis for the analysis and narrative creation--which we discussed in the [last post]()--used to share our insights and recommendations with the project team and other business units.

The end result will be a programmatic solution that can be used at the start of each sprint to really dig into the details and augment the the Scrum Master's ability to coach the team and business on how to improve their processes.

<!--more-->
## Assumptions

This articles assumes you are familiar with the following and/or are interested in reading code discussion for the following technologies:

* [JIRA's](https://www.atlassian.com/software/jira) [API](https://developer.atlassian.com/server/jira/platform/rest-apis/)
* [JQL](https://www.atlassian.com/software/jira/guides/expand-jira/jql)
* [Jupyter Notebook](https://jupyter.org/)
* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

You can find the complete source code for this post [here]().

## Imports and Notebook Config

We start the notebook off with the import statements and notebook configuration settings we'll need throughout development:

```python

```

## Helper Functions

Next we instantiate the summary dataframe object that will hold the data processing results, and then we define four helper functions:

```python

```

<br/>
* A  **_get_** and  **_put_** function to read/write values from the summary dataframe object
* An  **_autolabel_** function to draw values above bar graph elements
  * A good article on this can be found [here](http://composition.al/blog/2015/11/29/a-better-way-to-add-labels-to-bar-charts-with-matplotlib/)
* A function to create JIRA hyperlinks from text values and URL parameters

## Define and Initialize

In the next section we start setting everything up for the data processing and feature creation, as well as capturing the sprint's metrics for future trend analysis.

We start off by defining the name of the sprint:


The  **_summary_** dataframe object--along with it's keys--is created next.  The goal is to populate and save this object based on the collected and processed JIRA data.  We can then take similar summary objects from a collection of sprints and examine the historical data for trends and patterns over time at a later date.

This code block for example captures statistics on the number of JIRA story items assigned to the sprint during planning:

We also capture the statics and warning thresholds for story point size and number of rollovers:

Reminder:  You can view the full source code--including the entirety of the  **_summary_** object's keys--[here](*).

## Load the Data

This section adds a column to the  **_summary_** dataframe object for the current sprint and then reads in the  **_csv_** data extracted from JIRA.

## Data Processing and Feature Creation

And now we get to the good stuff!

Note that we won't review every line of processing and/or feature creation, but we will review a fair number of examples.  You can always view the full source code--including the entirety of the processing/feature creation section--[here](*).

First a set of constants for the sprint are defined:

```python
# Init sprint values for calculations below
put('laborCostPerHour', 50)
put('hoursPerSprint', 60)
put('pointsPerSprint', 10)
put('rollOverThreshold', 2)
put ('storyPointSizeWarningThreshold', 8)
put('totalResourceCount', 10)
```

Note that we are making use of our  **_put_** helper function to populate the appropriate key and column values in the  **_summary_** dataframe object.

Once that's accomplished the constants can be utilized to populate the data elements for the budget summary table:

```python
# Calculate total resource hours
put('totalResourceHours', get('totalResourceCount') * get('hoursPerSprint'))

# Calculate cost per story point
put('totalStartingCostPerPoint',
    (get('hoursPerSprint') / get('pointsPerSprint') )
    * get('laborCostPerHour'))

# Record total number of JIRA items in the sprint
put('totalStartingItemCount', df.shape[0])

# Record total starting story points in the sprint
put('totalStartingItemPoints', df['Story Points'].sum());

# Add $ values to each item
df['Cost'] = (df['Story Points'] * get('totalStartingCostPerPoint'))

# BAC and cost ratios for each item
put('BAC', np.sum(df['Cost']))
df['Cost Ratio'] = df['Cost'].values / get('BAC') * 100
```

In the written assumptions area of the report we stated the following for the reader:
* A JIRA  **_task_** item is classified the same as a JIRA  **_story_** item

An implementation example of this assumption in the code:

```python
# Merge Tasks and Stories
df.loc[df.Type == 'Task', 'Type'] = 'Story'
```

Here JIRA items of type  **_Task_** are being renamed to type  **_Story_** inside the dataframe.  This will allow them to be grouped later on for aggregation purposes.

Next we group the JIRA items by type, record story point metrics, and calculate how much each story contributes to the total velocity of the sprint:

```python
# Story point ratio
df['Story Point Ratio'] = df['Story Points'].values / get('totalStartingItemPoints') * 100

# Total number of items to item type ratios
for item in ['Story', 'Spike', 'Bug']:
    put('totalStarting'+item+'CountRatio', df.groupby('Type').size()[item]
       / get('totalStartingItemCount') * 100)  

# Desc. stats on story points
stats = df['Story Points'].describe()
for stat in ['mean', 'std', 'min', 'max']:
    put('totalStarting' + stat.capitalize(), stats[stat])

# Type count and story point sums
for item in ['Story', 'Spike', 'Bug']:
    put('totalStarting' + item + 'Count', df.groupby('Type').size()[item])
    put('totalStarting' + item + 'Points', df.groupby('Type')['Story Points'].sum()[item])

# Type story point ratios
for item in ['Story', 'Spike', 'Bug']:
    put('totalStarting' + item + 'PointsRatio', df.groupby('Type')['Story Points'].size()[item]
        / get('totalStartingItemPoints') * 100)
```

We can also utilize Python's [list comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) to make short work of the rollover count calculations:

```python
df['rollOverCount'] = [len(x.split(',')) - 1 for x in df['Sprint'].values ]
```
JIRA stores the number of times a story has rolled over in a comma delimited list in the  **_Sprint_** column.  This allows us to split by the comma on that field and take the length to find out how many times a story has rolled over.  We want to subtract one from the length, so we don't count the current sprint against the total.  We take the values from this calculation and assign it to the  **_rollOverCount_** field in the  **_summary_** dataframe object.

As a final example we can also create new features of interest by combining elements we developed and populated earlier:

```python
# Calculate APV and PAV ratio
put('APV', get('totalStartingStoryValue') + get('totalStartingSpikeValue'))
put('APVRatio', (get('APV')/get('BAC'))*100)
```

So in this example the APV value allows us to calculate how much of the sprint's  **_budget_** we are spending on development work vs. technical debt and support work.

The final results are written to disk for archiving and future trend analysis:

```python
#Write the results to disk
summary.to_csv('./' + sprint +'-etc.csv', )
```


## Budget Analysis
The budget analysis--for now--is a simple table, because we are working towards providing the rest of the business insights into the gross impact of technical debt and support items:


To do this we add a number of rows to a newly created Pandas dataframe object and print it.  Fast, easy, and effective.

Once our initial educational goals are accomplished in this area we can expand the budget analysis further.

Note that the last line of the code block cell allows us to output the budget dataframe object in pure HTML without the index column:

```python
display(HTML(budget.to_html(index=False)))
```

## Story Point Analysis

We start this section off with a summary table:


### Distribution of User Stories and User Story Points by Type

Next we draw our first to bar graphs by making calls to the [plt.subplot()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function


Note that we color the first bar relating to technical debt and support issues  **_light coral_** to distinguish it for the reader.  We use our helper function to place numbers over each bar representing their value on the Y-axis and remove extraneous chart ink and junk by clearing out any backgrounds, axis lines, and graph borders.


### Distribution of User Story Points by Size

This graph is created in a similar way to the previous two graphs.  Once difference; however, is we utilize a mask to color those bars exceeding the  **_storyPointSizeWarningThreshold_** value we set in the  **_summary_** dataframe object.


### User Stories Assigned Zero Points

This tabular report element is included as a reference to the reader to assist in examining the exact user stories assigned zero points .  It also provides the reader with a link to directly navigate to those items in JIRA for additional action.

The table uses the  **_makeStoryLink()_** helper function we defined at the start of the notebook in order to create the JIRA URL links in the table.  The  **_makeStoryLink()_** helper function is passed as part of a dictionary parameter to the [style.format()](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Finer-Control:-Display-Values) method chain function.

Note that using the  **_style.format()_** method returns a  **_pandas.io.formats.style.Styler_** object, and you can read more about it's methods and properties [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.html).

This allows us for example to call the [hide_index()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.hide_index.html#pandas.io.formats.style.Styler.hide_index) method of the  **_pandas.io.formats.style.Styler_** object in order to remove the dataframe's index column from displaying in the table.

### User Stories Assigned the Greatest Story Point Values

This table has been created in the same manner and for the same purpose as the one above in order to assist in examining larger sized user stories.

### User Story Rollover Distribution

This graph has been created in the same way as the preceding graph element, and helps the reader to ascertain how many JIRA items have rolled over sprint-to-sprint.  

### Most Rolled Over User Stories

This tabular element has been created utilizing the same methods as previous tables, and provides additional information and direct links to those JIRA items that have rolled over more than defined threshold value.

## User Story Point Statistics

### Descriptive Statistics

In this table we review some of the more traditional statistics of the data such as standard deviation, mean, and quartile ranges.  Pandas has the built in [describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) method which makes creating this table a straightforward task.

As I described in the [narrative](https://nrasch.github.io/beginning-of-sprint-analysis#user-story-point-statistics) the team should focus on the following elements initially for improvement:

* Std
* Min
* Max

### Box Plot

The final element of the report is the humble [box plot](https://en.wikipedia.org/wiki/Box_plot).  This figure provides a nice visual counterpoint to the tabular data just above it in the report, and makes it easy to quickly identify outliers and other items of concern such as story points with a zero value as indicated by the lower whisker range.


## Wrapping Up

In this post we explored how to programatically create a start-of-sprint analysis report utilizing Python, Panas, and Matplotlib.  The goals of the report were to present actionable data analysis on start-of-sprint metrics as well as record those metrics for future analysis and trend reporting.

One of the main benefits of creating this report programatically is that it can now be run on-demand against future sprint JIRA data extracts.  This allows us to create consistent, repeatable reports at the start of each sprint for dissemination, and gives us an apples-to-apples comparison of our Agile Scrum process improvement progress.  It also allows the Scrum team to diagnose and respond to potential issues occurring during refinement, planning, etc.

We also have the advantage of being able to leverage any of the wide range of Python data science tools which are constantly being added to and improving.  :)





```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
