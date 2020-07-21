---
layout: post
title:  "Beginning of Sprint Analysis - Technical Implementation"
tags: [ Agile, Scrum, Project Management, Python]
featured_image_thumbnail: assets/images/posts/2020/beginning-of-sprint-analysis_thumbnail.jpg
featured_image: assets/images/posts/2020/beginning-of-sprint-analysis_title.jpg
featured: false
hidden: false
---
In this post we'll exercise our data science chops to describe how we created the report and analysis document that we examined the narrative for in the [previous article]({% post_url 2020-07-04-beginning-of-sprint-analysis %}).  

Our goal for this write up is to create a [Jupyter Notebook](https://jupyter.org/) that can be applied against data extracted at the beginning of a new sprint from [JIRA](https://www.atlassian.com/software/jira), and then have the notebook process the information and output a reporting asset.  

This in turn is the basis for the analysis and narrative creation--which we discussed in the [last post]({% post_url 2020-07-04-beginning-of-sprint-analysis %})--used to share our insights and recommendations with the project team and other business units.

The end result will be a programmatic solution that can be used at the start of each sprint to really dig into the details and augment the the Scrum Master's ability to coach the team and business on how to improve their processes.

**NOTE:**  You can view the complete source code for this post [here](https://nbviewer.jupyter.org/urls/nrasch.github.io/assets/html/2000-01-01-Sprint-Starting-Analysis.ipynb).

<!--more-->

## Assumptions

This articles assumes you are familiar with and/or are interested in reading code discussion for the following technologies:

* [JIRA's](https://www.atlassian.com/software/jira) [API](https://developer.atlassian.com/server/jira/platform/rest-apis/)
* [JQL](https://www.atlassian.com/software/jira/guides/expand-jira/jql)
* [Jupyter Notebook](https://jupyter.org/)
* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)

**NOTE:**  You can view the complete source code for this post [here](https://nbviewer.jupyter.org/urls/nrasch.github.io/assets/html/2000-01-01-Sprint-Starting-Analysis.ipynb).

## Extracting Data from JIRA

I have found one of the best ways to quickly and easily pull all the information I want out of JIRA for dashboards and reports is via the [JIRA Command Line Interface (CLI) plugin](https://marketplace.atlassian.com/apps/6398/jira-command-line-interface-cli?hosting=cloud&tab=overview).  

Not only can you extract data, but you can also created and modify stories, projects, tasks, etc. as well.  This has come in very handy in the past when I wanted to make changes to a large number of stories and needed to avoid having to do it by hand.  You can find a full reference and user guide [here](https://bobswift.atlassian.net/wiki/spaces/JCLI/overview).

For example, here is the command I execute to pull the data for analysis at the beginning of the sprint from the three main project boards:

```
acli --action getIssueList --jql "project in (P1, P2, P3) AND sprint in openSprints() and sprint not in futureSprints()" --user "XXXX" --password 'YYYY' --server "http://some.server.com" --file "~/beginning_sprint_data.csv" --outputFormat 999 --dateFormat "yyyy-MM-dd"
```

**_Tip_**:  You can quickly fine tune your JQL query in JIRA using the **_Issues > Search for issues > Advanced_** area of JIRA, and then cut-and-paste the JQL into the CLI statement.  

And here is a [quick primer](https://www.atlassian.com/software/jira/guides/expand-jira/jql#advanced-search) on advanced searching in JIRA if you haven't utilized this feature before.

Next we'll jump into analysis report, and how it was created.

## Imports and Notebook Config

We start the notebook off with the import statements and notebook configuration settings we'll need throughout development:

```python
## IMPORTS AND NOTEBOOK CONFIG ##

import pandas as pd
from pandas import set_option

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import numpy as np

import math

from IPython.core.display import display, HTML

%matplotlib inline
```

## Helper Functions

Next we instantiate the summary dataframe object that will hold the data processing results, and then we define four helper functions:

```python
##  HELPER FUNCTIONS  ##

# Summary dataframe object
summary = pd.DataFrame()

# Helper getter function
def get(index, column = ''):
    if (column == ''):
        column = sprint

    return summary.loc[index, column]

# Helper setter function
def put(index, value, column = ''):
    if (column == ''):
        column = sprint

    summary.loc[index, column] = value
    return summary.loc[index, column]


# Helper function to add values above the bar graph elements
def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        # If we can fit the label above the column, do that;
        # otherwise, put it inside the column.
        if p_height > 0.95: # arbitrary; 95% looked good to me.
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        ax.text(rect.get_x() + rect.get_width()/2., label_position,
                '%d' % int(height),
                ha='center', va='bottom')


# Helper fuction to return hyperlink
def makeStoryLink(val, url = 'http://<YOUR_JIRA_URL>/browse/'):
    url = url + val
    return '<a href="{}">{}</a>'.format(url,val)
```

<br/>
* A **_get()_** and **_put()_** function to read/write values from the summary dataframe object
* An **_autolabel()_** function to draw values above bar graph elements
  * A good article on this can be found [here](http://composition.al/blog/2015/11/29/a-better-way-to-add-labels-to-bar-charts-with-matplotlib/)
* The **_makeStoryLink()_** helper function to create JIRA hyperlinks from text values and URL parameters

## Define and Initialize

In the next section we start setting everything up for the data processing and feature creation, as well as capturing the sprint's metrics for future trend analysis.

We start off by defining the name of the sprint:

```python
##  DEFINE VARIABLES  ##

# Define sprint label
sprint = '2000-01-01'
```

Next the **_summary_** dataframe object--along with its indices--is instantiated.  The goal is to populate and save this object based on the collected and processed JIRA data.  We can then take similar summary objects from a collection of sprints and examine the historical data for trends and patterns over time at a later date.

Here is an example of creating the **_summary_** object along with indices that capture statistics on the number of JIRA story items assigned to the sprint initially:

```python
# Define summary container
summary = pd.DataFrame(index = [

    ...

    # Total number of jira items
    'totalStartingItemCount',
    # Total number of jira items by type
    'totalStartingStoryCount',
    'totalStartingSpikeCount',
    'totalStartingBugCount',
    # Number of items by type / total number of items
    'totalStartingStoryCountRatio',
    'totalStartingSpikeCountRatio',
    'totalStartingBugCountRatio',

    ...
```

We also capture the statics and warning thresholds for story point size and number of rollovers:

```python
  ...

  # Sprint roll over stats
  'totalStartingRolloverCount',
  'rollOverCountMean',
  'rollOverCountStd',
  'rollOverCountMin',
  'rollOverCountMax',

  # Warning limit for number of times a story point can be rolled over from sprint to sprint
  'rollOverThreshold',

  # Warning limit for story point size
  'storyPointSizeWarningThreshold'

  ...
```

Reminder:  You can view the full source code--including the entirety of the **_summary_** object--[here](*).

## Load the Data

The next section adds a column to the **_summary_** dataframe object for the current sprint and then reads in the **_csv_** data extracted from JIRA:

```python
##  LOAD THE BEGINNING SPRINT DATA AND SET VALUES  ##

# Init the summary container
summary[sprint] = np.nan

# Load the beginning sprint data
df = pd.read_csv('./' + sprint + '-anon.csv')

# Define sprint values as outlined in our assumptions section
# above for the calculations below
put('laborCostPerHour', 50)
put('hoursPerSprint', 60)
put('pointsPerSprint', 10)
put('rollOverThreshold', 2)
put ('storyPointSizeWarningThreshold', 8)
put('totalResourceCount', 10)

pass;
```
Note that we are making use of our **_put()_** helper function to populate the appropriate key and column values in the **_summary_** dataframe object.

## Data Processing and Feature Creation

And now we get to the good stuff!

Note that we won't review every line of processing and/or feature creation, but we will review a fair number of examples.  You can always view the full source code--including the entirety of the processing/feature creation section--[here](*).

The sprint values we defined can now be utilized to populate the data elements for the budget summary table:

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
We can also take care of combining **_Task_** and **_Story_** items into one category:

```python
# Merge Tasks and Stories
df.loc[df.Type == 'Task', 'Type'] = 'Story'
```

Here JIRA items of type **_Task_** are being renamed to type **_Story_** inside the dataframe.  This will allow them to be grouped later on for aggregation purposes.

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

JIRA stores the number of times a story has rolled over in a comma delimited list in the **_Sprint_** column.  This allows us to split by the comma on that field and take the length to find out how many times a story has rolled over.  We want to subtract one from the length, so we don't count the current sprint against the total.  We take the values from this calculation and assign it to the **_rollOverCount_** field in the **_summary_** dataframe object.

As a final example we can also create new features of interest by combining elements we developed and populated earlier:

```python
# Calculate APV and APAV ratio
put('APV', get('totalStartingStoryValue') + get('totalStartingSpikeValue'))
put('APVRatio', (get('APV')/get('BAC'))*100)
```

So in this example the APV value allows us to calculate how much of the sprint's **_budget_** we are spending on development work vs. technical debt and support work.

The final results are written to disk for archiving and future trend analysis:

```python
#Write the results to disk
summary.to_csv('./' + sprint +'-etc.csv', )
```


## Budget Analysis
The budget analysis--for now--is a simple table as we are initially working towards educating external business units on the gross impact of technical debt and support items.  Let's not swamp them in details just yet...

```python
pd.set_option('precision', 2)

budget = pd.DataFrame( columns = ['Budget Item', 'Total', 'Ratio', 'Item Count'])
row = [
    ['Planned Development',
     '${:,.2f}'.format(get('APV')),
     '{:,.2f}%'.format(get('APVRatio')),
     get('totalStartingStoryCount') + get('totalStartingSpikeCount')
    ]
    ,['Planned Tech Debt / Support',
      '${:,.2f}'.format(get('totalStartingBugValue')),
      '{:,.2f}%'.format(get('totalStartingBugCostRatio')),
      get('totalStartingBugCount')
     ]
    ,['Total Sprint Budget',
      '${:,.2f}'.format(get('BAC')),
      '100%',
      get('totalStartingStoryCount') + get('totalStartingSpikeCount') + get('totalStartingBugCount')
     ]
    ,['', '', '', '']
    ,['Cost per Story Point',
      '${:,.2f}'.format(get('totalStartingCostPerPoint')),
      '', ''
     ]
]

for r in row:
    budget.loc[len(budget)] = r

display(HTML(budget.to_html(index=False)))
```

The table is created by adding a number of rows to a newly created Pandas dataframe object and calling the **_display(HTML())_** function passing in the dataframe as an argument.  

Note we suppress the index values of the dataframe from displaying with the **_.to_html(index=False)_** method of the dataframe object.

## Story Point Analysis

### Sprint User Story and Point Summary

We start this section off with a summary table:

```python
pd.set_option('precision', 0)

items = pd.DataFrame( columns = ['Item', 'Value'])
rows = [
    ['Total Number of Stories', get('totalStartingItemCount')]
    ,['Total Number of Story Points', get('totalStartingItemPoints')]
]

for r in rows:
    items.loc[len(items)] = r

display(HTML(items.to_html(index=False)))
```

### Distribution of User Stories and User Story Points by Type

Next we draw our first two bar graphs by making calls to the [plt.subplot()](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function:

```python
# Setup params for displaying the two subplots
plt.rcParams.update({'font.size': 12})
plt.figure(num=None, figsize=(11, 5))
plt.subplots_adjust(wspace=.55)


####
# Distribution of Sprint Items by Type graph

# Define plotting area for the figure
ax = plt.subplot(1, 2, 1)

# Instantiate and configure the graph
x = np.arange(len(df.groupby('Type')['Key']))
y = list(df.groupby('Type')['Key'].count())
ax.set_ylim([0,np.max(y) + 5])


# Labels, ticks, and titles
plt.xlabel('JIRA Item Type')
plt.ylabel('Number of Items')
plt.title('Distribution of Sprint Items by Type', fontsize=16)

xLabels = df.groupby('Type')['Key'].count().index
plt.xticks(x, xLabels)
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis=u'both', which=u'both',length=0)

# Create the bar graph
bars = plt.bar(x, y, color = 'steelblue', alpha=0.7)
plt.bar(x[0], y[0], color = 'lightcoral', alpha = 0.7)

# Draw the value for each bar above it
autolabel(bars, plt.gca())

# Remove figure outline
plt.box(False)


####
# Distribution of Story Points by Type graph

# Define plotting area
ax = plt.subplot(1, 2, 2)

# Instantiate and configure the graph
x = np.arange(len(df.groupby('Type')['Story Points']))
y = list(df.groupby('Type')['Story Points'].sum())
ax.set_ylim([0,np.max(y) + 15])

# Labels, ticks, and titles
plt.xlabel('JIRA Item Type')
plt.ylabel('Number of Story Points')
plt.title('Distribution of Story Points by Type', fontsize = 16)

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis = u'both', which = u'both',length = 0)
xLabels = df.groupby('Type')['Story Points'].sum().index
plt.xticks(x, xLabels);

# Create the bar graph
bars = plt.bar(x, y, color = 'steelblue', alpha=0.7)
plt.bar(x[0], y[0], color = 'lightcoral', alpha = 0.7)

# Draw the value for each bar above it
autolabel(bars, plt.gca())

# Remove figure outline
plt.box(False)
```

Note that we color the first bar relating to technical debt and support issues **_light coral_** to distinguish it for the reader.  We use our helper function to place numbers over each bar representing their value on the Y-axis, and extraneous chart junk is removed by clearing out any backgrounds, axis lines, and graph borders.


### Distribution of User Story Points by Size

```python
# Distribution of Sprint Items by Story Point Size graph

# Gather data set for the graph
tmp = df.groupby('Story Points').size()

# Graph settings
plt.figure(num=None, figsize=(9, 5))
plt.subplots_adjust(wspace = .75)

# Define plotting area
ax = plt.subplot(1, 1, 1)

# Instantiate and configure the graph
x = np.arange(len(list(tmp.index)))
y = tmp.values
ax.set_ylim([0,np.max(y) + 5])

# Labels, ticks, and titles
plt.xlabel('Story Point Size')
plt.ylabel('Number JIRA Items')
plt.title('Distribution of Sprint Items by Story Point Size', fontsize = 16)

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis = u'both', which = u'both',length = 0)
xLabels = list(tmp.index)
plt.xticks(x, xLabels);

# Create the bar graph
bars = plt.bar(x, y, color = 'steelblue', alpha = 0.7)

# Draw the value for each bar above it
autolabel(bars, plt.gca())

# Remove figure outline
plt.box(False)

# Apply coloring on bars based on story point value
mask = tmp.index >= get('storyPointSizeWarningThreshold')
plt.bar(x[mask], y[mask], color = 'palegoldenrod', alpha = 0.7);
```

This graph is created in a similar way to the previous two graphs.  Once difference; however, is we utilize a mask to color those bars exceeding the **_storyPointSizeWarningThreshold_** value we set in the **_summary_** dataframe object:

```python
# Apply coloring on bars based on story point value
mask = tmp.index >= get('storyPointSizeWarningThreshold')
plt.bar(x[mask], y[mask], color = 'palegoldenrod', alpha = 0.7);
```

## Least and Greatest Story Point Items

The next section of the report deals with the least and greatest sized stories.

### User Stories Assigned Zero Points

```python
(
    df.loc[df['Story Points'] == summary.loc['totalStartingMin'][0]]
    [['Key', 'Summary', 'Assignee', 'Story Points']]
    .style.format({'Key': makeStoryLink})
).hide_index()
```

This tabular report element is included as a reference to the reader to assist in examining the exact user stories assigned zero points .  It also provides the reader with a link to directly navigate to those items in JIRA for additional action.

The table uses the **_makeStoryLink()_** helper function we defined at the start of the notebook in order to create the JIRA URL links in the table.  The **_makeStoryLink()_** helper function is passed as part of a dictionary parameter to the [style.format()](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Finer-Control:-Display-Values) method chain function.

Note that using the **_style.format()_** method returns a **_pandas.io.formats.style.Styler_** object, and you can read more about it's methods and properties [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.html).

This allows us for example to call the [hide_index()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.hide_index.html#pandas.io.formats.style.Styler.hide_index) method of the **_pandas.io.formats.style.Styler_** object in order to remove the dataframe's index column from displaying in the table.

### User Stories Assigned the Greatest Story Point Values

```python
(
    df.loc[df['Story Points'] >= get('storyPointSizeWarningThreshold')]
        .sort_values(['Story Points', 'Key'], ascending = False)
    [['Key', 'Summary', 'Assignee', 'Story Points']]
    .style.format({'Key': makeStoryLink})
).hide_index()
```

This table has been created in the same manner and for the same purpose as the one above in order to assist in examining larger sized user stories.

### User Story Rollover Distribution

```python
# Distribution of Sprint Items by Rollovers graph

# Gather data set for the graph
tmp = df[df['rollOverCount'] > 0].groupby('rollOverCount').size()

# Graph settings
plt.rcParams.update({'font.size': 12})
plt.figure(num=None, figsize=(9, 5))
plt.subplots_adjust(wspace = .75)

# Define plotting area
ax = plt.subplot(1, 1, 1)

# Instantiate and configure the graph
y = tmp.values
x = np.arange(len(list(tmp.index)))
ax.set_ylim([0,np.max(y) + 5])

# Labels, ticks, and titles
plt.xlabel('Number of Rollovers')
plt.ylabel('Number JIRA Items')
plt.title('Distribution of Sprint Items by Number of Times Rolled Over', fontsize = 16)

xLabels = list(tmp.index)
ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.tick_params(axis = u'both', which = u'both',length = 0)
plt.xticks(x, xLabels);

# Create the bar graph
bars = plt.bar(x, y, color = 'steelblue', alpha = 0.7)

# Remove figure outline
plt.box(False)

# Draw the value for each bar above it
autolabel(bars, plt.gca())

# Apply coloring on bars based on graph values
mask = x > get('rollOverThreshold')
plt.bar(x[mask], y[mask], color="lightcoral", alpha = .7);
```

This graph has been created in the same way as the preceding graph element, and helps the reader to ascertain how many JIRA items have rolled over sprint-to-sprint.  

### Most Rolled Over User Stories

```python
(
    df.loc[df['rollOverCount'] > get('rollOverThreshold')+1]
    [['Key', 'Summary', 'Assignee', 'Story Points', 'rollOverCount']]
    .sort_values(['rollOverCount'], ascending = False)
    .style.format({'Key': makeStoryLink})
).hide_index()
```

This tabular element has been created utilizing the same methods as previous tables, and provides additional information and direct links to those JIRA items that have rolled over more than defined threshold value.

## User Story Point Statistics

### Descriptive Statistics

```python
set_option('precision', 1)
df['Story Points'].describe()
```

In this table we review some of the more traditional statistics of the data such as standard deviation, mean, and quartile ranges.  Pandas has the built in [describe()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) method which makes creating this table a straightforward task.

As I described in the [narrative](https://nrasch.github.io/beginning-of-sprint-analysis#user-story-point-statistics) the team should focus on the following elements initially for improvement:

* Std
* Min
* Max

### Box Plot

```python
df['Story Points'].plot(kind="box", figsize = (6,6))
plt.show()
```

The final element of the report is the humble [box plot](https://en.wikipedia.org/wiki/Box_plot).  This figure provides a nice visual counterpoint to the tabular data just above it in the report, and makes it easy to quickly identify outliers and other items of concern such as story points with a zero value as indicated by the lower whisker range.


## Wrapping Up

In this post we explored how to programatically create a start-of-sprint analysis report utilizing Python, Panas, and Matplotlib.  The goals of the report were to present actionable data analysis on start-of-sprint metrics as well as record those metrics for future analysis and trend reporting.

One of the main benefits of creating this report programatically is that it can now be run on-demand against future sprint JIRA data extracts.  This allows us to create consistent, repeatable reports at the start of each sprint for dissemination, and gives us an apples-to-apples comparison of our Agile Scrum process improvement progress.  It also allows the Scrum team to diagnose and respond to potential issues occurring during refinement, planning, etc.

We also have the advantage of being able to leverage any of the wide range of Python data science tools which are constantly being added to and improving.  :)

And don't forget:  You can view the complete source code for this post [here](https://nbviewer.jupyter.org/urls/nrasch.github.io/assets/html/2000-01-01-Sprint-Starting-Analysis.ipynb).
