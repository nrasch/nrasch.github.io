---
layout: post
title:  "Creating a Dashboard with Laravel, React, and Highcharts"
tags: [ Laravel, PHP, Web Development, React, Highcharts ]
featured_image_thumbnail: assets/images/posts/2019/creating-a-dashboard-with-laravel-react-and-highcharts_thumbnail.png
featured_image: assets/images/posts/2019/creating-a-dashboard-with-laravel-react-and-highcharts_title.png
featured: false
hidden: false
---

In this post we explore creating a dashboard utilizing [Laravel](https://laravel.com/), [React](https://reactjs.org/), and [Highcharts](https://www.highcharts.com/).

<!--more-->

## Prerequisites and assumptions

For this post I assume you're using the Application Template created in previous posts.  You can find the series here:

* [Creating a Base Laravel Project Template Part One]({% post_url 2019-07-26-creating-a-base-laravel-project-template %})
* [Deploying Multiple Laravel Projects Remotely on Linux]({% post_url 2019-08-09-deploying-multiple-laravel-projects-remotely-on-linux %})

If this isn't the case you'll likely need to slightly modify some of the commands below to match your setup.

You can also find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/HighchartsAndReact).

## The process

This is the process we'll follow:

1. Install any required [npm](https://www.npmjs.com/) packages into the base application
2. Obtain and 'install' the sample data we'll be utilizing
3. Implement the Laravel components required for the project
4. Develop the React assets required for the project

Let's get started!

## Install npm packages

First we want to install the npm packages we'll need later on:

```bash
npm install react-loading-overlay
npm install highcharts highcharts-react-official
npm install react-modal
```

## Sample data set

Next we'll need a sample data set to work with.  For this write up we'll chose something pre created, so we can focus on the code and not complex ETL processes and SQL queries:

* Source:  <https://dev.mysql.com/doc/sakila/en/sakila-installation.html>

Assuming you are utilizing [MySQL](https://www.mysql.com/) these would be the steps to load and make the data accessible:

### Download the data
```bash
wget https://downloads.mysql.com/docs/sakila-db.zip
unzip sakila-db.zip
```

### Load the data into SQL
Now we need to load the data.  From the terminal open a MySQL session with the command **_sudo mysql_**, and execute the following commands:

```bash
SOURCE <DOWNLOAD_DIR>/sakila-db/sakila-schema.sql;
SOURCE <DOWNLOAD_DIR>/sakila-db/sakila-data.sql;

USE sakila;
SHOW TABLES;
SELECT COUNT(*) FROM film;

GRANT ALL PRIVILEGES ON sakila.* TO 'app_template'@'localhost';
FLUSH PRIVILEGES;
quit;
```

Note that if you aren't using setup described in the [Deploying Multiple Laravel Projects Remotely on Linux]({% post_url 2019-08-09-deploying-multiple-laravel-projects-remotely-on-linux %}) write up then your application's SQL user will be different than the one shown above, and you should adjust accordingly.

### Configure Laravel

Next, we need to tell Laravel where to find the sample data as well as add the migrations we've already developed as part of the [application template]({% post_url 2019-07-26-creating-a-base-laravel-project-template %}).

#### Update .env

Edit the **_.env_** file and make the following changes:

```bash
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=sakila
DB_USERNAME=app_template
DB_PASSWORD=!app_template!
```

This enables Laravel to interact with our new database that will hold the back-end information for this project.

#### Reload the migrate the DATABASE

To reload the database and apply all the Laravel migrations execute the following command from the terminal:

```bash
terminal
php artisan migrate:refresh --seed
```

This will load in all the Laravel migrations (i.e. user tables, etc.) into the database holding the sample data.

## Implement Laravel components

Now that we have the database portion wrapped up we need to add the various Laravel components we'll require for the application.

### Routing

We'll start by adding the route for the React single page application (SPA):

Edit the **_routes/web.php_** file and make the following changes:

```php
Route::get('/charts/get_sales', 'ChartController@getSales');
Route::get('/charts/get_categories', 'ChartController@getCategories');
Route::get('/charts/get_rental_volume', 'ChartController@getRentalVolume');
Route::get('/charts', 'ChartController@index')->name('charts');
```

This will direct all **_/charts_** traffic to the **_ChartController@index_** method which will render the view that in turn loads the React SPA.  We also add a number of pseudo api calls, **_/get_sales_**, **_/get_categories_**, etc., to the route, which we'll utilize to return data to the React components later on.

### Navigation menu element

Let's also add a navigation item to the left-hand navigation bar.  Edit the **_resources/views/components/left_nav.blade.php_** file and make the following changes:

```php
<a href="/charts" class="list-group-item list-group-item-action bg-light"><i class="fas fa-chart-line mr-2"></i>Charts</a>
```

This will ensure users can find and browse to our new dashboard.

### Create the ChartController object

Next we need to create the controller that will render the view containing the React SPA.  From the command line:

```bash
php artisan make:controller ChartController
```

Once that's finished edit the new controller file **_app/Http/Controllers/ChartController.php_**, and modify its contents to the following:

{% raw %}
```php
<?php
namespace App\Http\Controllers;

use Illuminate\Support\Facades\DB;
use Illuminate\Http\Request;

class ChartController extends Controller
{

    /**
     * Highcharts options common to all charts/graphs
     *
     * @var array
     */
    private $common_options = array();

    /**
     * Highcharts line graph options
     *
     * @var array
     */
    private $line_options = array();


    /**
     * Class constructor
     */
    public function __construct()
    {
        // Set common Highchart options shared by all graphs/charts
        $this->common_options['Alberta'] = [
            'color' => '#434348'
        ];
        $this->common_options['QLD'] = [
            'color' => '#7cb5ec'
        ];

        // Set Highchart options for line graphs
        $this->line_options['Alberta'] = array_merge($this->common_options['Alberta'], [
            'dashStyle' => 'Dash'
        ]);
        $this->line_options['QLD'] = array_merge($this->common_options['QLD'], [
            'dashStyle' => 'solid'
        ]);
    }


    /**
     * Return the 'charts' view which will load the React SPA
     *
     * @return \Illuminate\View\View|\Illuminate\Contracts\View\Factory
     */
    public function index()
    {
        return view('charts');
    }


    /**
     * Pull the sales data from the database, format, and return as JSON
     *
     * @param Request $request
     * @return \Illuminate\Http\Response
     */
    public function getSales(Request $request)
    {
        // Create query
        $query = DB::table('payment')->select(DB::raw("sum(payment.amount) AS 'Sales', DATE_FORMAT(payment.payment_date, '%Y-%m') AS 'Date', address.district as 'District'"))
            ->join('customer', 'payment.customer_id', '=', 'customer.customer_id')
            ->join('store', 'customer.store_id', '=', 'store.store_id')
            ->join('address', 'store.address_id', '=', 'address.address_id')
            ->groupBy('address.district', DB::raw("DATE_FORMAT(payment.payment_date, '%Y-%m')"))
            ->orderBy('address.district')
            ->orderByRaw("DATE_FORMAT(payment.payment_date, '%Y-%m')");

        // Apply any filters from user
        if ($request->input('district')) {
            $query->where('address.district', '=', $request->input('district'));
        }

        // Execute query
        $sales = $query->get();

        // Return JSON response
        return response()->json($this->prepData($sales, 'Date', 'District', 'Sales', $this->line_options))
            ->setEncodingOptions(JSON_NUMERIC_CHECK);
    }

    /**
     * Pull the category data from the database, format, and return as JSON
     *
     * @param Request $request
     * @return \Illuminate\Http\Response
     */
    public function getCategories(Request $request)
    {
        // Create query
        $query = DB::table('category')->select(DB::raw("address.district AS 'District', category.name as 'Category', COUNT(inventory.inventory_id) AS 'Count'"))
            ->join('film_category', 'category.category_id', '=', 'film_category.category_id')
            ->join('inventory', 'film_category.film_id', '=', 'inventory.film_id')
            ->join('store', 'inventory.store_id', '=', 'store.store_id')
            ->join('address', 'store.address_id', '=', 'address.address_id')
            ->groupBy('address.district', 'category.name')
            ->orderBy('address.district')
            ->orderBy('category.name');

        // Apply any filters from user
        if ($request->input('district')) {
            $query->where('address.district', '=', $request->input('district'));
        }

        // Execute query
        $counts = $query->get();

        // Return JSON response
        return response()->json($this->prepData($counts, 'Category', 'District', 'Count', $this->common_options))
            ->setEncodingOptions(JSON_NUMERIC_CHECK);
    }

    /**
     * Pull the rental volume data from the database, format, and return as JSON
     *
     * @param Request $request
     * @return \Illuminate\Http\Response
     */
    public function getRentalVolume(Request $request)
    {
        // Create query
        $query = DB::table('rental')->select(DB::raw("STR_TO_DATE(CONVERT(rental.rental_date, char), '%Y-%m-%d') as 'Date', COUNT(rental_id) as 'Count'"))
            ->join('inventory', 'rental.inventory_id', '=', 'inventory.film_id')
            ->join('store', 'inventory.store_id', '=', 'store.store_id')
            ->join('address', 'store.address_id', '=', 'address.address_id')
            ->groupBy(DB::RAW("STR_TO_DATE(CONVERT(rental.rental_date, char), '%Y-%m-%d')"))
            ->orderBy('Date');

        // Apply any filters from user
        if ($request->input('district')) {
            $query->where('address.district', '=', $request->input('district'));
        }

        // Execute query
        $counts = $query->get();

        // Highcharts needs time series data in milliseconds, so convert
        $data = $counts->map(function ($item, $key) {
            return [
                strtotime(substr($item->Date, 0, 10)) * 1000,
                $item->Count
            ];
        });

        // Create the Hightchart series object
        $series = [
            'type' => 'area',
            'name' => 'Sales Volume',
            'data' => $data
        ];

        // Return JSON response
        return response()->json([
            'data' => [
                'series' => $series
            ]
        ])->setEncodingOptions(JSON_NUMERIC_CHECK);
    }


    /**
     * Takes a database query collection and transforms it into a Highchart series format
     *
     * @param \Illuminate\Support\Collection $collection    Database query collection
     * @param string $pluck_column                          Column in the collection to group the data by
     * @param string $pluck_value                           Values in the collection to extract
     * @param string $graph_options                         Highchart graph/chart options to apply
     * @return array
     */
    private function prepData($collection, $category_column, $pluck_column, $pluck_value, $graph_options)
    {
        // Assign these to the class, so we can access them in the closure below
        $this->pluck_column = $pluck_column;
        $this->pluck_value = $pluck_value;

        // Collect the category names for the Highchart series
        $cats = $collection->pluck($category_column)->unique();

        // Create the Highchart series data element component
        $data = $collection->mapToGroups(function ($item, $key) {
            return [
                $item->{$this->pluck_column} => $item->{$this->pluck_value}
            ];
        });

        // Create the Highchart series element
        $series = array();
        foreach ($data->keys() as $key) {
            $tmp['name'] = $key;
            $tmp['data'] = $data[$key];
            $series[] = array_merge($tmp, $graph_options[$key]);
        }

        // Return the final data element with category and series components
        return ['data' => ['categories' => $cats, 'series' => $series]];
  }

}
```
{% endraw %}

This controller is responsible for not only returning the view that renders the React SPA, but also for gathering and formatting the data required by the view.

Since there is a lot going on in this controller let's examine some highlights:

#### The **_construct()_** method

{% raw %}
```php
public function __construct()
    {
        // Set common Highchart options shared by all graphs/charts
        $this->common_options['Alberta'] = [
            'color' => '#434348'
        ];
        $this->common_options['QLD'] = [
            'color' => '#7cb5ec'
        ];

        // Set Highchart options for line graphs
        $this->line_options['Alberta'] = array_merge($this->common_options['Alberta'], [
            'dashStyle' => 'Dash'
        ]);
        $this->line_options['QLD'] = array_merge($this->common_options['QLD'], [
            'dashStyle' => 'solid'
        ]);
    }
```
{% endraw %}

This method creates and holds the options for the Highcharts graphs displayed in the view.  For example we are controlling the colors used in the charts, as well as the line types for the line graphs.

#### The **_getSales_**, **_getCategories_**, and **_getRentalVolume_** methods

{% raw %}
```php
public function getSales(Request $request)
    {
        // Create query
        $query = DB::table('payment')->select(DB::raw("sum(payment.amount) AS 'Sales', DATE_FORMAT(payment.payment_date, '%Y-%m') AS 'Date', address.district as 'District'"))
            ->join('customer', 'payment.customer_id', '=', 'customer.customer_id')
            ->join('store', 'customer.store_id', '=', 'store.store_id')
            ->join('address', 'store.address_id', '=', 'address.address_id')
            ->groupBy('address.district', DB::raw("DATE_FORMAT(payment.payment_date, '%Y-%m')"))
            ->orderBy('address.district')
            ->orderByRaw("DATE_FORMAT(payment.payment_date, '%Y-%m')");

        // Apply any filters from user
        if ($request->input('district')) {
            $query->where('address.district', '=', $request->input('district'));
        }

        // Execute query
        $sales = $query->get();

        // Return JSON response
        return response()->json($this->prepData($sales, 'Date', 'District', 'Sales', $this->line_options))
            ->setEncodingOptions(JSON_NUMERIC_CHECK);
    }
```
{% endraw %}

Since we don't have an API component in this project the **_getSales_**, **_getCategories_**, and **_getRentalVolume_** methods act as such.  So for example in the **_getSales_** method above we are creating the Eloquent ORM instance, applying any filters selected by the user, and then returning the data as formatted JSON for direct use by Highchart graph's  series component.

If you examine the [code](https://github.com/nrasch/AppTemplate/blob/HighchartsAndReact/app/Http/Controllers/ChartController.php) for the **_getCategories_** and **_getRentalVolume_** methods you'll see they follow the same pattern.

####  The **_prepData()_** method

{% raw %}
```php
private function prepData($collection, $category_column, $pluck_column, $pluck_value, $graph_options)
    {
        // Assign these to the class, so we can access them in the closure below
        $this->pluck_column = $pluck_column;
        $this->pluck_value = $pluck_value;

        // Collect the category names for the Highchart series
        $cats = $collection->pluck($category_column)->unique();

        // Create the Highchart series data element component
        $data = $collection->mapToGroups(function ($item, $key) {
            return [
                $item->{$this->pluck_column} => $item->{$this->pluck_value}
            ];
        });

        // Create the Highchart series element
        $series = array();
        foreach ($data->keys() as $key) {
            $tmp['name'] = $key;
            $tmp['data'] = $data[$key];
            $series[] = array_merge($tmp, $graph_options[$key]);
        }

        // Return the final data element with category and series components
        return ['data' => ['categories' => $cats, 'series' => $series]];
  }
```
{% endraw %}

This is a helper method to made our code a little [DRYer](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).  It takes the collection object returned from the query and converts it into a structure directly usable by the Highchart series object.  Assuming your are running this locally you can browse to <http://localhost:8080/charts/get_sales> to see how the helper has structured the data and the JSON for the annual sales chart's usage:

```json
{"data":
    {
        "categories":["2005-05","2005-06","2005-07","2005-08","2006-02"],
        "series":[
            {
                "name":"Alberta",
                "data":[2694.62,5148.57,15739.22,13136.09,283.02],
                "color":"#434348",
                "dashStyle":"Dash"
            },
            {
                "name":"QLD",
                "data":[2129.81,4483.31,12634.67,10936.04,231.16],
                "color":"#7cb5ec",
                "dashStyle":"solid"
            }
        ]
    }
}
```

This can be fed directly into Highcharts with no processing required on the client side.

### Create the Laravel view

Next we need to create the Laravel blade file that will be shown to the user when they browse to the **_/charts_** URL.  Edit the **_resources/views/charts.blade.php_** file and add the following:

{% raw %}
```php
@extends('layouts.app')

@section('content')
  <div class="container-fluid">
    <div class="mt-5" id="react-charts" />
  </div>
@endsection

@section('js_after')
  <script type="text/javascript">
    $(document).ready(function() {
        //
    } );
  </script>
@endsection
```
{% endraw %}

As can be seen the only job of this file is to provide a named DIV element for the React application to attach to and then load in on.  We also provided a placeholder for any custom javascript we want executed when the page loads.

### Add React SPA to Laravel mix

The last thing we need to do on the Laravel side of things is to include the React SPA into our [Laravel mix](https://laravel.com/docs/5.8/mix) directives.  Edit the **_resources/js/app.js_** file and add the following:

```php
require('./components/Charts');
```

This will ensure our SPA code and assets are included in the compiled **_js/app.js_** file.

That should be the end of our Laravel tasks, and from here on out the rest of our work will be creating the React javascript assets.

---

## Develop React assets

### The SPA 'skeleton'

The first thing we want to do is create the initial SPA application component that will load and display all the other React assets.  We'll call this the SPA's 'skeleton' since everything else depends on it and the structure it provides.  The skeleton will start by creating the base HTML of the dashboard that the other React components will render into.

Create and then add following content to the 'resources/js/components/Charts.js' file:

{% raw %}
```javascript
import React, { Component } from 'react';
import ReactDOM from 'react-dom';

import SalesChart from './SalesChart';
import CategoryChart from './CategoryChart';
import RentalsChart from './RentalsChart';

export default class Charts extends Component {

    constructor(props) {
  		super(props);
  	}

    render() {
        return (
          <div>
            <div className="row justify-content-center">

              <div className="col-6">
                <div className="card">
                  <div className="card-header">
                    <i className="fas fa-chart-pie"></i>
                    <span className="ml-2">Annual Sales</span>
                  </div>
                  <div className="card-body">
                    <SalesChart />
                  </div>
                </div>
              </div>

              <div className="col-6">
                <div className="card">
                  <div className="card-header">
                    <i className="fas fa-chart-pie"></i>
                    <span className="ml-2">Film Inventory by Category</span>
                  </div>
                  <div className="card-body">
                    <CategoryChart />
                  </div>
                </div>
              </div>

            </div>

            <div className="row justify-content-center mt-4">

              <div className="col-12">
                <div className="card">
                  <div className="card-header">
                    <i className="fas fa-chart-pie"></i>
                    <span className="ml-2">Rental Volume Over Time</span>
                  </div>
                  <div className="card-body">
                    <RentalsChart />
                  </div>
                </div>
              </div>

            </div>

        </div>

        );
        // END return
    }
    // END render()
}

if (document.getElementById('react-charts')) {
    ReactDOM.render(<Charts />, document.getElementById('react-charts'));
}
```
{% endraw %}

This breaks the page up into two main panels using the [bootstrap grid system](https://getbootstrap.com/docs/4.0/layout/grid/), and then subdivides the first panel into two subsections.  In each panel/section we want to load a React component that in turn provides access to a Highchart element giving insight into some facet of the sample data set.

So for example here is what the layout looks like before we insert the graph components:

![Dashboard SPA base structure](assets/images/posts/2019/chart_spa_base_structure.png)

From the HTML and screenshot above we can see we are going to have three graphs:

1. Annual Sales
2. Film Inventory by Category
3. Rental Volume Over Time

And each of these graphs will be React components which are loaded by the import statements at the top of the page:

```javascript
import SalesChart from './SalesChart';
import CategoryChart from './CategoryChart';
import RentalsChart from './RentalsChart';
```

### Graph components base class

Since each of the React graph components are going to perform similar tasks we can create a base class that encapsulates shared functionality.  We can then inherit this parent object and extend the shared functionality to incorporate whatever is required for each specific chart.

Create the **_resources/js/components/BaseChart.js_** file and add the following code to it:

{% raw %}
```javascript
import React, { Component } from 'react';
import { render } from 'react-dom';
import HighchartsReact from 'highcharts-react-official';
import Highcharts from 'highcharts';
import LoadingOverlay from 'react-loading-overlay';

import FormModal from './Common/FormModal';

require('highcharts/modules/exporting')(Highcharts)
require('highcharts/modules/export-data')(Highcharts)

export default class BaseChart extends Component {

    /**
    * Class constructor
    */
    constructor(props) {
        super(props);

        this.state = {
            // Show/hide the chart overlay on ajax requests to notify the user activity is happening
            showOverlay: false,
            // Show/hide graph filter options modal
            isFilterModalOpen: false,
            // District chart filter value
            districtFilter: 0,
            // Tracks if a filter has been selectec by the user which requires the chart data to be updated
            needDataUpdate: true,
            // Set filter modal title and content label
            modalTitle: "PLACEHOLDER",
            modalContentLabel: "PLACEHOLDER",
        };
        // END this.state = {

        // Bindings
        this.toggleFilterModal = this.toggleFilterModal.bind(this);
        this.saveFilter = this.saveFilter.bind(this);
    }
    // END constructor(props) {

    /**
    * Actions to take once the component has mounted
    */
    componentDidMount() {
        this.refreshData();
    }

    /**
    * Shows/hides the chart filter modal
    */
    toggleFilterModal() {

        // If a filter has been selected refresh the chart data
        if (this.state.isFilterModalOpen && this.state.needDataUpdate) {
            this.refreshData();
        }

        // Toggle the modal
        this.setState({
            isFilterModalOpen: !this.state.isFilterModalOpen,
        });
    }

    /**
    * Save any user selected filters in the state
    */
    saveFilter(event) {
        this.setState({
            // Utilize computed property names
            [event.target.id]: event.target.value,
            needDataUpdate: true
        });
    }

    // Create the HTML to be drawn on the page
    render() {
        const { chartOptions } = this.state;

        return (
            <div>
                {/* Form overlay to visually indicate activity is occurring to the user */}
                <LoadingOverlay
                    active={this.state.showOverlay}
                    spinner
                    text='Working...'
                    >
                    {/* Render Highchart graph */}
                    <HighchartsReact
                        highcharts={Highcharts}
                        options={chartOptions}
                        />
                    <button className="mt-3 btn btn-primary" onClick={this.toggleFilterModal}>
                        <i className="fas fa-bars mr-3"></i>
                        Chart filter options
                    </button>
                    {/* END Render Highchart graph */}
                </LoadingOverlay>
                {/* END Form overlay to visually indicate activity is occurring to the user */}

                {/* Graph filter options modal */}
                <div>
                    <FormModal
                        isOpen={this.state.isFilterModalOpen}
                        onRequestClose={this.toggleFilterModal}
                        contentLabel={this.state.modalContentLabel}
                        title={this.state.modalTitle}
                        modalAppElement="#react-charts"
                        styleOverride={ new Object({width: '40%', left: '35%',}) }
                        >
                        {/* Graph filter options form */}
                        <form>
                            <div className="form-group">
                                <label className="mr-sm-2" htmlFor="districtFilter">Store</label>
                                <select className="custom-select mr-sm-2 col-2" id="districtFilter" name="districtFilter" value={ this.state.districtFilter } onChange={ this.saveFilter }>
                                    <option value="0">All Stores</option>
                                    <option value="Alberta">Alberta</option>
                                    <option value="QLD">QLD</option>
                                </select>
                            </div>
                            <button className="btn btn-primary mb-3" onClick={this.toggleFilterModal}>Apply</button>
                        </form>
                        {/* END Graph filter options form */}

                    </FormModal>
                </div>
                {/* END Graph filter options modal */}
            </div>
        )
    }
    // END render()
}
```
{% endraw %}

This base class is rather straight forward, and the inline code comments should explain what's occurring.  In a nutshell:

1. First the class defines the component's **_state_** values that will control the chart filter modal, the values of the chart's filter options, and if the chart's data needs to be refreshed
2. Next the **_componentDidMount()_** method causes a call to the back-end to occur, so that chart's data can be fetched and loaded
3. The **_toggleFilterModal()_** method controls showing and hiding the filter modal, as well as making a call to refresh the chart's data if a filter option was selected by the user
4. Next the **_saveFilter(event)_** method utilizes computed property names to dynamically assign whichever filter the user selects into the appropriate state variable
5. And finally the **_render()_** method:
  * Draws a loading overlay over the chart element to indicate data is being requested from the backend to the user
  * Displays the actual Highchart graph element
  * Creates the filter modal along with the filter selection elements to display to the user

The **_FormModal_** element in the **_render()_** method is a simple, reusable component to display a modal on the screen, and we'll explore it next.


### FormModal component

In the base class above we made reference to a **_FormModal_** component which we'd like to use to display the chart filtering options to the user.  Let's go ahead and build this component now.  Edit the **_resources/js/components/Common/FormModal.js_** file and add the following code:


{% raw %}
```javascript
import React from 'react';
import ReactDOM from 'react-dom';
import Modal from 'react-modal';

// Modal settings
Modal.defaultStyles.overlay.backgroundColor = 'rgba(0,0,0,0.4)';
const modalContentStyle = {
  overlay: {
    zIndex: 1000
  },
  content : {
    top                 : '25%',
    left                : '27%',
    right               : 'auto',
    bottom              : 'auto',
    marginRight         : '-50%',
    width				: '60%',
    marginTop			: '-50px',
    marginLeft			: '-50px',
    backgroundColor		: '#fefefe',
  }
};
// END Modal settings

export default class FormModal extends React.Component {

  constructor(props) {
    super(props);

    // Assign the modal to an element for screen readers, etc.
    Modal.setAppElement(props.modalAppElement);

    // Make a deep copy of the modalContentStyle object
    let modalStyle = JSON.parse(JSON.stringify(modalContentStyle));

    // If there were any supplied modal style overrides apply them
    if (props.styleOverride) {
      for (var key in props.styleOverride) {
        modalStyle.content[key] = props.styleOverride[key];
      }
    }

    // Assign the final modal styles to the state, so they can be applied to newly created modal(s)
    this.state = {
      modalContentStyle: modalStyle,
    }
  }

  render() {
    // Modal is being shown; render contents
    return(
      <Modal
        isOpen={this.props.isOpen}
        onRequestClose={this.props.onRequestClose}
        style={this.state.modalContentStyle}
        contentLabel={this.props.title}
        title={this.props.title}
      >
        <div className="block block-themed block-transparent mb-0">
          <div className="block-header bg-primary-dark">
            <h3 className="block-title">{ this.props.title }</h3>
          </div>
          <div className="block-content font-size-sm">
            {this.props.children}
          </div>
          <div className="block-content block-content-full text-right border-top">
          </div>
        </div>
      </Modal>
    );
  }
}
```
{% endraw %}


Most of the code in the file deals with assigning the default CSS to the modal, and then overwriting the defaults with any custom CSS passed to the **_Modal_** object as **_props_**:

*  We first define the default CSS values as a constant:

{% raw %}
```javascript
// Modal settings
Modal.defaultStyles.overlay.backgroundColor = 'rgba(0,0,0,0.4)';
const modalContentStyle = {
  overlay: {
    zIndex: 1000
  },
  content : {
    top                 : '25%',
    left                : '27%',
    right               : 'auto',
    bottom              : 'auto',
    marginRight         : '-50%',
    width				: '60%',
    marginTop			: '-50px',
    marginLeft			: '-50px',
    backgroundColor		: '#fefefe',
  }
};
// END Modal settings
```
{% endraw %}

*  We then make a copy of the immutable constant in the class constructor, apply any styling overrides, and assign the result to the state:

{% raw %}
```javascript
  constructor(props) {
    super(props);

    // Assign the modal to an element for screen readers, etc.
    Modal.setAppElement(props.modalAppElement);

    // Make a deep copy of the modalContentStyle object
    let modalStyle = JSON.parse(JSON.stringify(modalContentStyle));

    // If there were any supplied modal style overrides apply them
    if (props.styleOverride) {
      for (var key in props.styleOverride) {
        modalStyle.content[key] = props.styleOverride[key];
      }
    }

    // Assign the final modal styles to the state, so they can be applied to newly created modal(s)
    this.state = {
      modalContentStyle: modalStyle,
    }
  }
```
{% endraw %}

*  And finally we instantiate an instance of the **_react-modal_** component, build the bootstrap HTML structure, and render any children that were passed as props:

{% raw %}
```javascript
render() {
  // Modal is being shown; render contents
  return(
    <Modal
      isOpen={this.props.isOpen}
      onRequestClose={this.props.onRequestClose}
      style={this.state.modalContentStyle}
      contentLabel={this.props.title}
      title={this.props.title}
    >
      <div className="block block-themed block-transparent mb-0">
        <div className="block-header bg-primary-dark">
          <h3 className="block-title">{ this.props.title }</h3>
        </div>
        <div className="block-content font-size-sm">
          {this.props.children}
        </div>
        <div className="block-content block-content-full text-right border-top">
        </div>
      </div>
    </Modal>
  );
}
```
{% endraw %}

### The annual sales chart

With base class and modal components finished we can now develop the charts.  We'll start with the annual sales graph:  Edit the **_resources/js/components/SalesChart.js_** file and add the following code:

{% raw %}
```javascript
import BaseChart from './BaseChart';

export default class SalesChart extends BaseChart {

    /**
    * Class constructor
    */
    constructor(props) {
        super(props);

        this.state = {
            // Keeping the Highchart options in the state to avoid unnecessary updates
            // as per the Highchart recommendations
            //
            // Highchart API reference: https://api.highcharts.com/highcharts/
            chartOptions: {
                chart: {
                    height: 300,
                    scrollablePlotArea: {
                        minWidth: 700
                    }
                },

                title: {
                    text: 'Annual sales'
                },

                subtitle: {
                    text: 'Source: https://dev.mysql.com/doc/sakila/en/'
                },

                xAxis: {},

                yAxis: [{ // left y axis
                    title: {
                        text: 'Total Sales per Year'
                    },
                    labels: {
                        align: 'left',
                        x: 3,
                        y: 16,
                        format: '${value:.,0f}'
                    },
                    showFirstLabel: false,
                }],

                legend: {
                    align: 'left',
                    verticalAlign: 'top',
                    borderWidth: 0
                },

                tooltip: {
                    shared: true,
                    crosshairs: true
                },

                plotOptions: {
                    series: {
                        cursor: 'pointer',
                        point: {
                            events: {}
                        },
                        marker: {
                            lineWidth: 1
                        }
                    }
                },

                series: [],

                exporting: {
                    enabled: true,
                    buttons: {
                        contextButton: {
                            menuItems: ['downloadPNG', 'downloadJPEG', 'downloadPDF', 'separator', 'downloadCSV', 'downloadXLS']
                        },
                    },
                },

            },

            // Set filter modal title and content label
            modalTitle: "Annual sales graph filter options",
            modalContentLabel: "Annual sales graph filter options",
        }
        // END this.state
    }
    // END constructor(props)

    /**
    * Make an ajax call to the backend to fetch data for the graph
    */
    refreshData() {

        // Show the overlay while the ajax request is processing
        this.setState({
            showOverlay: true,
        });

        // Utilize axios to make the ajax call to the backend
        axios.get('/charts/get_sales', {
            // Include any query filters
            params: {
                district: this.state.districtFilter,
            }
        })
        .then(response => {
            if (response.data.data) {
                this.setState({
                    // Update the chart's series which will refresh it
                    chartOptions: {
                        series: response.data.data.series,
                        xAxis: {
                            categories: response.data.data.categories,
                        },
                    }
                });
            } else {
                this.setState({
                    chartOptions: {
                        series: [],
                        xAxis: {}
                    }
                });
            }
        })
        .catch(function (error) {
            console.log(error);
        })
        .then( () => {
            // Hide the ajax processing overlay
            this.setState({
                showOverlay: false,
                needDataUpdate: false,
            });
        });
    }
    // END refreshData()
}
```
{% endraw %}

There are two main parts to this script:

1. Configuring the options for the Highchart
2. Refreshing the data

#### Configuring the options for the Highchart

As per [Highchart's recommendations](https://github.com/highcharts/highcharts-react#optimal-way-to-update) the chart options are kept in the component's state:

Example:

{% raw %}
```javascript
this.state = {
  // To avoid unnecessary update keep all options in the state.
  this.state = {
    // To avoid unnecessary update keep all options in the state.
    chartOptions: {
      xAxis: {
        categories: ['A', 'B', 'C'],
      },
      series: [
        { data: [1, 2, 3] }
      ],
      plotOptions: {
        series: {
          point: {
            events: {
              mouseOver: this.setHoverData.bind(this)
            }
          }
        }
      }
    },
    hoverData: null
  };
```
{% endraw %}

We've followed this recommendation, and the **_chartOptions_** value is populated with the configuration for the chart we wish to display.  You can find the full set of chart options [here](https://api.highcharts.com/highcharts/).

#### Refreshing the data

The **_refreshData()_** method:

1)  Displays the loading overlay to let the user know the chart is attempting to refresh
{% raw %}
```javascript
// Show the overlay while the ajax request is processing
this.setState({
    showOverlay: true,
});
```
{% endraw %}

2)  Utilizes **_axios_** to fetch new data for the graph and includes any user selected filtering options
{% raw %}
```javascript
axios.get('/charts/get_sales', {
    // Include any query filters
    params: {
        district: this.state.districtFilter,
    }
})
```
{% endraw %}

3)  Assigns the response to the **_chartOptions -> series_** state value which triggers the chart to refresh
{% raw %}
```javascript
if (response.data.data) {
    this.setState({
        // Update the chart's series which will refresh it
        chartOptions: {
            series: response.data.data.series,
            xAxis: {
                categories: response.data.data.categories,
            },
        }
    });
}
```
{% endraw %}


### The inventory and sales volume charts

The next two charts, **_Film Inventory by Category_** and **_Rental Volume Over Time_**, follow the same pattern:

1. Define chart options
2. Make an ajax call to obtain data for chart

Because of this we won't display the source code here, but you can view the source React for them [here](https://github.com/nrasch/AppTemplate/blob/HighchartsAndReact/resources/js/components/CategoryChart.js) and [here](https://github.com/nrasch/AppTemplate/blob/HighchartsAndReact/resources/js/components/RentalsChart.js) respectively.

## Wrapping up

And with that we are finished.  Compile the javascript assets with **_npm run dev_**, reload the Laravel instance, and you should see the following:

![Dashboard SPA view one](assets/images/posts/2019/react_dashboard_view_one.png)

---

![Dashboard SPA view two](assets/images/posts/2019/react_dashboard_view_two.png)

## Summary

This post has covered the steps to create a dashboard utilizing Laravel, React, and Highcharts.  From here the dashboard could be expanded, incorporated into a full blown Larvel/React application, or be put to use with a more sophisticated data set.

You can find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/HighchartsAndReact).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
