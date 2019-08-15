---
layout: post
title:  "Mini Dashboard with React and Highcharts"
tags: [ Laravel, PHP, Web Development, React ]
featured_image_thumbnail: assets/images/posts/2019/deploying-multiple-laravel-projects-remotely-on-linux_thumbnail.png
featured_image: assets/images/posts/2019/deploying-multiple-laravel-projects-remotely-on-linux_title.png
featured: true
hidden: true
---

In this post we explore creating a mini dashboard utilizing [Laravel](https://laravel.com/), [React](https://reactjs.org/), and [Highcharts](https://www.highcharts.com/).

<!--more-->

## Prerequisites and assumptions

For this post I assume you using the Application Template created in previous posts.  You can find the series here:

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

First we want to install any npm packages we'll need later on:

```bash
npm install react-loading-overlay
npm install highcharts highcharts-react-official
npm install react-modal
```

## Sample data set

Next we'll need a sample data set to work with.  For this write up we'll chose something simple:

* Source:  https://dev.mysql.com/doc/sakila/en/sakila-installation.html

Assuming you are utilizing [MySQL](https://www.mysql.com/) these would be the steps to load and make the data accessible:

### Download the data
```bash
wget https://downloads.mysql.com/docs/sakila-db.zip
unzip sakila-db.zip
```

### Load the data into SQL
Now we need to load the data.  From the terminal open a MySQL session with the command `sudo mysql`, and execute the following commands:

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

Note that if you aren't using the [Deploying Multiple Laravel Projects Remotely on Linux]({% post_url 2019-08-09-deploying-multiple-laravel-projects-remotely-on-linux %}) writeup then your application SQL user will be different than the one shown above, and you should adjust accordingly.

### Configure Laravel

We need to tell Laravel where to find the sample data as well as add the migrations we've already developed as part of the [application template]({% post_url 2019-07-26-creating-a-base-laravel-project-template %}).

#### Update .env

Edit the `.env` file and make the following changes:

```bash
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=sakila
DB_USERNAME=app_template
DB_PASSWORD=!app_template!
```

This let's Laravel interact with our new database that will hold the data for this project.

#### Reload the migrate the DATABASE

To reload the database and apply migrations execute the following command from the terminal:

```bash
terminal
php artisan migrate:refresh --seed
```

This will load in all the Laravel migrations (i.e. user tables, etc.) into the database holding the sample data.

## Implement Laravel components

Now that we have the database portion wrapped up we need to add the various Laravel components we'll require for the application.

### Routing

We'll start by adding the route for the React single page application (SPA):

Edit the `routes/web.php` file and make the following changes:

```php
Route::get('/charts', 'ChartController@index')->name('charts');
```

This will direct all `/charts` traffic to the `ChartController@index` method which will render the view that loads the React SPA.

### Nav bar element

Let's also add a navigation item to the left-hand nav bar.  Edit the `resources/views/components/left_nav.blade.php` file and make the following changes:

```php
<a href="/charts" class="list-group-item list-group-item-action bg-light"><i class="fas fa-chart-line mr-2"></i>Charts</a>
```

This will ensure users can find and browse to our new mini dashboard.

### Create the ChartController object

Next we need to create the controller that will render the view containing the React SPA.  From the command line:

```bash
php artisan make:controller ChartController
```

Once that's finished edit the new controller file `app/Http/Controllers/ChartController.php`, and modify its contents to the following:

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

This controller is responsible for not only rendering the view that contains the React SPA, but also for gathering and formatting the data required by the view.

### Add React SPA to Laravel mix

The last thing we need to do on the Laravel side is include the React SPA into our [Laravel mix](https://laravel.com/docs/5.8/mix) directives.  Edit the `resources/js/app.js` file and add the following:

```php
require('./components/Charts');
```

This will ensure our SPA code and assets are included in the compiled `js/app.js` file.

That should be the end of things we need to do in Laravel, and from here on out 99% of our work will be creating the React javascript assets.

## Develop React assets

### The SPA 'skeleton'

The first thing we want to do is create the initial SPA application component that will load and display all the other assets.  We'll call this the SPA's 'skeleton' since everything else depends on it and the structure it provides.  The skeleton will start by rendering the base HTML structure of the mini dashboard that the other React components will render into.

Create and then add following content to the 'resources/js/components/Charts.js' file:

{% raw %}
```html
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

This breaks the page up into two main panels, and then subdivides the first panel into two sections.  In each panel/section we want to load a React component that in turn provides access to a Highchart element giving insight into some facet of our sample data set.

From the HTML above we can see we are going to have three graphs on the following topics:

1. Annual Sales
2. Film Inventory by Category
3. Rental Volume Over Time

## Summary

This post has covered the steps to create a mini dashboard utilizing Laravel, React, and Highcharts.  From here the dashboard could be expanded, incorporated into a full blown Larvel/React application, or be put to use with a more sophisticated data set.

You can find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/HighchartsAndReact).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
