---
layout: post
title:  "Creating a User Management UI with Laravel and React - Part Two"
tags: [ Laravel, PHP, Web Development, React ]
featured_image_thumbnail: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_thumbnail.png
featured_image: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_title.png
featured: false
hidden: false
---

In this second post of the series we develop the 'create new user' functionality for a [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) based user management single page application (SPA) utilizing [Laravel](https://laravel.com/) and [React](https://reactjs.org/).

<!--more-->

## Series posts

For reference here are the other posts in the series:

* [Part one]({% post_url 2019-08-22-creating-a-user-management-ui-with-laravel-and-react-part-one %}) - Add the user accounts index
* [Part two]() - Add the ability to create new user accounts

## Prerequisites and assumptions

For this discussion I assume you've followed the instructions in the [first post]({% post_url 2019-08-22-creating-a-user-management-ui-with-laravel-and-react-part-one %}), and we continue by building on what we did previously.

You can also find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUserManager_PartTwo).

## Before we begin

There is a lot going on this post as we'll soon see.  As such it may be best to view the source code first, and then return to this document for commentary on the various components.  Also, for reference, here are all the files we've either modified or created for this post:

* routes/web.php
* app/Http/Controllers/UserController.php
* app/Http/Requests/UserFormRequest.php

* resources/js/components/Index.js
* resources/js/components/CreateForm.js
* resources/js/components/FlashMessage.js
* resources/js/components/FormModal.js
* resources/js/components/TableActions.js

The good news; however, is that with all of this done the next two parts in the series, editing and deleting, should be much simpler.

With that out of the way let's get started!

## Create Laravel assets

To start we'll create the Laravel assets needed to support the React components.

### Routing

First we'll add a new route for the React single page application (SPA) dealing with new user creation.

Edit the **_routes/web.php_** file and make the following changes:

```php
<?php

// Default route
Route::get('/', function () {
    return redirect( route('home'));
});

Auth::routes();

Route::get('/home', 'HomeController@index')->name('home');

Route::get('/get_users', 'UserController@getUsers');
Route::post('/create_user', 'UserController@store');
Route::get('/users', 'UserController@index')->name('users');
```

We added a new route, `/create_user`, which accepts a POST request.  This route will handle the actions that need to take place when the create new user form is submitted via React to the Laravel back end.

We chose to add this new route as a POST request, because we'd like to follow Laravel's resource controller action standards ([reference](https://laravel.com/docs/5.8/controllers#resource-controllers)).

### app/Http/Controllers/UserController.php

Let's update the user controller next.  Edit the `app/Http/Controllers/UserController.php` file and add the following code:

{% raw %}
```php
<?php

namespace App\Http\Controllers;

use App\User;
use Illuminate\Http\Request;
use Spatie\Permission\Models\Role;
use App\Http\Requests\UserFormRequest;

class UserController extends Controller
{
	/**
	* Display a listing of the resource.
	*
	* @return \Illuminate\Http\Response
	*/
	public function index()
	{
		return view('users');
	}

	/**
	* Store a newly created resource in storage.
	*
	* @param App\Http\Requests\UserFormRequest $request
	* @return \Illuminate\Http\JsonResponse
	*/
	public function store(UserFormRequest $request)
	{
		// Create the user based on request param values
		$user = User::create($request->all());

		// Assign role(s) to user
		$roles = $request->input('roles') ? $request->input('roles') : [];
		$user->assignRole($roles);

		// Create response to be returned to the view
		$response['result']['type'] = 'success';
		$response['result']['message'] = 'The user was successfully created!';
		$response['data'] = $user->__toString();

		// Return JSON response
		return response()->json($response);
	}

	/**
	* Update the specified resource in storage.
	*
	* @param  \Illuminate\Http\Request  $request
	* @param  \App\User  $user
	* @return \Illuminate\Http\Response
	*/
	public function update(Request $request, User $user)
	{
		//
	}

	/**
	* Remove the specified resource from storage.
	*
	* @param  \App\User  $user
	* @return \Illuminate\Http\Response
	*/
	public function destroy(User $user)
	{
		//
	}

	/**
	* Fetch and return a JSON array of Users
	*
	* @return \Illuminate\Http\JsonResponse
	*/
	public function getUsers()
	{
		// Use 'with' option to enable eager loading for the user roles
		$users = User::with('roles')->get();

		// Uncomment line below to simulate no data returned
		//$users = array();

		// We have a sleep here so we can observe the loading overlay in the view
		sleep(1);

		// Return JSON response
		return response()->json(['data' => $users]);
	}
}
```
{% endraw %}

We've removed much of the boiler plate code that Laravel generates, and added logic to support the React front end views.  We won't cover the code we added last post again, but we'll review what we've added since then.

#### The **_store()_** method

{% raw %}
```php
public function store(UserFormRequest $request)
{
	// Create the user based on request param values
	$user = User::create($request->all());

	// Assign role(s) to user
	$roles = $request->input('roles') ? $request->input('roles') : [];
	$user->assignRole($roles);

	// Create response to be returned to the view
	$response['result']['type'] = 'success';
	$response['result']['message'] = 'The user was successfully created!';
	$response['data'] = $user->__toString();

	// Return JSON response
	return response()->json($response);
}
```
{% endraw %}

First, this method utilizes a `UserFormRequest` Laravel form request object as a parameter to the method to encapsulate form validation.  More to follow on this below.,,,

Next the method creates a new user which is populated from the validated request form data, assigns the selected role(s) to the new user object, and then creates the JSON response which is returned to the view.

Note that if validation fails then Laravel will take care of returning a 422 status response along with whatever validation rules failed to the view.  Later on, when we develop the React components, we'll display these 'flash' messages to the user in the user interface (UI).

### app/Http/Requests/UserFormRequest.php

Next let's create the `UserFormRequest` object called in the controller to validate the user's input.  Edit the `app/Http/Requests/UserFormRequest.php` file and add the following code:

{% raw %}
```php
<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Support\Facades\Auth;

class UserFormRequest extends FormRequest
{
	/**
	 * Determine if the user is authorized to make this request.
	 *
	 * @return bool
	 */
	public function authorize()
	{
		return Auth::user()->can('manage_users');
	}

	/**
	 * Get the validation rules that apply to the request.
	 *
	 * @return array
	 */
	public function rules()
	{
		// Define general rules that will apply to all request types
		$rules = [
			'name' => 'required',
			'roles' => 'required',
		];

		// Creating a new record
		if ($this->isMethod('post')) {
			$rules = array_merge($rules,
				[
					'email' => 'required|email|unique:users,email',
					'password' => 'required|confirmed',
				]
			);
		}
		// Updating an existing record
		elseif ($this->isMethod('put')) {
			$rules = array_merge($rules,
				[
					'email' => [
							'required',
							'email',
							Rule::unique('users')->ignore($this->route('id')),
					],
					'password' => 'sometimes|confirmed',
				]
			);
		}
		// Return false for any other method
		else {
			return false;
		}

		return $rules;
	}
}
```
{% endraw %}

Let's briefly examine the two methods in this class:

#### The **_authorize()_** method

{% raw %}
```php
public function authorize()
{
	return Auth::user()->can('manage_users');
}
```
{% endraw %}

Laravel will first evaluate the **_authorize()_** method for the form request object (FRO) before proceeding.  Note that if you are utilizing [policies](https://laravel.com/docs/5.8/authorization#creating-policies) they will not be checked for controller methods utilizing the FRO.  Instead the FRO's **_authorize()_** method will be executed to calculate access control.  If the method returns `true` script execution continues, and if the method returns `false` an authorization exception will be thrown.  

This can be a gotcha; however, because many times you'll see online examples simply returning `true` in the **_authorize()_** method without taking any further action or permission checking.  This will allow access to the controller method unless blocked by a secondary method such as role checking middleware which is probably not what we want.

#### The **_rules()_** method

{% raw %}
```php
public function rules()
{
	// Define general rules that will apply to all request types
	$rules = [
		'name' => 'required',
		'roles' => 'required',
	];

	// Creating a new record
	if ($this->isMethod('post')) {
		$rules = array_merge($rules,
			[
				'email' => 'required|email|unique:users,email',
				'password' => 'required|confirmed',
			]
		);
	}
	// Updating an existing record
	elseif ($this->isMethod('put')) {
		$rules = array_merge($rules,
			[
				'email' => [
						'required',
						'email',
						Rule::unique('users')->ignore($this->route('id')),
				],
				'password' => 'sometimes|confirmed',
			]
		);
	}

	return $rules;
}
```
{% endraw %}

This method collects and then returns the validation rules for either 1) a new user object or 2) edits to an existing user object.  You can read more about Laravel validation [here](https://laravel.com/docs/5.8/validation).

That should be the end of our Laravel tasks, and from here on out the rest of our work will be creating the React javascript assets.

---

## Develop React assets

### Install npm packages

First we'll want to install the npm packages we'll need later on:

```bash
npm install formik
npm install yup
npm install lodash
npm install react-modal

npm run dev
```

This will install the following package:

* [Formik](https://jaredpalmer.com/formik/)
* [Yup](https://github.com/jquense/yup)
* [lodash](https://github.com/lodash/lodash)
* [react-modal](https://github.com/reactjs/react-modal)

Next let's examine changes to the `resources/js/components/Index.js` file.

### resources/js/components/Index.js

Due to the size of this file we won't be reproducing it here in its entirety.  Instead we'll review the changes we've made to it below.  You can view the source [here](), or you can execute the following command to view a diff summary from the last write up in this series:

```bash
git diff 3c6fa220882ae81f5387b88ce35bda9a75717ac8..85624fc42cf9e293b57b428e0a2199e29084da62 -- resources/js/components/Users.js
```

#### Index.js - Imports

```javascript
// Our custom components
import TableExportAndSearch from './TableExportAndSearch';
import TableActions from './TableActions';
import CreateForm from './CreateForm'
import FormModal from './FormModal
```

We've added three new import statements to the top of the file to load the new React assects we'll be developing below.

#### Index.js - Constructor

{% raw %}
```javascript
// Class constructor
constructor(props) {

	super(props);

	this.state = {
		// Container for data fetched from the backend
		userData: [],
		// Track if we have an outstanding call to the backend in progress
		loading: false,

		user: null,

		modalsOpen: {
			create: false,
			edit: false,
			delete: false,
		}
	};

	//Bindings
	this.fetchUserData = this.fetchUserData.bind(this);
	this.toggleModal = this.toggleModal.bind(this);
}
```
{% endraw %}

There are two main changes in the constructor

1. We added a `modalsOpen` object to the state that will allow us to track which of the create, edit, and delete modal windows are opened
2. We added a new binding for the `toggleModal` function which will be called to actually show/hide the create, edit, and delete modal windows


#### Index.js - toggleModal()

Next we've added the `toggleModal()` function:

{% raw %}
```javascript
// Show/hide the create/edit/delete modals and track which user we are taking action on
toggleModal(modal, user) {
	const currentModalState = this.state.modalsOpen[modal];

	if (this.state.modalsOpen[modal]) {
		this.fetchUserData();
	}

	// Note the brackets around the word 'modal'
	// Computed properties:  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer#Computed_property_names
	this.setState({
		modalsOpen: {
			[modal]: !currentModalState
		},
		user: user,
	});
}
```
{% endraw %}

This method toggles the shown/hidden state for each modal using computed properties, and when a modal closes it triggers the index data table to be reloaded which will pickup any new users, edits to existing users, or user deletions.

Note it also accepts a user object as the second parameter, and we'll utilize this in the future to determine which user we should apply edits to or delete by passing it to the appropriate edit/delete modal.

#### Index.js - render() :: actions constant

{% raw %}
```javascript
// Define a list of actions we want to be able to take on a given user
// These are displayed in the datatables's dummy 'Action' column which is defined below
const actions = [
	{
		title: "Edit User",
		onClick: this.toggleModal,
		modelType: 'edit',
		class: "text-secondary",
		icon: "fa fa-fs fa-pencil-alt",
	},
	{
		title: "Delete User",
		onClick: this.toggleModal,
		modelType: 'delete',
		class: "text-danger",
		icon: "fa fa-fs fa-trash",
	}
];
```
{% endraw %}

Our next set of changes occur in the `render()` method.  First we create an `actions` constant that we use to define what should happen when a user clicks the `edit` or `delete` icons we'll place in the data table rows later on for each user.  Further along in the code we'll pass this constant to the data table as part of a `user actions` [dummy column](https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/column-props.html#columnisdummyfield-bool).

#### Index.js - render() :: columns constant

Next we update the columns constant to include the new `user actions` [dummy column](https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/column-props.html#columnisdummyfield-bool).

{% raw %}
```javascript
// Define data table columns and which data fields the values should come from
// Note we also add the 'Actions' dummy column that utilizes the 'const actions' we defined above
const columns = [
	{
		dataField: 'id',
		text: 'User ID',
		sort:true,
	}, {
		dataField: 'name',
		text: 'Name',
		sort:true,
	}, {
		dataField: 'email',
		text: 'Email',
		sort:true,
	}, {
		dataField: 'roles',
		text: 'Roles',
		sort:true,
		formatter: this.roleTableFormatter,
		csvFormatter: this.roleCSVFormatter,
	}, {
		// Create a custom, dummy column with clickable icons to edit and delete the row's user
		// Example here:  https://react-bootstrap-table.github.io/react-bootstrap-table2/storybook/index.html?selectedKind=Work%20on%20Columns&selectedStory=Dummy%20Column&full=0&addons=1&stories=1&panelRight=0&addonPanel=storybook%2Factions%2Factions-panel
		dataField: 'actions',
		isDummyField: true,
		text: 'Actions',
		formatter: (cell, row) => {
			return (
				<TableActions item={ row } actions={ actions } />
			);
		},
		sort: false,
		csvExport: false,
	},
]
```
{% endraw %}





-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------
-----------------------











 Next we want to create the initial SPA application component that will load and display all the other React assets.  We'll call this the SPA's 'skeleton' since everything else depends on it and the structure it provides.  The skeleton will start by creating the base HTML of the dashboard that the other React components will render into.

Create and then add following content to the 'resources/js/components/Users.js' file:

{% raw %}
```javascript
import React, { Component } from 'react';
import ReactDOM from 'react-dom';

import Index from './Index.js'

export default class Users extends Component {
	render() {
		return (
			<div>
				<div className="row justify-content-center">
					<div className="col-12">
						{/* Load the Index component */}
						<Index />
					</div>
				</div>
			</div>
		);
	}
}

if (document.getElementById('users')) {
	ReactDOM.render(<Users />, document.getElementById('users'));
}

```
{% endraw %}

This file is fairly simple and straightforwards.  It performs the following tasks:

1. Imports the **_Index_** component which will display the list of Users and provide access to the CRUD functions
2. Builds the HTML structure the SPA resides in
3. Attaches to the user DIV element and instantiates the SPA

### Index component

The **_Index_** component is responsible for loading and displaying the list of application users as well as providing links to the create, edit, and delete functions.  

Create the **_resources/js/components/Index.js_** file and add the following code to it:

{% raw %}
```javascript
// Standard import items
import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

// Our custom components
import TableExportAndSearch from './TableExportAndSearch';

// React BootstrapTable import items
import 'react-bootstrap-table-next/dist/react-bootstrap-table2.min.css';
import 'react-bootstrap-table2-paginator/dist/react-bootstrap-table2-paginator.min.css';
import 'react-bootstrap-table2-toolkit/dist/react-bootstrap-table2-toolkit.min.css';

import BootstrapTable from 'react-bootstrap-table-next';
import paginationFactory from 'react-bootstrap-table2-paginator';
import ToolkitProvider from 'react-bootstrap-table2-toolkit';
import overlayFactory from 'react-bootstrap-table2-overlay';


export default class Index extends Component {

	// Class constructor
	constructor(props) {

		super(props);

		this.state = {
			// Container for data fetched from the backend
			userData: [],
			// Track if we have an outstanding call to the backend in progress
			loading: false,
		};

		//Bindings
		this.fetchUserData = this.fetchUserData.bind(this);
	}

	// Actions to take once the component loads
	componentDidMount() {
		this.fetchUserData();
	}

	// Fetch list of users from backend and assign to state data property
	fetchUserData() {

		// Indicate component is fetching data
		this.setState({
			loading: true,
		});

		// Make ajax call to backend to fetch user data
		axios.get('/get_users')
		.then( (response) => {
			// If data was returned assign to state userData variable
			// otherwise assign an empty array to the userData variable
			if (response.data) {
				this.setState({
					userData: response.data.data
				});
			} else {
				// No data was returned from the ajax call
				this.setState({
					userData: []
				});
			}
		})
		.catch(function (error) {
			// Log any errors
			console.log(error);
		})
		.then( () => {
			// Indicate component is finished fetching data
			this.setState({
				loading: false,
			});
		});
	}
	// END fetchUserData()

	// Show/hide the create new user modal
	toggleCreateModal() {
		//
	}

	// Wrap the user's role(s) in a bootstrap badge element
	// Called by the user data table below
	roleTableFormatter(cell, row) {
		return (
			<span>
			{ cell.map( (val, index) => {
				return <span key={index} className="badge badge-primary mr-2">{ val.name }</span>
			})}
			</span>
		)
	}

	// Extract the user's role(s) into a comma delimited string
	// Called by the user data table below
	roleCSVFormatter(cell, row, rowIndex) {
		let roles = cell.map( (val, index) => {
			return val.name;
		});

		// Note the backticks below... template literals
		// Ref:  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals
		return `${roles.join(",")}`;
	}


	// Examine this.props and this.state and return class response (typical React elements)
	render() {

		// Configure what should be shown if no data is returned from the ajax calll
		// Ref https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/overlay.html
		const NoDataIndication = () => (
			<span className="font-italic">No data found....</span>
		);

		// Define data table columns and which data fields the values should come from
		const columns = [
			{
				dataField: 'id',
				text: 'User ID',
				sort:true,
			}, {
				dataField: 'name',
				text: 'Name',
				sort:true,
			}, {
				dataField: 'email',
				text: 'Email',
				sort:true,
			}, {
				dataField: 'roles',
				text: 'Roles',
				sort:true,
				formatter: this.roleTableFormatter,
				csvFormatter: this.roleCSVFormatter,
			},

		];

		// Prepare and return React elements
		return(
			<div>
				{/* Card div */}
				<div className="card">
					<div className="card-header">User Index</div>

					{/* Card body div */}
					<div className="card-body">
						{/*
							Provide export and search features via the ToolkitProvider
							Ref:  https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/toolkits-getting-started.html
						*/}
							<ToolkitProvider
								bootstrap4={ true }
								keyField="id"
								data={ this.state.userData }
								columns={ columns }
								search
								exportCSV
							>
							{
								props => (
									<div>
										{/* Show/hide create user modal */}
										<div>
											<button type="button" className="btn btn-outline-success" onClick={this.toggleCreateModal}>
											<i className="fa fa-fw fa-plus"></i> Create User
											</button>
										</div>
										{/* END Show/hide create user modal */}

										{/* Render CSV export and search components for data table */}
										<TableExportAndSearch csvProps={ props.csvProps } searchProps= { props.searchProps } />
										{/* END Render CSV export and search components for data table */}

										{/* Create the user data table */}
										<BootstrapTable
											{ ...props.baseProps }
											loading={ this.state.loading }
											pagination={ paginationFactory() }
											striped={ true }
											bordered={ true }
											hover={ true }
											rowClasses="font-size-sm"
											noDataIndication={ () => <NoDataIndication /> }
											overlay={ overlayFactory({ spinner: true, background: 'rgba(220,220,220,0.3)', text: 'Loading....' }) }
										/>
										{/* END Create the user data table */}
									</div>
								)
							}
							</ToolkitProvider>
							{/* END ToolkitProvider */}

						</div>
						{/* END Card body div */}

					</div>
					{/* END Card div */}

				</div>
		);
	}
	// END render

}

}
```
{% endraw %}

Next let's review some of the key functions of this file.

#### The **_fetchUserData()_** method

{% raw %}
```javascript
// Fetch list of users from backend and assign to state data property
	fetchUserData() {

		// Indicate component is fetching data
		this.setState({
			loading: true,
		});

		// Make ajax call to backend to fetch user data
		axios.get('/get_users')
		.then( (response) => {
			// If data was returned assign to state userData variable
			// otherwise assign an empty array to the userData variable
			if (response.data) {
				this.setState({
					userData: response.data.data
				});
			} else {
				// No data was returned from the ajax call
				this.setState({
					userData: []
				});
			}
		})
		.catch(function (error) {
			// Log any errors
			console.log(error);
		})
		.then( () => {
			// Indicate component is finished fetching data
			this.setState({
				loading: false,
			});
		});
	}
	// END fetchUserData()
```
{% endraw %}

This function first sets the **_loading_** flag to true, which in turn renders a loading overlay onto the page.  This let's the user know something is going on in the background.  Next a GET request is made via **_axios_** to the Laravel back end to obtain a list of users.  If this completes successfully the **_userData_** variable is populated with the results.  And lastly, the **_loading_** flag is set to false which removes the loading overlay and lets the user know that fetching the data is complete.

#### The **_roleTableFormatter()_** method

{% raw %}
```javascript
roleTableFormatter(cell, row) {
	return (
		<span>
		{ cell.map( (val, index) => {
			return <span key={index} className="badge badge-primary mr-2">{ val.name }</span>
		})}
		</span>
	)
}
	// END fetchUserData()
```
{% endraw %}

As we saw earlier the user roles are contained in a nested JSON attribute in the data returned from the back end.  This method not only extracts the roles, but it also wraps them into a badge element visual component.

![User roles badge example](assets/images/posts/2019/roles-badge-example.png)

#### The **_roleCSVFormatter()_** method

{% raw %}
```javascript
roleCSVFormatter(cell, row, rowIndex) {
	let roles = cell.map( (val, index) => {
		return val.name;
	});

	// Note the backticks below... template literals
	// Ref:  https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals
	return `${roles.join(",")}`;
}
```
{% endraw %}

As saw earlier the user roles are contained in a nested JSON attribute in the data returned from the back end.  This method collects the user roles into a comma delimited string for use when exporting the data to CSV.

#### The **_render()_** method

There is a lot going on in this method, so we'll examine the components individually:

**No data**
{% raw %}
```javascript
const NoDataIndication = () => (
	<span className="font-italic">No data found....</span>
);
// END fetchUserData()
```
{% endraw %}
This code configures what the datatable should show when no data is for the table is avaiable.

**Table columns**
{% raw %}
```javascript
const columns = [
	{
		dataField: 'id',
		text: 'User ID',
		sort:true,
	}, {
		dataField: 'name',
		text: 'Name',
		sort:true,
	}, {
		dataField: 'email',
		text: 'Email',
		sort:true,
	}, {
		dataField: 'roles',
		text: 'Roles',
		sort:true,
		formatter: this.roleTableFormatter,
		csvFormatter: this.roleCSVFormatter,
	},
];
```
{% endraw %}
This code defines which columns should appear in the datatable, the column heading title, if the column is sortable, and how to format the HTML and CSV outputs for the column in the case of the `roles` elelment.


**Return the final React elements**
{% raw %}
```javascript
// Prepare and return React elements
		return(
			....
			....
			....
		)
// END render
```
{% endraw %}

The code in this section may look complicated, but it's actually rather straightforwards:

* First we define the HTML container for the React components
{% raw %}
```javascript
<div>
	{/* Card div */}
	<div className="card">
		<div className="card-header">User Index</div>

		{/* Card body div */}
		<div className="card-body">
// END fetchUserData()
```
{% endraw %}

* Next we call the `ToolkitProvider` component which provides CSV exports and searching:
{% raw %}
```javascript
<ToolkitProvider
	bootstrap4={ true }
	keyField="id"
	data={ this.state.userData }
	columns={ columns }
	search
	exportCSV
>
// END fetchUserData()
```
{% endraw %}

* We then pass a custom create user button, the table export and search components, and the actual bootstrap table itself to the `ToolkitProvider`'s props:

{% raw %}
```javascript
{
props => (
	<div>
		{/* Show/hide create user modal */}
		<div>
			<button type="button" className="btn btn-outline-success" onClick={this.toggleCreateModal}>
			<i className="fa fa-fw fa-plus"></i> Create User
			</button>
		</div>
		{/* END Show/hide create user modal */}

		{/* Render CSV export and search components for data table */}
		<TableExportAndSearch csvProps={ props.csvProps } searchProps= { props.searchProps } />
		{/* END Render CSV export and search components for data table */}

		{/* Create the user data table */}
		<BootstrapTable
			{ ...props.baseProps }
			loading={ this.state.loading }
			pagination={ paginationFactory() }
			striped={ true }
			bordered={ true }
			hover={ true }
			rowClasses="font-size-sm"
			noDataIndication={ () => <NoDataIndication /> }
			overlay={ overlayFactory({ spinner: true, background: 'rgba(220,220,220,0.3)', text: 'Loading....' }) }
		/>
		{/* END Create the user data table */}
	</div>
)
}
````
{% endraw %}

Whew!  And after all that we have one more component to write.

### Table export and search component

The **_TableExportAndSearch_** component is responsible drawing the CSV export button and search field to the datatable.  

Create the **_resources/js/components/TableExportAndSearch.js_** file and add the following code to it:

{% raw %}
```javascript
import React, {Component} from 'react';
import { Search, CSVExport } from 'react-bootstrap-table2-toolkit';

export default class TableExportAndSearch extends Component {
	render() {

		const { SearchBar } = Search;
		const { ExportCSVButton } = CSVExport;

		return (
			<div className="row text-right mb-1">
				{/* col-md-6 offset-md-9 */}
				<div className="col-md-6 offset-md-9">
					{/* inner row */}
					<div className="row mt-2">

						{/* Render CSV export button */}
						<div className="mr-5">
							<ExportCSVButton
								{	 ...this.props.csvProps}
								className="btn btn-primary form-control btn-sm"
							>
								CSV
							</ExportCSVButton>
						</div>
						{/* END Render CSV export button */}

						{/* Render table search field */}
						<div className="col-xs-3">
							<SearchBar
								{ ...this.props.searchProps }
								placeholder="Search...."
							/>
						</div>
						{/* END table search field */}

					</div>
					{/* END inner row */}
				</div>
				{/* END col-md-6 offset-md-9 */}
			</div>
		);
	}
}
```
{% endraw %}

This file simply creates an HTML structure to contain the React bootstrap table's CSV export button and search field, and then places the ExportCSVButton and SearchBar components respectively into the structure.

## The end result

To wrap up we first want to compile the code we wrote above using the following terminal command:

```bash
npm run dev
```

Once that's finished browse to your Laravel application, click the `Users` link in the navigation bar, and you should see the following:

![User manager part one final product](assets/images/posts/2019/user-index-part-one-final-product.png)

## Summary

This post has covered the first step to creating a User management SPA utilizing Laravel and React.  In the next part of the series we'll explore adding the capability to create new user accounts, and have them saved to the database.

You can find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUseManager_PartTwo).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
