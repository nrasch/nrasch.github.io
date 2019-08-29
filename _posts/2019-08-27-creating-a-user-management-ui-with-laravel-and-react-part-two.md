---
layout: post
title:  "Creating a User Management UI with Laravel and React - Part Two"
tags: [ Laravel, PHP, Web Development, React ]
featured_image_thumbnail: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_thumbnail.png
featured_image: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_title.png
featured: false
hidden: false
---

In this second post of the series we add the ability to create new users to our [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) based user management single page application (SPA) utilizing [Laravel](https://laravel.com/) and [React](https://reactjs.org/).

<!--more-->

## Series posts

Posts in this series:

* [Part one]({% post_url 2019-08-22-creating-a-user-management-ui-with-laravel-and-react-part-one %}) - Initial setup with user accounts data table
* [Part two]({% post_url 2019-08-27-creating-a-user-management-ui-with-laravel-and-react-part-two %}) - Adding the ability to create new users
* [Part three]({% post_url 2019-08-29-creating-a-user-management-ui-with-laravel-and-react-part-three %}) - Adding the ability to edit existing users

With more to follow...

## Prerequisites and assumptions

For this discussion I assume you've followed the instructions in the [first post]({% post_url 2019-08-22-creating-a-user-management-ui-with-laravel-and-react-part-one %}), and we continue by building on what we did previously.

You can also find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUserManager_PartTwo).

## Before we begin

There is a lot going on this post as we'll soon see.  As such it may be best to view the [source code])(https://github.com/nrasch/AppTemplate/tree/ReactUserManager_PartTwo) first, and then return to this document for commentary on the various components.  Also, for reference, here are all the files we've either modified or created for this post:

**Laravel**
* routes/web.php
* app/Http/Controllers/UserController.php
* app/Http/Requests/UserFormRequest.php

**React**
* resources/js/components/Index.js
* resources/js/components/CreateForm.js
* resources/js/components/FlashMessage.js
* resources/js/components/FormModal.js
* resources/js/components/TableActions.js

The good news; however, is that once we finish this the next two parts in the series, editing and deleting, should be much simpler.

With that out of the way let's get started!

## Create Laravel assets

To start we'll create the Laravel assets needed to support the React components.

### Routing

First we'll add the new routes for the React single page application (SPA) dealing with user creation.

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

We added a new route, **_/create_user_**, which accepts a POST request.  This route will handle the actions that need to take place when the create new user form is submitted via React to the Laravel back end.

We chose to add this new route as a POST request, because we'd like to follow Laravel's resource controller action standards ([reference](https://laravel.com/docs/5.8/controllers#resource-controllers)).

### app/Http/Controllers/UserController.php

Let's update the user controller next.  Edit the **_app/Http/Controllers/UserController.php_** file and add the following code:

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

This method utilizes a **_UserFormRequest_** Laravel form request object as a parameter to the method for input validation.  More to follow on this below.,,,

Next the method creates a new user which is populated from the validated request form data, assigns the selected role(s) to the new user object, and then creates the JSON response which is returned to the view.

Note that if validation fails then Laravel will take care of returning a **_422_** status response along with whatever validation rules failed to the view.  Later on, when we develop the React components, we'll display these 'flash' messages to the user in the user interface (UI).

### app/Http/Requests/UserFormRequest.php

Next let's create the **_UserFormRequest_** object called in the controller to validate the user's input.  Edit the **_app/Http/Requests/UserFormRequest.php_** file and add the following code:

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
							Rule::unique('users')
							->ignore($this->route('id')),
					],
					'password' => 'sometimes|confirmed',
				]
			);
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

Laravel will first evaluate the **_authorize()_** method for the form request object (FRO) before proceeding.  Note that if you are utilizing [policies](https://laravel.com/docs/5.8/authorization#creating-policies) they will not be checked for controller methods utilizing the FRO.  Instead the FRO's **_authorize()_** method will be executed to calculate access control.  If the method returns **_true_** script execution continues, and if the method returns **_false_** an authorization exception will be thrown.  

This can be a gotcha; however, because many times you'll see online examples simply returning **_true_** in the **_authorize()_** method without taking any further action or permission checking.  This will allow access to the controller method unless blocked by a secondary method such as role checking middleware, which is probably not what we want.

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
						Rule::unique('users')
						->ignore($this->route('id')),
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

This will install the following packages:

* [Formik](https://jaredpalmer.com/formik/)
* [Yup](https://github.com/jquense/yup)
* [lodash](https://github.com/lodash/lodash)
* [react-modal](https://github.com/reactjs/react-modal)

Next let's examine changes to the **_resources/js/components/Index.js_** file.

### resources/js/components/Index.js

Due to the size of this file we won't be reproducing it here in its entirety.  Instead we'll review the changes we've made to it below.  You can view the source [here](https://github.com/nrasch/AppTemplate/blob/ReactUserManager_PartTwo/resources/js/components/Index.js), or you can execute the following command to view a diff summary from the last write up in this series:

```bash
git diff 3c6fa220882ae81f5387b88ce35bda9a75717ac8..6c4801f7eb52ea79ccdda3d71f8a27829a096a79 -- resources/js/components/Index.js
```

#### Index.js - Imports

```javascript
// Our custom components
import TableExportAndSearch from './TableExportAndSearch';
import TableActions from './TableActions';
import CreateForm from './CreateForm'
import FormModal from './FormModal
```

We've added three new import statements to the top of the file to load the new React assets we'll be developing below.

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

1. We added a **_modalsOpen_** object to the state that will allow us to track which of the create, edit, and delete modal windows are opened
2. We added a new binding for the **_toggleModal_** function which will be called to actually show/hide the create, edit, and delete modal windows


#### Index.js - toggleModal()

Next we've added the **_toggleModal()_** function:

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

Note it also accepts a user object as the second parameter, and we'll utilize this in the future to determine which user we should apply edits to and/or delete.

#### Index.js - render() :: actions constant

{% raw %}
```javascript
// Define a list of actions we want to be able to take on a given user
// These are displayed in the datatables's dummy 'Action' column which is defined below
const actions = [
	{
		title: "Edit User",
		onClick: this.toggleModal,
		modalType: 'edit',
		class: "text-secondary",
		icon: "fa fa-fs fa-pencil-alt",
	},
	{
		title: "Delete User",
		onClick: this.toggleModal,
		modalType: 'delete',
		class: "text-danger",
		icon: "fa fa-fs fa-trash",
	}
];
```
{% endraw %}

Our next set of changes occur in the **_render()_** method.  First we create an **_actions_** constant that we use to define what should happen when a user clicks the **_edit_** or **_delete_** icons we'll place in the data table rows later on for each user.  Further along in the code we'll pass this constant to the data table as part of a **_user actions_** [dummy column](https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/column-props.html#columnisdummyfield-bool).

#### Index.js - render() :: columns constant

Next we update the columns constant to include the new **_user actions_** [dummy column](https://react-bootstrap-table.github.io/react-bootstrap-table2/docs/column-props.html#columnisdummyfield-bool).

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

In the code above we create the dummy column containing possible table row actions by adding a **_formatter_** property to the object that returns a **_TableActions_** React component.  As we'll see below the **_TableActions_** component will draw a set of icons into the table's action column.  Clicking on the icons will present an edit/delete modal to the user for the given table row.  Although we won't implement the edit/delete functionality until the next write up, we can display the icons in the data table now.


#### Index.js - render() :: ToolkitProvider :: props

The first change we made to the **_ToolkitProvider :: props_** was to flesh out the create new user button:

{% raw %}
```javascript
<button type="button" className="btn btn-outline-success"
	onClick={ (e) => this.toggleModal('create', null) }>
```
{% endraw %}

This simply makes a call to the **_toggleModal_** method we wrote earlier.  The click action passes the value **_create_** as the first argument to the method, so our function knows which modal to display to the user.  Since this is a new user we pass **_null_** as the second argument.

The second change is the inclusion of the code to handle displaying the create new user modal:

{% raw %}
```javascript
{/* Create user form modal */}
<div>
	<FormModal
		isOpen={this.state.modalsOpen['create']}
		onRequestClose={ (e) => this.toggleModal('create', null) }
		contentLabel="Create User"
		title="Create User"
		modalAppElement="#users"
		styleOverride={ new Object({width: '40%', left: '35%',}) }
	>
		{/* Define and render the actual create user form  */}
		<CreateForm
			onClose={ (e) => this.toggleModal('create', null) }
			onUpdate={ this.fetchUserData }
		/>
	</FormModal>
</div>
{/* END Create form modal */}
```
{% endraw %}

This utilizes the **_FormModel_** component we'll write later on, and passes the **_CreateForm_** component--which we'll also write later on--as its **_this.props.children_** value.  We also define what should happen when the modal is closed, the title of the modal, as well as passing along some custom CSS style options.

### resources/js/components/CreateForm.js

Now that we have all the pieces in place to handle displaying the create new user form modal let's go ahead and develop it next.  Note; however, due to its size we'll explore it piece-by-piece instead of pasting the whole file in as a single chunk.  As always you can view the source [here](https://github.com/nrasch/AppTemplate/blob/ReactUserManager_PartTwo/resources/js/components/CreateForm.js) for a complete view of the code.

{% raw %}
```javascript
// Standard import items
import React, {Component} from 'react';
import ReactDOM from 'react-dom';

// Formik table related imports
import { Formik, Field, Form, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import LoadingOverlay from 'react-loading-overlay';
import isEmpty from 'lodash/isEmpty'

// Our custom components
import FlashMessage from './FlashMessage';
```
{% endraw %}

We start by importing all the libraries and other assets we'll need for this component.

{% raw %}
```javascript
export default class CreateForm extends Component {

	constructor(props) {
		super(props);

		this.state = {
			// Show/hide Laravel sytle flash messages
			//  regarding actions taken on the page
			showFlashMessage: false,
			// Container for request response
			// data/message/etc sent back by the server
			requestResult: null,
			// Show/hide the form overlay on ajax requests
			//  to notify the user activity is happening
			showOverlay: false,
		}

		//Bindings
		this.hideFlashMessage = this.hideFlashMessage.bind(this);
	}
```
{% endraw %}

Next we declare the **_CreateForm_** class, and build the class constructor.  The constructor defines three properties in the state we'll use later on in the class:

* **showFlashMessage**:  Utilized to show/hide Laravel style flash messages regarding actions taken on the page.  Based loosely off the ideas in the [laracasts/flash](https://github.com/laracasts/flash) repository.
* **requestResult**:  We'll store the request response data/messages/etc sent back by the back end server
* **showOverlay**:  This property will track if we should show/hide the form overlay on ajax requests to notify the user activity is happening

{% raw %}
```javascript
	// Hide the the flash message at the top of the modal
	hideFlashMessage() {
			this.setState({
				showFlashMessage: false
			});
	}
```
{% endraw %}

Next we throw in a quick method to toggle showing the flash message(s).

{% raw %}
```javascript
	// Examine this.props and this.state and
	// return class response (typical React elements)
	render() {

		// Using Laravel validation on the back end,
		// so no need to add any properties
		// to this object
		const ValidationSchema = Yup.object().shape({
		});
```
{% endraw %}

We start the **_render()_** function off by declaring the **_ValidationSchema_** Yup object.  However, we don't populate it with any properties, because we are utilizing Laravel back end data validation.  If we wanted to change our minds later on and include front end validation then we could easily add items to the constant.

{% raw %}
```javascript
// Prepare and return React elements
return (
	<div>

		{/* Display Laravel style flash messages
		 in response to page actions */}
		<FlashMessage
			show={ this.state.showFlashMessage }
			result={ this.state.requestResult }
		/>
		{/* END Display Laravel style flash messages
		 in response to page actions */}

		{/* Form overlay to visually indicate
		activity is occurring to the user */}
		<LoadingOverlay
			active={this.state.showOverlay}
			spinner
			text='Working...'
		>
```
{% endraw %}

The next code block opens up the return statement, and begins to populate the items that will be rendered on the page to the user.  The **_FlashMessage_** component handles displaying Laravel style flash messages based on the response from the back end server, and the **_LoadingOverlay_** allows us to indicate to the user that activity is occurring when the form is submitted.  Both of these components will be described in the next section.

{% raw %}
```javascript
{/* Form block */}
<div onClick={this.hideFlashMessage}>
	{/* Formik form */}
	<Formik
		initialValues={{
			name: '',
			email: '',
			password: '',
			password_confirmation: '',
			roles: [1],
		}}
		validationSchema={ValidationSchema}
```
{% endraw %}

We then add an **_onClick_** action to the form's div that hides any visible flash messages, and then we begin building the create new user form using **_Formik_**.  The first prop we set for the **_Formik_** component is the initial values of the form components.  Since this is a new user the name, email, etc. are all set to empty strings.  We also set the front end validation rules using the empty **_ValidationSchema_** constant we created earlier.

{% raw %}
```javascript
	onSubmit={(values, actions) => {
		// Show the overlay while the ajax request is processing
		this.setState({
			showOverlay: true,
		});

		// Submit the request to the server and handle the response
		axios.post(
			'/create_user',
			values,
			{timeout: 1000 * 10},
		)
		.then(response => {
			if (response.data.result) {
				// Store the data/message/etc
				// sent back by the server in the state
				this.setState({
					requestResult: response.data.result,
				});

				// Reset the form if the user was created successfully
				if (response.data.result.type == 'success') {
					actions.resetForm(this.props.initialValues);
				}
			};
		})
		.catch( error => {
			// Init container for flash error message
			let data = {};

			// Is this a Laravel back end validation error?
			if (typeof error.response !== 'undefined') {
				if (error.response.status == '422') {
					// Render the errors on the page's form's elements
					actions.setErrors(error.response.data.errors);

					// Define flash message to show user
					data = {
						type: 'danger',
						message: <p className="mb-0">
							<i className="far fa-frown ml-1"></i>
							&nbsp;&nbsp;
							Unable to complete request.  
							Please correct the problems below.</p>,
					};
				}
			}

			// Define flash message to show user
			// if one hasn't already been set
			if (_.isEmpty(data)) {
				data = {
					type: 'danger',
					message: <p className="mb-0">
						<i className="far fa-frown ml-1"></i>
						&nbsp;&nbsp;
						Error:  Unable to process your request at this time.  
						Please try again later.</p>,
				};
			}

			// Pass the flash message data to the
			// flash message display component
			this.setState({
				requestResult: data,
			});
		})
		.then( () => {

			// Hide the ajax processing overlay
			this.setState({
				showOverlay: false,
			});

			// Tell the form we are done submitting
			actions.setSubmitting(false);

			// Show the flash message with
			// the results of the page action
			this.setState((state, props) => ({
				showFlashMessage: true,
			}));

		});
	}}
>
```
{% endraw %}

The next code block--while long--is standard ajax request/response type code.  We start by showing the overlay letting the user know we are taking action behind the scenes.  Then we begin the **_axios_** POST request by defining the endpoint URL and collecting the form values to submit.  

The first **_then_** block handles a success condition by storing the server response in the state's **_requestResult_** property, and clearing the form of previous values for reuse.

A **_catch_** block follows, and handles one of two cases:  1) The back end server has returned validation errors, or 2) some sort of general error has occurred such as lack of network connectivity.  In either case the cause of the error is saved in the state's **_requestResult_** property which can then be displayed to the user as a flash message.  In the case of form validation errors we'll also display notification for each form element that failed validation.

The final **_then_** block performs the post request actions by hiding the overlay, setting the form's submit state to false, and displaying the flash message(s) to the user with the results of the form submit request (i.e. success or error).

{% raw %}
```javascript
{({ errors, dirty, status, touched, isSubmitting, setFieldValue }) => (

<Form className="mb-5">

	{/* Form data fields (name, email, password, etc.) */}
	<div className="col-lg-8 col-xl-5">
		<div className="form-group">
			<label htmlFor="name">
				Name<span className="text-danger">*</span>
			</label>
			<Field
				name="name"
				type="text"
				className={ "form-control " + (errors.name && touched.name ?
					 'is-invalid' : null) }
				placeholder="User's Name..."
				/>
			<ErrorMessage
				name="name"
				component="div"
				className="invalid-feedback font-weight-bold"
			/>
		</div>

		<div className="form-group">
			<label htmlFor="email">
				Email<span className="text-danger">*</span>
			</label>
			<Field
				name="email"
				type="email"
				className={ "form-control " +
					(errors.email && touched.email ? 'is-invalid' : null) }
				placeholder="User's Email..."
				/>
			<ErrorMessage name="email" component="div"
				className="invalid-feedback font-weight-bold" />
		</div>

		<div className="form-group">
			<label htmlFor="password">
				Password<span className="text-danger">*</span>
			</label>
			<Field
				name="password"
				type="password"
				className={ "form-control " + (errors.password && touched.password
					? 'is-invalid' : null) }
				placeholder="User's Password..."
				/>
			<ErrorMessage name="password" component="div"
				className="invalid-feedback font-weight-bold" />
		</div>

		<div className="form-group">
			<label htmlFor="password_confirmation">
				Confirm Password<span className="text-danger">*</span>
			</label>
			<Field
				name="password_confirmation"
				type="password"
				className={ "form-control " + (errors.password_confirmation &&
					touched.password_confirmation ? 'is-invalid' : null) }
				placeholder="Confirm Password..."
				/>
			<ErrorMessage name="password_confirmation" component="div"
				className="invalid-feedback font-weight-bold" />
		</div>

		<div className="form-group">
			<label htmlFor="roles">
				User Roles<span className="text-danger">*</span>
			</label>
			<Field
				name="roles"
				component="select"
				className={ "form-control " + (errors.roles && touched.roles
					? 'is-invalid' : null) }
				onChange={ (e) =>
					setFieldValue(
						"roles",
						[].slice
							.call(e.target.selectedOptions)
							.map(option => option.value),
						false
					)
				}
				multiple={true}
			>
			{ [ [1, 'Administrator'], [2, 'User'] ].map( (item, index) => {
				return (<option key={ index } value={ item[0] }>{ item[1] }</option>);
			})}
			</Field>
			<ErrorMessage name="roles" component="div"
				className="invalid-feedback font-weight-bold" />
		</div>

	</div>
	{/* END Form data fields (name, email, password, etc.) */}
```
{% endraw %}

And now we are into the meat of the form.  The code block above creates the actual HTML form elements such as name, email, and password fields utilizing **_Formik_** **_Field_** components.  Also note that each **_Field_** component is followed by a **_ErrorMessage_** component.  

Example for the Name field:

```javascript
<ErrorMessage
	name="name"
	component="div"
	className="invalid-feedback font-weight-bold"
/>
```
This component is what is responsible for drawing a red box around each form element that failed validation along with the validation error message for that field.  

But how does **_Formik_** know the form field has a validation error and what validation error text to display?  Answer:  The code we wrote in the **_onSubmit_** **_Formik_** prop injects the error(s) received from the Laravel back end into the **_Formik_** component using the **_Formik_** **_action_** method.  Here is a review of the code we wrote that does this, and notice how we use the **_Formik_** **_action_** method to assign the errors received from the POST request:

{% raw %}
```javascript
// Is this a Laravel back end validation error?
if (typeof error.response !== 'undefined') {
	if (error.response.status == '422') {
		// Render the errors on the page's form's elements
		actions.setErrors(error.response.data.errors);
```
{% endraw %}

Once this is complete **_Formik_** will display the errors on the form's field components.  You can read more about **_Formik_** validation [here](https://jaredpalmer.com/formik/docs/guides/validation).

{% raw %}
```javascript
{/* Form submit/close buttons */}
<div className="form-group ml-3 mt-4">
	<button
		type="submit"
		className="btn btn-outline-success"
		disabled={isSubmitting || !isEmpty(errors)}
		>
		<i className="fa fa-fw fa-plus mr-1"></i> Create User
	</button>

		<button
			type="button"
			className="btn btn-outline-secondary ml-3"
			onClick={this.props.onClose}
		>
			<i className="fa fa-times-circle mr-1"></i> Close
		</button>

</div>
{/* END Form submit/close buttons */}

</Form>
```
{% endraw %}

We then complete the form with the **_Submit_** and **_Cancel_** buttons.  Note that we disabled the **_Submit_** button when the form is actively submitting or validation errors are present.  

{% raw %}
```javascript
						)}
						{/* END
						{({ errors, dirty, status, touched, isSubmitting }) => ( */}
					</Formik>
					{/* END Formik form */}

				</div>
				{/* END Form block */}

				</LoadingOverlay>
				{/* END
				Form overlay to visually indicate activity is
			 	occurring to the user */}

			</div>
		);
	}
}
```
{% endraw %}

The rest of the code simply closes the **_div_** and React component blocks.

### resources/js/components/FlashMessage.js

With the create new user form wrapped up we can turn our attention to the React component that displays the flash messages at the top of the modal.

{% raw %}
```javascript
import React from 'react';

export default class FashMessage extends React.Component {
	constructor(props) {
		super(props)
	}

	render() {
		// Render nothing if the flash message shouldn't be shown
		if (!this.props.show) {
			return null;
		}

		// Render the flash message
		return (
			<div
				className={ "alert alert-" + this.props.result.type }
				role="alert"
			>
				{ this.props.result.message }
			</div>
		);
	}
}
```
{% endraw %}

This component utilizes the **_show_** prop to determine if it should render or not, and if so it returns a bootstrap styled div containing the flash message (see the screenshots at the bottom of this post for examples).

### resources/js/components/FormModal.js

{% raw %}
```javascript
import React from 'react';
import ReactDOM from 'react-dom';
import Modal from 'react-modal';

// Modal settings
Modal.defaultStyles.overlay.backgroundColor = 'rgba(0,0,0,0.4)';
const modalContentStyle = {
	content : {
		top               : '25%',
		left              : '27%',
		right             : 'auto',
		bottom            : 'auto',
		marginRight       : '-50%',
		width             : '60%',
		height            : '80%',
		marginTop         : '-10%',
		marginLeft        : '-50px',
		backgroundColor   : '#fefefe',
		overflow          : 'auto',
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
				<div className="mb-0">
					<div className="bg-primary-dark">
						<h3>{ this.props.title }</h3>
					</div>
					<div className="font-size-sm">
						{this.props.children}
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
	content : {
		top               : '25%',
		left              : '27%',
		right             : 'auto',
		bottom            : 'auto',
		marginRight       : '-50%',
		width             : '60%',
		height            : '80%',
		marginTop         : '-10%',
		marginLeft        : '-50px',
		backgroundColor   : '#fefefe',
		overflow          : 'auto',
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
				<div className="mb-0">
					<div className="bg-primary-dark">
						<h3>{ this.props.title }</h3>
					</div>
					<div className="font-size-sm">
						{this.props.children}
					</div>
				</div>
			</Modal>
		);
	}
}
```
{% endraw %}


### resources/js/components/TableActions.js

{% raw %}
```javascript
import React from 'react';

export default class TableActions extends React.Component {

	// Create and return React elements for each action (i.e. create, delete, etc.)
	render() {
		return (
			<div>
				{ this.props.actions.map( (action, index) => {
					return(
						<span
							title={ action.title }
							key={ index }
							onClick={
								(e) => action.onClick(
									action.modalType, this.props.item
								)}
							className={ "mr-3 " + action.class }
						>
							<i className={ action.icon }></i>
						</span>
					)
				})}
			</div>
		);
	}
}
```
{% endraw %}

The code in this component loops through each element we want to appear in the user data table's **_actions_** column.  It then renders an icon representing the action (i.e. a trashcan for deletion), as well as what should happen when the action's icon is clicked.  

So for example the **_edit_** action will be represented by a pencil icon, and clicking it will open the **_edit user_** modal window.  Also note we pass a second argument to the **_onClick_** action: **_this.props.item_**.  

This is a reference to the row being clicked on, and we'll use this in the next phase to populate the edit user form as well as identify which user we want to remove for deletion requests.

## The end result

To wrap up we first want to compile the code we wrote above using the following terminal command:

```bash
npm run dev
```

Once that's finished browse to your Laravel application, click the **_Users_** link in the navigation bar, and you should see the following:

![User manager part two final product](assets/images/posts/2019/user-index-part-two-final-product.png)

Next let's view what the create new user modal looks like:

![User manager create new user modal](assets/images/posts/2019/user-manager-create-user-modal.png)

And the flash messaging with form validation errors:

![User manager create new user modal validation errors](assets/images/posts/2019/user-manager-create-user-modal-validation-errors.png)

And finally a success response:

![User manager create new user modal success](assets/images/posts/2019/user-manager-create-user-modal-success.png)


## Summary

This post has covered the second step in creating a User management SPA utilizing Laravel and React.  In the next part of the series we'll explore adding the capability to edit existing users.

You can find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUseManager_PartTwo).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
