---
layout: post
title:  "Creating a User Management UI with Laravel and React - Part Four"
tags: [ Laravel, PHP, Web Development, React ]
featured_image_thumbnail: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_thumbnail.png
featured_image: assets/images/posts/2019/creating-a-user-manager-with-laravel-and-react_title.png
featured: true
hidden: true
---

In this fourth post of the series we add the ability to delete existing users from our [CRUD](https://en.wikipedia.org/wiki/Create,_read,_update_and_delete) based user management single page application (SPA) utilizing [Laravel](https://laravel.com/) and [React](https://reactjs.org/).

<!--more-->

## Series posts

Posts in this series:

* [Part one]({% post_url 2019-08-22-creating-a-user-management-ui-with-laravel-and-react-part-one %}) - Initial setup with user accounts data table
* [Part two]({% post_url 2019-08-27-creating-a-user-management-ui-with-laravel-and-react-part-two %}) - Adding the ability to create new users
* [Part three]({% post_url 2019-08-29-creating-a-user-management-ui-with-laravel-and-react-part-three %}) - Adding the ability to edit existing users
* [Part four]({% post_url 2019-09-04-creating-a-user-management-ui-with-laravel-and-react-part-four %}) - Adding the ability to delete existing users

With more to follow...

## Prerequisites and assumptions

For this discussion I assume you've followed the instructions in the [third post]({% post_url 2019-08-29-creating-a-user-management-ui-with-laravel-and-react-part-three %}), and we continue by building on what we did previously.

You can also find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUserManager_PartFour).

## Create Laravel assets

To start we'll create the Laravel assets needed to support the React components.

### Routing

First let's add the new route for the React single page application (SPA) dealing with user deletion.  At the same time let's also wrap the ability to create, edit, and delete users in middleware ensuring that only the **_administrator_** role is able to perform these actions.

We do this by implementing **_RoleMiddleware_** from the **_spatie/laravel-permission_** Laravel package around the routes we wish to protect.  You can read more about it [here](https://docs.spatie.be/laravel-permission/v3/basic-usage/middleware/).

Edit the **_routes/web.php_** file and add the following code:

```php
<?php

// Default route
Route::get('/', function () {
    return redirect( route('home'));
});

Auth::routes();

Route::get('/home', 'HomeController@index')->name('home');

Route::get('/get_users', 'UserController@getUsers');

// Ensure only administrators are able to add/edit/delete users
Route::group(['middleware' => ['auth', 'role:administrator']], function () {
  Route::post('/create_user', 'UserController@store');
  Route::put('/edit_user/{id}', 'UserController@update');
  Route::delete('/delete_user/{id}', 'UserController@destroy');
});

Route::get('/users', 'UserController@index')->name('users');
```

We added a new route, **_/delete_user/{id}_**, which accepts a DELETE request containing the ID of the user account we wish to remove.  This route will handle the actions that need to take place when the delete user form is submitted via React to the Laravel back end.

We chose to add this new route as a DELETE request, because we'd like to follow Laravel's resource controller action standards ([reference](https://laravel.com/docs/5.8/controllers#resource-controllers)).

We've also wrapped the create, edit, and delete routes in middleware only allowing users with the role **_administrator_** to perform these actions.

### app/Http/Kernel.php

Because we've wrapped the create, edit, and delete routes in **_spatie/laravel-permission_** route middleware we need to modify the **_app/Http/Kernel.php_** file as described in the [documentation](https://docs.spatie.be/laravel-permission/v3/basic-usage/middleware/).

Edit the **_app/Http/Kernel.php_** file and add the make the following change:

```php
...

protected $routeMiddleware = [
    ...
    'role' => \Spatie\Permission\Middlewares\RoleMiddleware::class,
];
```

This will enable us to utilize the **_role:administrator_** directive in the **_routes/web.php_** file we modified above.


### app/Http/Controllers/UserController.php

Let's update the user controller next.  Edit the **_app/Http/Controllers/UserController.php_** file and replace the **_destroy()_** method with the following code:

{% raw %}
```php
/**
* Remove the specified resource from storage.
*
* @param  integer  $id
* @return \Illuminate\Http\Response
*/
public function destroy($id)
{
  // Pull the user record from the database
  $user = User::findOrFail($id);

  // Remove the user account from the DB
  $user->delete();

  // Create response to be returned to the view
  $response['result']['type'] = 'success';
  $response['result']['message'] = 'The user was successfully deleted!';
  $response['data'] = $user->__toString();

  // Return JSON response
  return response()->json($response);

}
```
{% endraw %}

This method first attempts to pull the user record being modified from the database with a **_findOrFail()_** method call.  Assuming that completes successfully we then proceed to delete the user and remove them from the database.  Next, we return a JSON result to the view to confirm the selected user account was indeed removed.

That should be the end of our Laravel tasks, and from here on out the rest of our work will be creating the React javascript assets.

---

## Develop React assets

### resources/js/components/Index.js

Since we did most of the heavy lifting in a [previous post]({% post_url 2019-08-27-creating-a-user-management-ui-with-laravel-and-react-part-two %}) the changes to the **_resources/js/components/Index.js_** file are relatively minor:

{% raw %}
```javascript
// ==> ADD THE DELETE USER MODAL CODE BLOCK @ LINE 299

{/* Delete user form modal */}
<div>
	<FormModal
		isOpen={ this.state.modalsOpen['delete'] }
		onRequestClose={ (e) => this.toggleModal('delete', this.state.user) }
		contentLabel="Delete User Confirmation"
		title="Delete User Confirmation"
		modalAppElement="#users"
		styleOverride={ new Object({width: '40%', left: '35%', height: '45%'}) }
	>
		{/* Define and render the actual delete user form  */}
		<DeleteForm
			onClose={ (e) => this.toggleModal('delete', this.state.user) }
			onUpdate={ this.fetchUserData }
			user={ this.state.user }
		/>
	</FormModal>
</div>
{/* END Delete user form modal */}
```
{% endraw %}

This code is almost exactly the same as the **_Create user form modal_** code block above it in the source.  

The only differences are the 1) update to the label and title, and 2) we are now passing **_this.state.user_** as the second argument to the **_this.toggleModal_** method.  As we'll see in a moment this allows the delete user form to properly identify which user account we want to remove, and then pass this information to the Laravel back end.

### resources/js/components/DeleteForm.js

Lastly we have the new **_resources/js/components/DeleteForm.js_** file.  Due to the size of this file we'll explore it piece-by-piece instead of pasting the whole file in as a single chunk.  As always you can view the source [here](https://github.com/nrasch/AppTemplate/blob/ReactUserManager_PartFour/resources/js/components/DeleteForm.js) for a complete view of the code.

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
export default class DeleteForm extends Component {

	constructor(props) {
		super(props);

		this.state = {
			// Show/hide Laravel style flash messages regarding actions
			// taken on the page
			showFlashMessage: false,
			// Container for request response data/message/etc sent back
			// by the server
			requestResult: null,
			// Show/hide the form overlay on ajax requests to notify
			//  the user activity is happening
			showOverlay: false,
			// Show/hide the delete confirmation form, 'cause it looks
			// odd to still have it once the item is deleted
			hideForm: false,
		}

		//Bindings
		this.hideFlashMessage = this.hideFlashMessage.bind(this);
	}
```
{% endraw %}

Next we declare the **_DeleteForm_** class, and build the class constructor.  The constructor defines four properties in the state we'll use later on in the class:

* **showFlashMessage**:  Utilized to show/hide Laravel style flash messages regarding actions taken on the page.  Based loosely off the ideas in the [laracasts/flash](https://github.com/laracasts/flash) repository.
* **requestResult**:  We'll store the request response data/messages/etc sent back by the back end server
* **showOverlay**:  This property will track if we should show/hide the form overlay on ajax requests to notify the user activity is happening
* **hideForm**:  This property will track if we should hide the delete user account confirmation text and submit button, which we'll want to do once the select user account has been deleted

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
			id: this.props.user.id,
		}}
		validationSchema={ValidationSchema}
```
{% endraw %}

We then add an **_onClick_** action to the form's div that hides any visible flash messages, and then we begin building the delete user form using **_Formik_**.  The first prop we set for the **_Formik_** component is the initial values of the form components.  Since this is an existing user we set the **_id_** value to the ID of the user account to be deleted.  We also set the front end validation rules using the empty **_ValidationSchema_** constant we created earlier.

{% raw %}
```javascript
	onSubmit={(values, actions) => {
		// Show the overlay while the ajax request is processing
		this.setState({
			showOverlay: true,
		});

		// Submit the request to the server and handle the response
		axios.post(
			'/delete_user/' + this.props.user.id,
			values,
			{timeout: 1000 * 10},
		)
		.then(response => {
			if (response.data.result) {
				// Store the data/message/etc
				// sent back by the server in the state
				this.setState({
					requestResult: response.data.result,
          hideForm: true,
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

The next code block--while long--is standard ajax request/response type code.  We start by showing the overlay letting the user know we are taking action behind the scenes.  Then we begin the **_axios_** DELETE request by defining the endpoint URL and collecting the form values to submit.  

The first **_then_** block handles a success condition by storing the server response in the state's **_requestResult_** property.  It also hides the delete user form's confirmation text and submit button, since it doesn't make sense to confirm and submit the form twice.

A **_catch_** block follows, and handles one of two cases:  1) The back end server has returned validation errors, or 2) some sort of general error has occurred such as lack of network connectivity.  In either case the cause of the error is saved in the state's **_requestResult_** property which can then be displayed to the user as a flash message.

The final **_then_** block performs the post request actions by hiding the overlay, setting the form's submit state to false, and displaying the flash message(s) to the user with the results of the form submit request (i.e. success or error).

{% raw %}
```javascript
{({ errors, dirty, status, touched, isSubmitting, setFieldValue }) => (

  <Form className="mb-5" hidden={ this.state.hideForm }>

    {/* Form data fields */}
    <div className="col-12">
      <div className="form-group text-center">
        <div>
          <i className="text-warning fa fa-4x fa-question-circle mb-2" />
        </div>
        <div className="h3">
          <strong>Are you sure?</strong>
        </div>
        <div className="h5">
          You will be unable to recover this user's account!
        </div>
      </div>

      <Field name="id" type="hidden" />
    </div>
    {/* END Form data fields */}

    {/* Form submit/close buttons */}
    <div className="form-group ml-3 mt-4 text-center">
      <button
        type="submit"
        className="btn btn-outline-success"
        disabled={isSubmitting || !isEmpty(errors)}
        >
        <i className="fa fa-fw fa-plus mr-1"></i> Delete User
      </button>

        <button type="button" className="btn btn-outline-secondary ml-3" onClick={this.props.onClose}>
          <i className="fa fa-times-circle mr-1"></i> Close
        </button>

    </div>
    {/* END Form submit/close buttons */}
```
{% endraw %}

And now we are into the meat of the form.  The code block above creates the form confirmation text elements, the submit button, and a hidden field with the user ID to be deleted.

{% raw %}
```javascript
              </Form>

            )}
            {/* END {({ errors, dirty, status, touched, isSubmitting }) => ( */}
          </Formik>
          {/* END Formik form */}

        </div>
        {/* END Form block */}

        </LoadingOverlay>
        {/* END Form overlay to visually indicate activity is occurring to the user */}

      </div>
    );
  }
}
```
{% endraw %}

The rest of the code simply closes the **_div_** and React component blocks.

## The end result

To wrap up we first want to compile the code we wrote above using the following terminal command:

```bash
npm run dev
```

Once that's finished browse to your Laravel application, click the **_Users_** link in the navigation bar, and then click the trashcan icon in one of the user rows.  You should then see the following:

![User manager part four delete user modal](assets/images/posts/2019/user-manager-delete-user-modal.png)

## Summary

This post has covered the fourth step in creating a User management SPA utilizing Laravel and React.  In the next part of the series we'll explore refactoring, [DRYing](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) up our code, along with some polishing.

You can find the source code for this post [here](https://github.com/nrasch/AppTemplate/tree/ReactUseManager_PartFour).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
