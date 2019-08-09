---
layout: post
title:  "Creating a Base Laravel Project Template Part Three"
tags: [ Laravel, PHP, Web Development ]
featured_image_thumbnail: assets/images/posts/2019/creating_base_laravel_project_template_title_thumbnail.png
featured_image: assets/images/posts/2019/creating_base_laravel_project_template_title.png
featured: false
hidden: false
---

In this post we continue work on our Application Template by formalizing the login/logout actions as well as adding roles and permissions with [Spatie](https://github.com/spatie/laravel-permission).

<!--more-->

### Source
You can find the complete source code for this post [on GitHub](https://github.com/nrasch/AppTemplate/tree/PartThree).

### Previous
Previous posts in this series:
* [Creating a Base Laravel Project Template Part One]({% post_url 2019-07-26-creating-a-base-laravel-project-template %})
  * Build the initial template and enable React support
* [Creating a Base Laravel Project Template Part Two]({% post_url 2019-07-30-creating-a-base-laravel-project-template-part-two %})
  * Apply a minimal template to the views

## Existing assets

Before we begin making modifications to the login and logout process let's examine what Laravel provides out-of-the-box:

### Login/Logout routing

If we open a terminal, change to the directory of our project, and type in **_php artisan route:list_** we can observe that Laravel has added the following Auth routes for us:

```
|        | GET|HEAD | login                  | login            | App\Http\Controllers\Auth\LoginController@showLoginForm                | web,guest    |
|        | POST     | login                  |                  | App\Http\Controllers\Auth\LoginController@login                        | web,guest    |
|        | POST     | logout                 | logout           | App\Http\Controllers\Auth\LoginController@logout                       | web          |
```

These routes--along with some others--are added to the application via the call to **_Auth::routes();_** in the **_routes/web.php_** file which should be on or around line 18 if you haven't modified it.

If you want to customize, add, or remove any of these you can remove the reference to **_Auth::routes();_** and add entries to the **_routes/web.php_** file manually.  So for example if you are working on a private non-public web application you may wish to remove references to registration.

We do; however, want to make one change, and that is to make the default route point to the **_home_** route.  Edit the **_routes/web.php_** file, and modify the default route to be the following:

```php
// Default route
Route::get('/', function () {
    return redirect( route('home'));
});
```

### LoginController

From the route table we know that login and logout actions are handled by the **_app/Http/Controllers/Auth/LoginController.php_** file.  If we examine this file it is rather sparse:

```php
namespace App\Http\Controllers\Auth;

use App\Http\Controllers\Controller;
use Illuminate\Foundation\Auth\AuthenticatesUsers;

class LoginController extends Controller
{
    use AuthenticatesUsers;

    /**
     * Where to redirect users after login.
     *
     * @var string
     */
    protected $redirectTo = '/home';

    /**
     * Create a new controller instance.
     *
     * @return void
     */
    public function __construct()
    {
        $this->middleware('guest')->except('logout');
    }
}
```

As you've probably already guessed all the magic happens via the **_AuthenticatesUsers_** trait.  You can examine the source code for it in your project directory here:  **_vendor/laravel/framework/src/Illuminate/Foundation/Auth/AuthenticatesUsers.php_**

It's fairly easy to read, and parsing it should give you a good idea of how the authentication process for logins works.  For example, here is what happens when some one logs out:

```php
public function logout(Request $request)
{
    $this->guard()->logout();

    $request->session()->invalidate();

    return $this->loggedOut($request) ?: redirect('/');
}
```

We won't be altering how this works in this post.  However, if in the future you need something custom for your application, you now know where to find the source files that control the login and logout functionality.  And from there you can  you can override one or more of the trait's methods, etc.

### Login/Logout views

The **_AuthenticatesUsers->showLoginForm()_** trait method contains a single line of code,**_return view('auth.login');_**.  This renders the **_resources/views/auth/login.blade.php_** blade file which extends **_resources/views/layouts/app.blade.php_**.

We won't be discussing the contents these files further, because we'll be overwriting them presently.  :)


## Customize the default login/logout functionality

Now that we know where everything is and how it works by default we can begin customizing things.  For now we don't need to modify the routes.  We examined them earlier, and the URLs that Laravel creates by default work for our purposes.

What we do need to do; however, is modify the views.

Our strategy will be to have one layout and view for the login area sans any navigation bars or menus, and then another set of layouts and views for authenticated users to interact with the application.

### Prerequisites

Before we write any code we need to install the [laravelcollective/html](https://github.com/LaravelCollective/html) package, which will help us when developing forms.  

**Please note that the documentation on using the package can be found [here](https://github.com/LaravelCollective/docs/blob/5.6/html.md).**  

Run the following command from the terminal to install the package:

```bash
composer require "laravelcollective/html":"^5.8.0"
```

Next, add the HTML provider to the **_providers_** array in the **_config/app.php_** file:

```php
  'providers' => [
    // ...
    Collective\Html\HtmlServiceProvider::class,
    // ...
  ],
```

Finally, add the **_Form_** and **_Html_** class aliases to the **_Class Aliases_** array of the **_config/app.php_** file:

```php
  'aliases' => [
    // ...
      'Form' => Collective\Html\FormFacade::class,
      'Html' => Collective\Html\HtmlFacade::class,
    // ...
  ],
```

Terminate any running **_artisan serve_** processes, and execute the following commands in the terminal:
```bash
$ npm run dev
$ composer dump-autoload && php artisan cache:clear && php artisan serve
```

The **_laravelcollective/html_** package should now be installed and configured to run in the application template.

### Create the login/logout views and functionality

Next we need to create the login view.  

#### Login/logout base layout

Create a new file called **_resources/views/layouts/auth.blade.php_** and add the following code to it:

```html
<!doctype html>
<html lang="{{ config('app.locale') }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, shrink-to-fit=no">

        <title>Application Template</title>

        <!-- Fonts -->
        <link rel="dns-prefetch" href="//fonts.gstatic.com">
        <link href="https://fonts.googleapis.com/css?family=Nunito" rel="stylesheet">

        <!-- Styles -->
        @yield('css_before')
          <link href="{{ asset('css/app.css') }}" rel="stylesheet">
        @yield('css_after')

        <!-- Scripts -->
        <script>window.Laravel = {!! json_encode(['csrfToken' => csrf_token(),]) !!};</script>
    </head>
    <body>
        <div id="page-container">
            <!-- Main Container -->
            <main id="main-container">
                @yield('content')
            </main>
            <!-- END Main Container -->
        </div>
        <!-- END Page Container -->

        @yield('js_after')
    </body>
</html>
```

This will serve as the layout for any login or logout forms we need to develop.

#### Login form

To create the login form edit the file **_resources/views/auth/login.blade.php_** and replace its contents with the following code:

```html
@extends('layouts.auth')

@section('content')
<!-- Page Content -->
<div class="bg-image"
<div class="hero-static bg-white-95">
  <div class="content">
    <div class="row justify-content-center">
      <div class="col-md-8 col-lg-6 col-xl-4">
        <!-- Sign In Block -->
        <div class="block block-themed block-fx-shadow mb-0">
          <div class="block-content">
            <div class="p-sm-3 px-lg-4 py-lg-5">
              <h1 class="mb-2">Application Template</h1>
              <p>Welcome, please login.</p>

              <!-- Sign In Form -->
              <form action="{{ route('login') }}" method="POST">

                @csrf

                <div class="py-3">
                  <div class="form-group">
                    <input type="text" class="form-control form-control-alt form-control-lg {{ $errors->has('email') ? ' is-invalid' : '' }}"
                    id="email" name="email" placeholder="E-mail Address" value="{{ old('email') }}" autofocus>

                    @if ($errors->has('email'))
                    <span class="text-danger" role="alert">
                      <strong>{{ $errors->first('email') }}</strong>
                    </span>
                    @endif
                  </div>

                  <div class="form-group">
                    <input type="password" class="form-control form-control-alt form-control-lg  {{ $errors->has('password') ? ' is-invalid' : '' }}"
                    id="password" name="password" placeholder="Password">

                    @if ($errors->has('password'))
                    <span class="text-danger" role="alert">
                      <strong>{{ $errors->first('password') }}</strong>
                    </span>
                    @endif
                  </div>

                </div>
                <div class="form-group row">
                  <div class="col-md-6 col-xl-5">
                    <button type="submit" class="btn btn-block btn-primary">
                      <i class="fa fa-fw fa-sign-in-alt mr-1"></i> Sign In
                    </button>
                  </div>
                </div>
              </form>
              <!-- END Sign In Form -->
            </div>
          </div>
        </div>
        <!-- END Sign In Block -->
      </div>
    </div>
  </div>
</div>
</div>
<!-- END Page Content -->
@endsection
```

This creates the form the user will fill out and submit to authenticate against the application.  It includes the CSRF token as well any feedback on validation errors that might have occurred.

#### Enable logout

And finally we need to enable the user to logout.  We will add an element to the top navigation bar that uses jQuery to submit a form that matches the **_logout_** route signature.  

First we need to add the jQuery that will submit the logout form.  Start by editing the **_resources/views/layouts/app.blade.php_** file, and modify the javascript at the end of the page like so:

```html
<!-- Scripts -->
<!-- Move this here and remove 'defered', or you'll have a jQuery not defined error!
See https://stackoverflow.com/questions/51595843/laravel-5-webpack-jquery-is-not-defined -->
<script src="{{ mix('js/app.js') }}"></script>

<script type="text/javascript">
  $("#menu-toggle").click(function(e) {
    e.preventDefault();
    $("#wrapper").toggleClass("toggled");
  });

  // Log the user out of the application
  $('#logoutFormLink').click( function(e) {
    e.preventDefault();
    $('#logoutForm').submit();
  });
</script>

@yield('js_after')
```

Now when the user clicks the logout link in the navigation bar the hidden logout form will be submitted.

Second we need to add the actual logout element and form to the top navigation area.  Edit the **_resources/views/components/top_nav.blade.php_** file, and add the logout element to the dropdown menu:

{% raw %}
```html
<div class="dropdown-menu dropdown-menu-right" aria-labelledby="navbarDropdown">
  <a class="dropdown-item" href="#">Action</a>
  <a class="dropdown-item" href="#">Another action</a>
  <div class="dropdown-divider"></div>
  <a class="dropdown-item" href="#">Something else here</a>

  <!-- Add the logout link and hidden form -->
	<a id="logoutFormLink" href="#" class="text-danger dropdown-item">
		<i class="fas fa-sign-out-alt"></i>
    Logout
	</a>
	{{ Form::open([
      'route' => 'logout',
      'method' => 'POST',
      'display' => 'none',
      'id' => 'logoutForm',
       ]) }}
  {{ Form::close() }}
</div>
```
{% endraw %}

This uses the **_laravelcollective/html_** package we installed earlier to create a hidden form that submits a logout request to the **_LoginController_** controller's **_logout_** method.

#### CSRF protection on the logout form?

From the [laravelcollective/html documentation](https://github.com/LaravelCollective/docs/blob/5.6/html.md#csrf-protection):

_Adding The CSRF Token To A Form_

_Laravel provides an easy method of protecting your application from cross-site request forgeries. First, a random token is placed in your user's session. If you use the Form::open method with POST, PUT or DELETE the CSRF token will be added to your forms as a hidden field automatically. Alternatively, if you wish to generate the HTML for the hidden CSRF field, you may use the token method:_

```php
echo Form::token();
```
So this was taken care of for us via the package.  :)

### Testing it all out

Let's test everything out.  Start by terminating any running **_artisan serve_** processes, and execute the following commands in the terminal:

```bash
$ npm run dev
$ composer dump-autoload && php artisan cache:clear && php artisan serve
```

Now browse to **_http://localhost:8080/login_**, and you should see this:

![Login screen](assets/images/posts/2019/login_screen.png)

Let's go ahead and submit an empty form to view the validation in action:

![Login screen validation](assets/images/posts/2019/login_screen_validation.png)

And once we log in we can inspect the logout link we created:

![Logout menu item](assets/images/posts/2019/logout_menu_item.png)

Clicking the link should log you out and redirect you to the login screen.

## Roles and permissions

To implement roles and permissions in our Application Template we are going to utilize [Spatie](https://github.com/spatie/laravel-permission).

### Installing

To install [Spatie](https://github.com/spatie/laravel-permission) we are going to follow the instructions found [here](https://docs.spatie.be/laravel-permission/v2/installation-laravel/).

Start by opening a terminal and executing the following command:

 ```bash
 $ composer require spatie/laravel-permission
 ```

Once the install completes edit the **_config/app.php_** file and register the service provider:

```php
'providers' => [
    // ...
    Spatie\Permission\PermissionServiceProvider::class,
];
```

Next publish the migration and the config:

```bash
$ php artisan vendor:publish --provider="Spatie\Permission\PermissionServiceProvider" --tag="migrations"
$ php artisan vendor:publish --provider="Spatie\Permission\PermissionServiceProvider" --tag="config"
```

### Create roles and permissions

To start we are going to create one permission and two roles.  We'll start with the permission, so we can assign it to one of the roles later on.  Create the permission and role seed files like so:

```bash
$ php artisan make:seed PermissionSeeder
$ php artisan make:seed RoleSeeder
```

#### database/seeds/PermissionSeeder.php

Next edit the **_database/seeds/PermissionSeeder.php_** file, and replace its contents with the following code:

```php
<?php

use Illuminate\Database\Seeder;
use Spatie\Permission\Models\Permission;

class PermissionSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
      // Clear out any cached configurations for the application, so that
      // Laravel uses the current values for the configuration
      Artisan::call('cache:clear');

      // From the Spatie docs:  If you manipulate permission/role data
      // directly in the database instead of calling the supplied methods,
      // then you will not see the changes reflected in the application
      //  unless you manually reset the cache.
      //
      // We are using the supplied methods, but we are going to clear it anyhow just to be on the safe side.
      app()[\Spatie\Permission\PermissionRegistrar::class]->forgetCachedPermissions();

      // Create the permission(s)
      Permission::create(['name' => 'manage_users']);
    }
}
```  

#### database/seeds/RoleSeeder.php

Now edit the **_database/seeds/RoleSeeder.php_** file, and replace its contents with the following code:

```php
<?php

use Illuminate\Database\Seeder;
use Spatie\Permission\Models\Role;
use Spatie\Permission\Models\Permission;

class RoleSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        $role = Role::create(['name' => 'administrator']);
        $role->givePermissionTo(Permission::all());
        $role->save();

        $role = Role::create(['name' => 'user']);
    }
}
```

This creates two roles, administrator and user, and then assigns the **_manage_users_** permission to the **_administator_** role.

#### database/seeds/DatabaseSeeder.php

We also need to ensure the new seeders we've created run, so let's add them to the **_database/seeds/DatabaseSeeder.php_** file:

```php
<?php

use Illuminate\Database\Seeder;

class DatabaseSeeder extends Seeder
{
    /**
     * Seed the application's database.
     *
     * @return void
     */
    public function run()
    {
        // $this->call(UsersTableSeeder::class);
        $this->call(PermissionSeeder::class);
        $this->call(RoleSeeder::class);
        $this->call(UserSeeder::class);
    }
}
```

Notice we place the permission and role seeders *before* the user seeder, because we want the permissions and roles to exist before we try to assign them.

#### database/seeds/UserSeeder.php

Our second to last code edit is to modify the **_database/seeds/UserSeeder.php_** file to actually assign the new permissions and roles to the users we created:

```php
<?php

use Illuminate\Database\Seeder;
use App\User;

class UserSeeder extends Seeder
{
  /**
  * Run the database seeds.
  *
  * @return void
  */
  public function run()
  {
    // Create an admin user
    $user = User::create([
      'name' => 'Admin',
      'email' => 'admin@admin.com',
      'password' => bcrypt('password')
    ]);
    // Assign the administrator role
    $user->assignRole(['administrator', 'user']);
    $user->save();

    // Create dev/test data for non-production environments
    if (env('APP_ENV') != 'production') {
      // Create N mumber of users
      factory(User::class, 20)->make()->each(function($user) {
        // Assign the user role
        $user->assignRole('user');
        $user->save();
        return true;
      });
    }
  }
}
```  

Notice how we make calls to the **_$user->assignRole_** method in order to apply our new permissions and roles.

And the last code edit is to assign the trait **_HasRoles_** to the **_app/User.php_** model:

```php
<?php

namespace App;

use Illuminate\Notifications\Notifiable;
use Illuminate\Contracts\Auth\MustVerifyEmail;
use Illuminate\Foundation\Auth\User as Authenticatable;
use Spatie\Permission\Traits\HasRoles;

class User extends Authenticatable
{
    use Notifiable, HasRoles;

    // The rest of the file continues...
```

#### Reseed the database

The final thing we need to do is reseed the database with our permissions and roles and then restart the application.  Run the following commands in the terminal:

```bash
$ php artisan migrate:refresh --seed
$ composer dump-autoload && php artisan cache:clear && php artisan serve
```

If everything went correctly the application should function as it did before we added the permissions and roles.

## Quick roles and permissions test

Let's toss in a quick test to ensure the permissions and roles are assigned to our User objects and are available to the application. We'll edit the **_resources/views/home.blade.php_** file, and modify the code for the **_<div class="card-header">Dashboard</div>_** element to the following:

{% raw %}
```html
<div class="card">
  <div class="card-header">Dashboard</div>

  <div class="card-body">
      @if (session('status'))
          <div class="alert alert-success" role="alert">
              {{ session('status') }}
          </div>
      @endif

      <p>You are logged in!</p>
      <p>Your permissions: {{ Auth::user()->getAllPermissions()->pluck('name') }}</p>
      <p>Your roles: {{ Auth::user()->getRoleNames() }}</p>

  </div>
</div>
```
{% endraw %}

Once we reload the page we should see the following for the **_administrator_** account:

![Roles and permissions check](assets/images/posts/2019/roles_and_permissions_check.png)


## Summary

We have now extended the Laravel application base template we've been working on to support proper logging in and out as well as roles and permissions.

You can find the source code for this post [on GitHub](https://github.com/nrasch/AppTemplate/tree/PartThree).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
