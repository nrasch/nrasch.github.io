---
layout: post
title:  "Creating a Base Laravel Project Template"
tags: [ Laravel, PHP, Web Development ]
featured_image_thumbnail: assets/images/posts/2019/creating_base_laravel_project_template_title_thumbnail.png
featured_image: assets/images/posts/2019/creating_base_laravel_project_template_title.png
featured: true
hidden: true
---

In this post we'll create a simple, base Laravel project template that we'll expand upon in future write ups to explore items such as roles and permissions, validation, CRUD via React, and so forth.

You can find the complete source code for this post [on GitHub](#).

<!--more-->

## Create the configure the base Laravel implementation

To start we create a new Laravel instance utilizing Composer:
<pre class="line-numbers">
<code class="language-bash">$ composer create-project --prefer-dist laravel/laravel AppTemplate
$ cd ./AppTemplate/
$ php artisan key:generate</code>
</pre>

Next, connect the SQL server of your choice to the application, and create an <code class="language-bash">app_template</code> database.  Don't forget to update the <code class="language-bash">config/database.php</code> and <code class="language-bash">.env</code> files with your database connection details.

Example:
<pre><code class="language-php">
// **********
// config/database.php
// **********
'default' => env('DB_CONNECTION', 'pgsql'),

'pgsql' => [
    'driver' => 'pgsql',
    'url' => env('127.0.0.1'),
    'host' => env('DB_HOST', '127.0.0.1'),
    'port' => env('DB_PORT', '5432'),
    'database' => env('DB_DATABASE', 'app_template'),
    'username' => env('DB_USERNAME', 'postgres'),
    'password' => env('DB_PASSWORD', 'password'),


// **********
// .env
// **********
  DB_CONNECTION=pgsql
  DB_HOST=127.0.0.1
  DB_PORT=5432
  DB_DATABASE=app_template
  DB_USERNAME=postgres
  DB_PASSWORD=password
</code></pre>

### Laravel's out-of-the-box authentiation

We'll also implement Laravel's out-of-the-box authentication for future customization:

<pre class="line-numbers">
<code class="language-bash">$ php artisan make:auth
$ php artisan db:migrate</code></pre>

This will make the following changes:

```bash
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   routes/web.php

Untracked files:
  (use "git add <file>..." to include in what will be committed)

	app/Http/Controllers/HomeController.php
	resources/views/auth/
	resources/views/home.blade.php
	resources/views/layouts/
```

And we can view the new routes that were added--which we'll customize later--with the following command:

<pre class="line-numbers">
<code class="language-bash">$ php artisan route:list

+--------+----------+------------------------+------------------+------------------------------------------------------------------------+--------------+
| Domain | Method   | URI                    | Name             | Action                                                                 | Middleware   |
+--------+----------+------------------------+------------------+------------------------------------------------------------------------+--------------+
|        | GET|HEAD | /                      |                  | Closure                                                                | web          |
|        | GET|HEAD | api/user               |                  | Closure                                                                | api,auth:api |
|        | GET|HEAD | home                   | home             | App\Http\Controllers\HomeController@index                              | web,auth     |
|        | GET|HEAD | login                  | login            | App\Http\Controllers\Auth\LoginController@showLoginForm                | web,guest    |
|        | POST     | login                  |                  | App\Http\Controllers\Auth\LoginController@login                        | web,guest    |
|        | POST     | logout                 | logout           | App\Http\Controllers\Auth\LoginController@logout                       | web          |
|        | POST     | password/email         | password.email   | App\Http\Controllers\Auth\ForgotPasswordController@sendResetLinkEmail  | web,guest    |
|        | GET|HEAD | password/reset         | password.request | App\Http\Controllers\Auth\ForgotPasswordController@showLinkRequestForm | web,guest    |
|        | POST     | password/reset         | password.update  | App\Http\Controllers\Auth\ResetPasswordController@reset                | web,guest    |
|        | GET|HEAD | password/reset/{token} | password.reset   | App\Http\Controllers\Auth\ResetPasswordController@showResetForm        | web,guest    |
|        | GET|HEAD | register               | register         | App\Http\Controllers\Auth\RegisterController@showRegistrationForm      | web,guest    |
|        | POST     | register               |                  | App\Http\Controllers\Auth\RegisterController@register                  | web,guest    |
+--------+----------+------------------------+------------------+------------------------------------------------------------------------+--------------+</code></pre>

## React

We also want [React](https://reactjs.org/) support for use in our future development efforts:

<pre class="line-numbers">
<code class="language-bash">$ php artisan preset react
$ npm install && npm run dev
</code></pre>

## Test what we've done so far

Let's start the application and ensure everything is working as expected so far:

<pre class="line-numbers">
<code class="language-bash">$ php artisan serve
</code></pre>

Once we browse to <code class="language-html">http://127.0.0.1:8000/</code> we should see this familiar screen if everything went correctly:

![Laravel start page](assets/images/posts/2019/laravel_start_page.png)

So far so good!

### React testing

Let's test the React functionality at this point now too.  We just need to make a few changes:

* Edit the <code class="language-bash">resources/views/layouts/welcome.blade.php</code> file, and add the following after the list of Laravel `<li>` links:

```html
<div id='ReactTest' />
```

Then add this code just before the <code class="language-bash"></body></code> tag as shown below:

{% raw %}
```html
    <!-- Laravel Application Core JS -->
    <script src="{{ mix('js/app.js') }}"></script>

  </body>
</html>
```
{% endraw %}

When the page is reloaded we should see the following, which indicates that React is working correctly:

![Laravel start page React test](assets/images/posts/2019/laravel_start_page_react_test.png)


## Generate users

And finally we should generate some users to test the authentication as well as set us up for the next write up.  We start by creating a User seeder:

<pre class="line-numbers">
<code class="language-bash">$php artisan make:seeder UserSeeder
</code></pre>

Next, add the following code to the <code class="language-bash">database/seeds/UserSeeder.php</code> file:

```php
<?php

use Illuminate\Database\Seeder;
use App\User;

class UserSeeder extends Seeder
{
    /**
     * Create application users.
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
        $user->save();

        // Create dev/test data for non-production environments
        if (env('APP_ENV') != 'production') {
            // Create N mumber of users
            factory(User::class, 20)->make()->each(function($user) {
                $user->save();
                return true;
            });
        }
    }
}
```
This will create not only an admin user, but twenty other random users as well.

In order to have the seeder populate the database we add the seeder to the <code class="language-bash">database/seeds/DatabaseSeeder.php</code> file:

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
        $this->call(UserSeeder::class);
    }
}
```

Now we can utilize the seeder to create a sample of new user accounts with the following command:

<pre class="line-numbers">
<code class="language-bash">$php artisan db:seed
</code></pre>

## Final testing

We are now ready to log in as the Admin user we created:

* Fill out the login form....

![Laravel login form page](assets/images/posts/2019/laravel_test_login_page.png)

* And success!

![Laravel login success page](assets/images/posts/2019/laravel_test_login_success.png)

## Summary

We now have a working Laravel application base template that can be utilized to develop further functionality, and that is exactly what we'll do in the next set of write ups.  :)

You can also find the source code for this post [on GitHub](#).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
