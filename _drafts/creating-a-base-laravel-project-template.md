---
layout: post
title:  "Creating a Base Laravel Project Template"
tags: [ Laravel, PHP, Web Development ]
featured_image_thumbnail: assets/images/posts/2018/12_thumbnail.jpg
featured_image: assets/images/posts/2018/12.jpg
featured: true
hidden: true
---

Creating a Base Laravel Project Template

<!--more-->

Create a new Laravel app using Composer:
<pre class="line-numbers">
<code class="language-bash">$ composer create-project --prefer-dist laravel/laravel AppTemplate
$ cd ./AppTemplate/
$ php artisan key:generate</code>
</pre>

Now log into your SQL server of choice and create an <code class="language-bash">app_template</code> database.  Don't forget to update the <code class="language-bash">config/database.php</code> and <code class="language-bash">.env</code> files with your database connection details.

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

We'll also implement Laravel's out of the box authentication and customize it in a later post:

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

We'll also set ourselves up for future [React](https://reactjs.org/) development:

<pre class="line-numbers">
<code class="language-bash">$ php artisan preset react
$ npm install && npm run dev
</code></pre>

Let's start the application and ensure everything is working as expected so far:

<pre class="line-numbers">
<code class="language-bash">$ php artisan serve
</code></pre>

Once we browse to <code class="language-html">http://127.0.0.1:8000/</code> We should see this familiar screen if everything went correctly:

![Laravel start page](assets/images/posts/2019/laravel_start_page.png)

---
