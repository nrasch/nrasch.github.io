---
layout: post
title:  "Creating a Base Laravel Project Template Part Two"
tags: [ Laravel, PHP, Web Development ]
featured_image_thumbnail: assets/images/posts/2019/creating_base_laravel_project_template_title_thumbnail.png
featured_image: assets/images/posts/2019/creating_base_laravel_project_template_title.png
featured: false
hidden: false
---

In this post we continue work on our Application Template by applying a very minimalist bootstrap theme to the user interface.   

<!--more-->

You can find the complete source code for this post [on GitHub](https://github.com/nrasch/AppTemplate/tree/PartTwo).

You can also find the first post in this series [here]({% post_url 2019-07-26-creating-a-base-laravel-project-template %}).

### Why such a sparse theme?

There are a large number of very detailed, feature filled [Bootstrap](https://getbootstrap.com/) and [Material Design](https://material.io/) templates available on the internet.  However, in this instance we've gone with a much more stripped down version.  We want the end result of our efforts to be an application template we can use to kick-start new projects quickly, use for interview homework exercises, etc.  We don't want to spend hours unhooking and removing all sorts of fancy libraries and styling we might not need out of the code base.

It will be **much** easier to add new styling, libraries, single page applications, etc. to a "blank slate" then try to incorporate new functionality into a complicated web of dependencies and auto loading scripts.

## Obtain the bootstrap template

We'll be utilizing the following open source template in our application template:
<https://github.com/BlackrockDigital/startbootstrap-simple-sidebar>

## Apply the template

Because we are using such a minimal template we don't have to do much to the App Template to make everything work.  :)

### Modify resources/sass/app.scss

Append the contents of the **_startbootstrap-simple-sidebar-gh-pages/css/simple-sidebar.css_** file to the **_resources/sass/app.scss_** file.  The final result should look like this:

{% raw %}
```css
// Fonts
@import url('https://fonts.googleapis.com/css?family=Nunito');

// Variables
@import 'variables';

// Bootstrap
@import '~bootstrap/scss/bootstrap';

/*!
 * Start Bootstrap - Simple Sidebar (https://startbootstrap.com/template-overviews/simple-sidebar)
 * Copyright 2013-2019 Start Bootstrap
 * Licensed under MIT (https://github.com/BlackrockDigital/startbootstrap-simple-sidebar/blob/master/LICENSE)
 */
body {
  overflow-x: hidden;
}

#sidebar-wrapper {
  min-height: 100vh;
  margin-left: -15rem;
  -webkit-transition: margin .25s ease-out;
  -moz-transition: margin .25s ease-out;
  -o-transition: margin .25s ease-out;
  transition: margin .25s ease-out;
}

#sidebar-wrapper .sidebar-heading {
  padding: 0.875rem 1.25rem;
  font-size: 1.2rem;
}

#sidebar-wrapper .list-group {
  width: 15rem;
}

#page-content-wrapper {
  min-width: 100vw;
}

#wrapper.toggled #sidebar-wrapper {
  margin-left: 0;
}

@media (min-width: 768px) {
  #sidebar-wrapper {
    margin-left: 0;
  }

  #page-content-wrapper {
    min-width: 0;
    width: 100%;
  }

  #wrapper.toggled #sidebar-wrapper {
    margin-left: -15rem;
  }
}
```
{% endraw %}

### Modify resources/views/layouts/app.blade.php

Next we are going to replace the contents of the **_resources/views/layouts/app.blade.php_** file with the following:

{% raw %}
```html
<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- CSRF Token -->
  <meta name="csrf-token" content="{{ csrf_token() }}">

  <title>{{ config('app.name', 'Laravel') }}</title>

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
  <div id="app">

    <div class="d-flex" id="wrapper">

      <!-- Sidebar -->
      @include('components.left_nav')
      <!-- END Sidebar -->

      <!-- Page Content -->
      <div id="page-content-wrapper">

        <!-- Top Nav -->
        @include('components.top_nav')
        <!-- END Top Nav -->

        <div class="container-fluid">
          <main class="mt-4">

            @yield('content')

            <!-- React example binding  -->
            <div class="mt-5" id="example" />
            <!-- END React example binding  -->

          </main>
        </div>
        <!-- END <div class="container-fluid"> -->

      </div>
      <!-- /#page-content-wrapper -->

    </div>
    <!-- /#wrapper -->

  </div>
  <!-- END <div id="app"> -->

  <!-- Scripts -->
  <!-- Move this here and remove 'defered', or you'll have a jQuery not defined error!
  See https://stackoverflow.com/questions/51595843/laravel-5-webpack-jquery-is-not-defined -->
  <script src="{{ mix('js/app.js') }}"></script>

  <script type="text/javascript">
    $("#menu-toggle").click(function(e) {
      e.preventDefault();
      $("#wrapper").toggleClass("toggled");
    });
  </script>

  @yield('js_after')

</body>
</html>
```
{% endraw %}

This isn't much different from the out-of-the-box layout template that comes by default with a new Laravel implementation.  Changes of note include:

* We @included the left and top-nav bars (which we'll be creating next)
* We added @yield statements for CSS and Javascript assets
* We moved the **_mix('js/app.js')_** script statement to the end of the body and removed the **_defered_** statement to prevent **_jQuery not found_** errors from occurring

Because our template is using basic bootstrap components and features we didn't need to make any changes to our **_app.js_** file.



### Create the Left and Top navigation elements:

To start execute the following commands from the terminal:

<pre class="line-numbers">
<code class="language-bash">$ mkdir resources/views/components
$ touch resources/views/components/left_nav.blade.php
$ touch resources/views/components/top_nav.blade.php
</code></pre>

Next, edit the **_resources/views/components/left_nav.blade.php_** file, and enter in the following code:
{% raw %}
```html
<div class="bg-light border-right" id="sidebar-wrapper">
  <div class="sidebar-heading">Start Bootstrap </div>
  <div class="list-group list-group-flush">
    <a href="#" class="list-group-item list-group-item-action bg-light">Dashboard</a>
    <a href="#" class="list-group-item list-group-item-action bg-light">Shortcuts</a>
    <a href="#" class="list-group-item list-group-item-action bg-light">Overview</a>
    <a href="#" class="list-group-item list-group-item-action bg-light">Events</a>
    <a href="#" class="list-group-item list-group-item-action bg-light">Profile</a>
    <a href="#" class="list-group-item list-group-item-action bg-light">Status</a>
  </div>
</div>     
```

This creates the placeholder left-nav menu items.

Now edit the **_resources/views/components/top_nav.blade.php_** file and enter in the following code:
```html
<nav class="navbar navbar-expand-lg navbar-light bg-light border-bottom">
  <button class="btn btn-primary" id="menu-toggle">Toggle Menu</button>

  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>

  <div class="collapse navbar-collapse" id="navbarSupportedContent">
    <ul class="navbar-nav ml-auto mt-2 mt-lg-0">
      <li class="nav-item active">
        <a class="nav-link" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#">Link</a>
      </li>
      <li class="nav-item dropdown">
        <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
          Dropdown
        </a>
        <div class="dropdown-menu dropdown-menu-right" aria-labelledby="navbarDropdown">
          <a class="dropdown-item" href="#">Action</a>
          <a class="dropdown-item" href="#">Another action</a>
          <div class="dropdown-divider"></div>
          <a class="dropdown-item" href="#">Something else here</a>
        </div>
      </li>
    </ul>
  </div>
</nav>
```
{% endraw %}

This places the top-nav bar on the page, and you can customize as you see fit.

### Test

Let's test everything out by executing the following command:

<pre class="line-numbers">
<code class="language-bash">$ composer dump-autoload && php artisan cache:clear && php artisan serve
</code></pre>

Once you browse to **_http://localhost:8000/home_** you should see the following:

![Laravel start page first progress check](assets/images/posts/2019/part_two_progress_check_one.png)

We can observe that the HTML, CSS, Javascript, and React all loaded and are working correctly.

## Summary

We have now extended the Laravel application base template we started last post with a minimal theme.  This should allow us to use the template for quick starts on projects or in use on interview homework exercises.  We should be able easily add advanced styling, components, single page applications, etc. without collisions and bugs with the existing template assets and libraries.

Next post we'll implement roles and permissions, as well as work on a User administration CRUD component.  

You can find the source code for this post [on GitHub](https://github.com/nrasch/AppTemplate/tree/PartTwo).

If you have any comments or questions please don't hesitate to reach out.

Thanks!
