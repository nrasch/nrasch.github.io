---
layout: post
title:  "Combining Laravel Policies and FormRequests"
tags: [ Laravel, PHP, Web Development ]
featured_image_thumbnail: assets/images/posts/2018/12_thumbnail.jpg
featured_image: assets/images/posts/2018/12.jpg
---

Today we'll be discussing how to use Policies together with Form Requests in a Laravel application.  We'll also cover some gotchas when using these two items together, as well as a potential model for organizing the Form Request logic into a single class.

<!--more-->

## Background

The examples in this post come from a web application I built for an employer to track and report on the department's weekly goals that were reported up the organizational ladder.

As such many of the controllers, views, models, etc. will deal with tasks, reporting periods, and teams.  Hopefully it is straightforward and intuitive, but please reach out if you have any questions.

## Assumptions and Prerequisites

This article assumes you have an roles and permissions mechanism in place such as [spatie/laravel-permission](https://github.com/spatie/laravel-permission) or [JosephSilber/bouncer](https://github.com/JosephSilber/bouncer).

(Note:  My particular development stack utilizes [spatie/laravel-permission](https://github.com/spatie/laravel-permission), and the code samples below will reflect as such.)

It is also assumed you have access to a recent version of the [Laravel Framework](https://laravel.com) such as v5.8.29.

If your development stack doesn't match what I have the concepts and ideas will still apply, but you might need to adjust some of the code to comply with your particular set up.
