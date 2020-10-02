---
layout: post
title:  "DevOps is a Cultural Shift, Not a Job Title"
tags: [ DevOps ]
featured_image_thumbnail: assets/images/posts/2020/devops-is-a-cultural-shift-not-a-job-title_thumbnail.jpeg
featured_image: assets/images/posts/2020/devops-is-a-cultural-shift-not-a-job-title_title.jpeg
featured: false
hidden: false
---

In this article, we argue that there is no such thing as a “DevOps” job title.  Instead, “DevOps” refers to a cultural shift within the organization to break down the silos between software development and operations to enhance the organization's ability to deliver value, generating projects and services.

We discuss what “DevOps” is and why you can’t hire yourself into it by title alone.

<!--more-->

## Introduction

As part of my job, it seems almost daily I have recruiters and sales people writing to me about their candidates and services they offer.  Lately, I’ve been noticing an uptick in “DevOps” offerings, as this clearly is a hot new sales term.  I’ve also seen a number of companies with “DevOps” job openings popup in my feeds as well.

However, I believe this term “DevOps” is being used in the wrong context.  To put it simply, DevOps is a culture, a way of doing things, and NOT a job title.  In fact, I would argue that pushing DevOps as a job title is an anti-pattern to the actual DevOps paradigm and way of doing things.

So, how do we go about convincing ourselves this argument is correct, and that the recruiters that keep spamming me don’t somehow have some inside scoop that I’m missing?


## What is “DevOps”?

First, let’s look at some external sources to help us define what “DevOps” is:

_DevOps is the practice of operations and development engineers participating together in the entire service lifecycle, from design through the development process to production support._  &nbsp;&nbsp;[The Agile Admin](https://theagileadmin.com/what-is-devops/)

The author of the quote above goes into quite some length to define the term DevOps, and it is worth a read.

Next, we have a quote from Martin Fowler’s website:

_The primary characteristic of DevOps culture is increased collaboration between the roles of development and operations. There are some important cultural shifts, within teams and at an organizational level that support this collaboration._  &nbsp;&nbsp;[Source](https://martinfowler.com/bliki/DevOpsCulture.html)


Scaled Agile also weighs in on the matter:

_DevOps is a mindset, a culture, and a set of technical practices. It provides communication, integration, automation, and close cooperation among all the people needed to plan, develop, test, deploy, release, and maintain a solution._  &nbsp;&nbsp;[Source](https://www.scaledagileframework.com/devops/)

Wikipedia also seems to agree:

_DevOps is a set of practices that combines software development (Dev) and IT operations (Ops). It aims to shorten the systems development life cycle and provide continuous delivery with high software quality. DevOps is complementary with Agile software development; several DevOps aspects came from Agile methodology._  &nbsp;&nbsp;[Source](https://en.wikipedia.org/wiki/DevOps)

And finally, a video, which I’ve found to be very helpful when explaining DevOps to non-technical folks:&nbsp;&nbsp;[What is DevOps? - In Simple English](https://www.youtube.com/watch?v=_I94-tJlovg)

## The Common Theme

A common theme is clear in the resources cited above:  DevOps “is a mindset, a culture, and a set of technical practices” that enables developers and operations to work together to deliver value to the organization.  It breaks down the departmental silos of development and operations and allows the two teams to work directly together on projects.

So, for example, assume we are creating a new microservice to calculate sales tax.  First, there is the software component of writing and testing the code that calculates the sales tax, and then there is the operational components of placing the code in a container, deploying, scaling, and monitoring the container, etc.

Instead of having the development team pass the code over to the operations team once it’s written, they work together from project inception till project completion to build a complete solution.  As the code is being written, for example, development and operations can discuss and plan for the load they anticipate the service being placed under, which regions it might need to be available in for latency mitigation, how the service will be monitored to ensure it is performing correctly, what API access controls are in place, how the service will interact with the frontend API gateway, and so on and so forth.

This collaboration from the start completely removes the hand-off issues that often occur when a solution is passed over from the development silo into the operations silo.  Because members from both disciplines are involved, any gotchas or special configurations that need to take place are dealt with inside the development cycle, thus decreasing the time to deployment, removing miscommunications, and ensuring everyone has equal ownership of the solution.

Now, let’s contrast this with two example job openings which I’ve anonymized for this article:

```
JOB: Senior DevOps Engineer
TYPE: Full-Time

Qualifications:   
• Minimum of 1 year of GoLang experience
• Expert in Google Cloud Platform  
• Expert with Kubernetes & Docker
• Expert with SQL / Postgres   
• Leadership in implementing CI/CD, SRE
• Experience implementing and maintaining compliance with various security standards (ISO 27000, SOC 2, etc.)
```

I don’t know about you, but this very much looks like the requirements for an operations/system administrator to me.

Let’s examine another:

```
TITLE: DevOps Engineer

ROLE AND RESPONSIBILITIES:
• Implement and change AWS to deploy and host development and testing components
• Coordinate major AWS environment changes with primary AWS team
• Implement Jenkins pipelines, creating Docker images and orchestration of builds
• Write and update automated scripts for installation of server software products
• Implement highly scalable & available applications using AWS services. – EC2, VPC, ELB, S3, AutoScaling, Cloudwatch, IAM, etc.
• Monitor a system’s performance and reliability as well as the daily data processing
• Perform production installations and upgrades of server software products

```

This position is also clearly for an operations/system administrator...

## Is This All Simply Semantics?

At first glance, I may seem to be acting semantically petty about job titles.  However, what concerns me is twofold:

1. Job postings, such as those above, seem to imply that the hiring organization simply thinks that “DevOps” is another term for operations/system administrators.
2. By hiring a “DevOps” role that is really an operations role in disguise, they are simply recreating the silos that “DevOps” seeks to remove (i.e., this is propagating an anti-pattern).

__Remember, software developers can be “DevOps” too!__

I worry that people are going to hire a few “DevOps” roles and suddenly think they are a “DevOps” shop, not understanding the paradigm shift and work that must occur to reap the benefits.  They are going to pay lots of extra money for that fancy new title and get what exactly?

Sadly, things aren’t as simple as a new hire or two; adding “DevOps” employees alone won’t do it.  Your organization must go through the work to put in place new cultural practices, combine development and operations within agile teams, build the [CI](https://en.wikipedia.org/wiki/Continuous_integration)/[CD](https://en.wikipedia.org/wiki/Continuous_delivery) pipelines, etc.  

A DevOps culture isn’t going to happen overnight, and hiring a “DevOps” engineer isn’t a magical shortcut.  Yes, the new hire can provide input into your transformational plans based on their past experiences (assuming, of course, they have dealt with a DevOps rollout), but the organization still has to have the willpower, commitment to change, and leadership buy-in to make it happen.

So, instead of trying to recruit a “DevOps Engineer,” how about something like this:

```
Position:  Operations Engineer with DevOps Experience

Description:  Work as part of a cross-functional agile team to deliver products and services as defined by the organization’s strategic roadmap.  Work collaboratively with software developers and testers to build out and deploy scalable, maintainable, secure, and monitored services in a cloud environment.  Have a T-shaped skill set with expertise in operations, including <Insert whatever tech stack here…  Kubernetes, for example>.
```

Or maybe something like this for the software development side of the equation:

```
Position:  Software Development Engineer with DevOps Experience

Description:  Work as part of a cross-functional agile team to deliver products and services as defined by the organization’s strategic roadmap.  Work collaboratively with operations and testers to build out and deploy scalable, maintainable, secure, and monitored services in a cloud environment.  Have a T-shaped skill set with expertise in operations, including <Insert whatever software development stack here…  React for example>.
```

Notice how both positions read almost exactly the same.  This is because they are part of the same team with the same objective:  Delivering solutions and business value as part of an agile team.  The only real difference is where the expertise part of their T-shaped skill set lies:  Either in operations or software development.

Also, notice that I keep mentioning T-shaped skill sets.  Over time, via collaboration as part of a cross-functional agile team, the developers and operations will organically engage in knowledge transfer and sharing.  While the software developers may never have the same level of expertise that the operations does on server management, for example; over time, they will gain a general understanding of the concepts, best practices, etc.  And vice versa for the operations personnel learning more about the software development process.

This provides us with an additional benefit by having experts on hand to clear critical bottlenecks and blockers, and non-expert generalists to help out on non-critical tasks to ensure that the experts can focus on where we really need them.  ([Here](https://medium.com/@jchyip/why-t-shaped-people-e8706198e437) is a good resource on T-shaped skill sets in teams for additional reading.)

If I saw something like my example job postings above, I would be much more assured that the hiring company understood how “DevOps” works.  It would be clear to me what role each position was going to play, and the benefits those positions would provide the organization for the salary investment (i.e., [ROI](https://www.investopedia.com/terms/r/returnoninvestment.asp)).  There would be more to the hire than service to a hot new buzzword, and instead, a strategic investment would be made to enable the company to deliver value to the customers faster while maintaining quality and supportability.

## Wrapping Up

As modern cloud-based application development continues to mature, new paradigms and frameworks will also continue to emerge, such as DevOps, Scaled Agile, and Lean Software Development.  These concepts promise companies the ability to deliver more, faster, and better in the competitive online space that the internet has become, and support the implementation of technologies and infrastructure, such as containerization, serverless applications, Infrastructure as Code ([IaC](https://en.wikipedia.org/wiki/Infrastructure_as_code)), and the like.

However, in many cases, the shift to these paradigms and technologies requires an investment from the organization to build a new culture, mindset, and ways of doing things.  Doing this, however, isn't as simple as hiring operations staff with a “DevOps” title.  It requires building cross-functional agile teams that remove the silos between developers and operations and having the organizational commitment to support this paradigm shift.

Thank you for reading!
