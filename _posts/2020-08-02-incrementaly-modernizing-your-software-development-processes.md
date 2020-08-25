---
layout: post
title:  "Incrementaly Modernizing Your Software Development Processes"
tags: [ Software Development, Management]
featured_image_thumbnail: assets/images/posts/2020/incrementaly-modernizing-your-software-development-processes_thumbnail.jpg
featured_image: assets/images/posts/2020/incrementaly-modernizing-your-software-development-processes_title.jpg
featured: false
hidden: false
---

In this article we explore a strategy to assist software development teams who are looking to advance their limited, initial, and/or legacy software development processes (SDP) towards a more mature state in a modular, customizable way that builds a foundation for larger modernization initiatives later on such as:
* [Continuous Integration (CI)](https://en.wikipedia.org/wiki/Continuous_integration)
* [Continuous Deployment (CD)](https://en.wikipedia.org/wiki/Continuous_delivery)
* [DevOps](https://theagileadmin.com/what-is-devops/)

We discuss the pros and cons of modernizing the SDP as well as present a customizable SDP template to help kickstart the process that supports realizing benefits along each step of the way.

<!--more-->

## Introduction

While leading projects and teams at smaller and medium sized companies, this is often the SDP process I find being used to power software development:

![Minimal SDP](assets/images/posts/2020/Minimal SDP.jpg)

Usually this is an artifact from when the team was initially forming, being scrappy, focusing on building initial [MVPs](https://www.agilealliance.org/glossary/mvp/) as quickly as possible, or just trying to survive those first few critical years and outlast the burn rate.

The problem; however, is when the team doesn’t mature their SDP model in order to take advantage of [CI](https://en.wikipedia.org/wiki/Continuous_integration)/[CD](https://en.wikipedia.org/wiki/Continuous_delivery), [DevOps](https://theagileadmin.com/what-is-devops/), and/or [agile product delivery](https://www.scaledagileframework.com/agile-product-delivery/) practices.  The organization is left vulnerable to competitors who leverage these practices as a form of competitive advantage, or is unable to pivot, deliver, and capitalize on emerging market and customer opportunities.

I watched this play out when I worked at [Return Path](https://returnpath.com).  One of our top competitors introduced what they claimed was a new “pixel tracking technology.”  To their credit, the competitor’s marketing team did a fantastic job of making a lot of buzz about it, and they pushed hard to make this a point of differentiation between our product offering and theirs.  Our sales department was being beaten up badly on calls with prospects, and we were at risk of losing market share.

Return Path; however, had made investments in their SDP pipeline, and was able to pivot and respond in the next quarter with their own “pixel tracking” offering to regain feature and competitive parity.

Imagine if Return Path would have had a clunky, outdated SDP pipeline which took months and months to deliver value to customers, and in the meantime, their competitor continued to innovate and advance their competitive lead. If that had happened Return Path could have had a huge problem on their hands.

Soon afterwards [DKIM](https://en.wikipedia.org/wiki/DomainKeys_Identified_Mail) became an emerging standard, and Return Path utilized their streamlined SLDC pipeline in order to quickly deliver a first-to-market offering in the space.   Return Path was then able to leverage their market lead aggressively to pressure other companies who were behind, set the competitive pace, and differentiate themselves on sales calls and in marketing materials.  

Delivering customer value and closing deals was clearly a better use of organizational time and resources vs. defensively responding to competitors who might have made the first move.

## What’s the Downside?

Building out your SDP in order to support moving towards full blown DevOps, CI, CD, etc. doesn’t happen overnight; some of the SDP template components suggested below will incorporate aspects of DevOps and CI at a reduced scale.  These in turn could require changes in the engineering department’s culture, processes, and infrastructure.

However; having said that, working through these changes is not a negative thing:  It gives the team much needed experience with implementing departmental pivots, and prepares the team to plan for the bigger organizational improvement challenges that are more a paradigm shift such as [organizational thinking](https://www.scaledagileframework.com/apply-systems-thinking/) for example.

Let’s examine a few quotes to back this idea up:

---

_The primary characteristic of DevOps culture is increased collaboration between the roles of development and operations. There are some important cultural shifts, within teams, and at an organizational level, that support this collaboration._  

_Rouan Wilsenach, [https://martinfowler.com/bliki/DevOpsCulture.html](https://martinfowler.com/bliki/DevOpsCulture.html)_

---

_DevOps is a mindset, a culture, and a set of technical practices. It provides communication, integration, automation, and close cooperation among all the people needed to plan, develop, test, deploy, release, and maintain a Solution._

_Scaled Agile, Inc., [https://www.scaledagileframework.com/devops/](https://www.scaledagileframework.com/devops/)_

---

_To get the full benefits of CI, you will need to automate your tests to be able to run them for every change that is made to the main repository. We insist on running tests on every branch of your repository and not just focus on the main branch. This way you will be able to capture issues early and minimize disruptions for your team._

_Atlassian, [https://www.atlassian.com/continuous-delivery/continuous-integration/how-to-get-to-continuous-integration](https://www.atlassian.com/continuous-delivery/continuous-integration/how-to-get-to-continuous-integration)_

---

_With continuous integration, the “system always runs,” meaning it’s potentially deployable, even during development. CI is most easily applied to software solutions where small, tested vertical threads can deliver value independently. In larger, multi-platform software systems, the challenge is harder. Each platform has technical constructs, and the platforms must be continuously integrated to prove new functionality. In complex systems comprised of software, hardware, and components and services provided by suppliers, CI is harder still._  

_Scaled Agile, Inc., [https://www.scaledagileframework.com/continuous-integration/](https://www.scaledagileframework.com/continuous-integration/)_

---

In each of these references--and there are countless others to be found--a cultural, architectural, and/or organizational set of alignments are highlighted as a requirement for successful implementation.  By doing this on a smaller scale as part of an incremental SDP improvement plan we gain valuable experience and lessons learned that set us up for success for the wider scale roll-out later on.

After considering these points, the amount of effort required to move from this

![Minimal SDP](assets/images/posts/2020/Minimal SDP.jpg)

to something like this

![Incremental SDP Template](assets/images/posts/2020/Incremental SDP Template.jpg)

may appear to be daunting or confusing on where to even begin. In the worst case it might appear impossible to progress, and thus preclude your forward movement altogether.

## The Solution

So hopefully at this point in the article you’re excited to start building out and leveraging your SDP capabilities!

However, you might be asking questions such as:
* How can I work through such a large initiative?  
* What if I’m not ready to go ‘all in’ and commit to a full overhaul of my SDP?  
* Is there a way for example to incrementally advance my SDP and realize the benefits as I progress?

A successful strategy I’ve utilized in the past to address these questions is to create an incremental SDP improvement plan that can be applied in modular stages.  Each portion of the plan can be leveraged independently of the others and yet still provides immediate value to the team and organization when completed.  Furthermore, the final combination of changes synergizes to create a solid foundation for larger more modern initiatives such as [SAFe](https://www.scaledagileframework.com/).   

Another benefit to an incremental improvement strategy is the fact that the team acquires hands-on experience, lessons learned, and best practices with each round of changes and adaptations.  This sort of learned flexibility and experience with change is invaluable as the organization becomes more flexible and dynamically pivots around value streams and [systems thinking](https://www.scaledagileframework.com/apply-systems-thinking/).

Let’s take a look at the SDP template diagram we posted earlier again:

![Incremental SDP Template](assets/images/posts/2020/Incremental SDP Template.jpg)

There are four main areas in the process flow:
* Software Development
* Continuous Integration
* Quality Control
* Post Deployment Validation

Each of these areas in turn is composed of standalone “modules” that can be built out in whatever order best suits your team and its needs.

So, for example, the “Software Development” area has the following modules listed as suggested implementation items:
* Linting
* Unit Tests
* SQL Execution Analysis
* Vulnerability Scans
* Version Control

Again, you can build these out in any order you wish.  As each module is completed you will be able to realize the benefits immediately, and the modules combine to support more advanced optimizations such as [DevOps](https://theagileadmin.com/what-is-devops/).

For ease of reference here are the links to the [JPG](assets/images/posts/2020/Incremental SDP Template.jpg) and [Visio](/assets/html/Incremental SDP Template.vsdx) source file for the SDP template displayed above.  

You can utilize these resources as-is, or you can adapt and modify to incorporate your own needs and ideas.  For example, perhaps you want to add [Extreme Programming](http://www.extremeprogramming.org/) into the mix or [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development) under the  “Software Development” area.  

Many of the modules in the template can be implemented by the development team solo such as Unit Tests or SQL Execution Analysis.  Others will require the development team to start breaking down the silos between development and operations such as implementing CI capabilities.

This mix should give the team a combination of experience, lessons learned, and best practices on changing and adapting both internally and in tandem with other departments as each piece of the plan is implemented.

Below I have included links to each subject area on the template for reference as well as brief commentary where appropriate:

### Linting

A linter “... is a tool that analyzes source code to flag programming errors, bugs, stylistic errors, and suspicious constructs.”  ([Source](https://en.wikipedia.org/w/index.php?title=Lint_(software)&oldid=907589761))

Linting is run by the developers before their code is sent to peer review or QC.

Think about linting like a final punch down inspection on a home that has just been built:  We want to examine the paint, drywall, electrical outlets, etc. for any obvious errors and/or defects before the official inspectors (QC in our case) perform their review.

**Fair warning**:  There is probably nothing other than the different git branching strategies that causes more fiery debate amongst developers than linting!  In particular, the “stylistic errors” portion of linting can and will be the cause of much consternation between team members, so be prepared to properly facilitate the linting meeting when you have it.  I have personally seen meetings dissolve into chaos as passions flare about the “proper” way to format and write code.

As an example of how long the debate has raged on for coding styles issues, please refer to [this article](https://thenewstack.io/spaces-vs-tabs-a-20-year-debate-and-now-this-what-the-hell-is-wrong-with-go/) that discusses the decades-long saga of spaces vs. tabs for indentation.

As we can see, this has been going on for so long now it has now entered popular culture...

#### Learn More
If you want to learn more here is the link to the [Wikipedia article](https://en.wikipedia.org/wiki/Lint_(software)), and here are two example linting libraries for [Python](https://www.pylint.org/) and [JavaScript](https://eslint.org/).

### Unit Tests
Unit tests are, in essence, code written by developers that test the features and functionality of the system. They are normally executed before the features and/or functionality are passed to QC for validation and/or executed as part of the CI process.

Example:
We want to write a login feature that accepts a username and password.  We have a number of acceptance criteria around this functionality:  We don't want to accept blank usernames or passwords, the username must exist in the user’s table of the database (DB), and the password given on the login page must match the user’s password in the user’s table of the DB.

The developer can--and should--write unit tests that send permutations of the username and password inputs to the login feature such as a blank username/password, incorrect username/password combinations, usernames/passwords containing foreign accent characters, etc.  

The unit tests are executed, and the output of the login feature is examined to ensure that in each case the proper response to the given input occurred.  So for example in the case of a blank username and/or password the login feature’s error handling should have returned a warning to the user that blank usernames and/or passwords are not acceptable values for the login form.

Once the unit tests are written they can be run on-demand by peers, QC team members, etc. to ensure the system is functioning as expected.  Later on, if changes are made to an area with unit test coverage the unit tests can be executed again to ensure everything continues to function correctly post changes.

#### Learn More
You can read more about unit testing [here](https://en.wikipedia.org/wiki/Unit_testing) and [here](http://softwaretestingfundamentals.com/unit-testing/).

And here are links to three example unit test libraries for [Python](https://docs.python.org/3/library/unittest.html), [JavaScript](https://mochajs.org/), and [Angular.js](https://docs.angularjs.org/guide/unit-testing).

### SQL Execution Analysis
Wikipedia defines SQL execution analysis as follows:

_Since SQL is declarative, there are typically many alternative ways to execute a given query, with widely varying performance. When a query is submitted to the database, the query optimizer evaluates some of the different, correct possible plans for executing the query and returns what it considers the best option. Because query optimizers are imperfect, database users and administrators sometimes need to manually examine and tune the plans produced by the optimizer to get better performance._

Typically after writing a SQL query the developer can run an “explain plan” that outlines the steps the SQL query optimizer has taken to optimize it.  The explain plan is examined for issues such as full table scans or missing indices, and these items can be addressed to decrease and optimize the DB query execution times.

I have personally seen queries go from ten-minute execution times down to a matter of seconds after a proper explain plan analysis and optimization had occurred.  The impact of a properly tuned SQL query should not be underestimated, and it can have a huge impact on the responsiveness of your application.

#### Learn More
You can read more about SQL execution analysis on Wikipedia [here](https://en.wikipedia.org/wiki/Query_plan).

And here are two concrete examples discussing explain plans for [MySQL](https://dev.mysql.com/doc/refman/8.0/en/execution-plan-information.html) and [PostgreSQL](https://www.postgresql.org/docs/9.4/using-explain.html).

### Vulnerability Scans

_A vulnerability scanner is a computer program designed to assess computers, networks or applications for known weaknesses. In plain words, these scanners are used to discover the weaknesses of a given system. They are utilized in the identification and detection of vulnerabilities arising from mis-configurations or flawed programming within a network-based asset such as a firewall, router, web server, application server, etc.  ([Source](https://en.wikipedia.org/wiki/Vulnerability_scanner))_

It is a good habit to build vulnerability scanning into your processes as early as possible.  As the business grows sooner or later a client will ask to either 1) see the results of your vulnerability scans and/or 2) more commonly want to perform their own vulnerability scans against your software as part of their auditing and due diligence.  In either case, you will want the client to receive a report that shows your system is secure, and that the client can trust you with their data and business.  

Nothing is worse than having a large business deal get fouled up while the development team writes patches and fixes to address a failed audit, or having a security breach that forces you to have to talk to your client about which pieces of their data on your system have been compromised.

#### Learn More

You can read more about vulnerability scanning [here](https://en.wikipedia.org/wiki/Vulnerability_scanner), and [here](https://owasp.org/www-community/Vulnerability_Scanning_Tools) is a link from the [Open Web Application Security Project® (OWASP)](https://owasp.org/) that lists a number of vulnerability scanning tools available on the market to address this area of concern.

### Version Control
I’m going to pretty much shamelessly plug/recommend [git](https://git-scm.com/) for version control:

_By far, the most widely used modern version control system in the world today is Git. Git is a mature, actively maintained open source project originally developed in 2005 by Linus Torvalds, the famous creator of the Linux operating system kernel. A staggering number of software projects rely on Git for version control, including commercial projects as well as open source. Developers who have worked with Git are well represented in the pool of available software development talent and it works well on a wide range of operating systems and IDEs (Integrated Development Environments)._

_Having a distributed architecture, Git is an example of a DVCS (hence Distributed Version Control System). Rather than have only one single place for the full version history of the software as is common in once-popular version control systems like CVS or Subversion (also known as SVN), in Git, every developer's working copy of the code is also a repository that can contain the full history of all changes.  ([Source](https://www.atlassian.com/git/tutorials/what-is-git))_

Fair warning:  Like linting, git branching can be a hot topic for development teams, so be prepared to facilitate the discussions and keep the meeting on track!

#### Learn More
[Here](https://www.youtube.com/watch?v=hwP7WQkmECE) is a quick video on git in one hundred seconds.

A git branching methodology I’ve utilized with success in the past can be found [here](https://nvie.com/posts/a-successful-git-branching-model/), and [Vincent Driessen](https://github.com/nvie) has written a set of Git extensions to provide high-level repository operations for the git branching model I referenced above [here](https://github.com/nvie/gitflow).

And finally, [here](https://www.git-tower.com/blog/git-hosting-services-compared/) is an article discussing a number of commercially available git hosting services.

### Automated Testing

_In software testing, test automation is the use of software separate from the software being tested to control the execution of tests and the comparison of actual outcomes with predicted outcomes.  Test automation can automate some repetitive but necessary tasks in a formalized testing process already in place or perform additional testing that would be difficult to do manually. Test automation is critical for continuous delivery and continuous testing.  ([Source](https://en.wikipedia.org/wiki/Test_automation))_

The great part about automated testing is the computer never gets tired, it always does things the same way, and you can have it repeat tests over and over.  This frees up your QC department from performing manual, time-consuming regression tests, and allows them to focus their expertise on smoke testing, edge cases, and other exploratory areas where human “fuzzy” thinking excels.

For example, assume that after every production software deployment we want to run a suite of one hundred regression tests on common areas of the application such as the login page to validate the system is performing as expected.  Now imagine we release on a weekly cycle, and you can quickly see how these one hundred tests are going to become tedious quickly.  Additionally, as testing fatigue sets in there is a possibility the QC engineer will miss an error in an area they’ve covered dozens and dozens of times.

However, if we utilize automated testing we can neatly sidestep this issue.  We can run the regression tests whenever we like and as often as we like, collect a results report each time for historical and auditing purposes, and ensure the application is performing as expected.  We can also tie the automated test suite to our CI/CD systems to have the tests executed each time a change to the code base is made ensuring constant feedback on quality.

#### Learn More
[Selenium](https://www.selenium.dev/documentation/en/) is an example of a test automation framework which I’ve utilized with great success on previous projects.  There are [other options](https://phoenixnap.com/blog/best-automation-testing-tools) available as well although many of them are built or add onto Selenium.


### Test Plans
Chances are if you are reading this you know what test plans are.  However, for completeness sake:

_A test plan documents the strategy that will be used to verify and ensure that a product or system meets its design specifications and other requirements. A test plan is usually prepared by or with significant input from test engineers.  ([Source](https://en.wikipedia.org/wiki/Test_plan))_

Furthermore, a test plan will usually confirm the acceptance criteria as defined by the product owner have been met, explore edge cases for logic/system flow errors, ensure the visual components of the system display correctly, validate system inputs and outputs, as well as other exploratory testing the QC engineer feels is appropriate.

I have had the opportunity in my career to work with a number of talented QC engineers who have impressed me with the obscure and hard-to-find issues they found through ad hoc, exploratory testing, and so providing the QC team time for this type of validation should not be discounted.  

This is also a good argument for automated testing:  The less time spent rehashing happy-path test routines, the more time QC engineers have to apply their expertise to harder-to-find issues that no doubt your most demanding, picky client will find ten minutes after release...

#### Learn More
We of course have the required [Wikipedia entry](https://en.wikipedia.org/wiki/Test_plan) for test planning, as well as [further discussion](https://devops.com/13-best-practices-successful-software-testing-projects/) on a handful of best practices for software test plan development.

As a final piece of advice:  QC is a **HUGE** subject area.  For example, here is a [link](https://www.guru99.com/types-of-software-testing.html) to an article that lists one hundred different types of software testing.  As your team and organization grows don’t skimp on finding at least a few QC experts who can fully develop your QC capabilities and train others in best practices.  

### Continuous Integration
The famous [Martin Fowler](https://martinfowler.com/) has this to say about CI:

_Continuous Integration is a software development practice where members of a team integrate their work frequently, usually each person integrates at least daily - leading to multiple integrations per day. Each integration is verified by an automated build (including test) to detect integration errors as quickly as possible. Many teams find that this approach leads to significantly reduced integration problems and allows a team to develop cohesive software more rapidly._

The frequency of work integration can be a subject of debate.  Articles are claiming that daily is a best practice, and I’ve seen other recommendations for ten plus times a day.  Likely you’ll want to work with your team and see what best fits your development style and your customers' needs.  For example, it is doubtful anyone in the banking industry wants to release ten times a day...

Also, note that implementing continuous integration is likely going to bring a few other issues to the forefront:  Linting, git branching and merging, DevOps for managing the CI server/service, and QC processes since they’ll want to test against a static set of assets (i.e. you’ll likely need a dedicated testing environment controlled by the QC team).  As such it might be best to work on this item farther down in your SDP implementation strategy once your team has some practice working together and incorporating change into their processes.  

#### Learn More
You can read more about CI [here](https://en.wikipedia.org/wiki/Continuous_integration) and [here](https://martinfowler.com/articles/continuousIntegration.html).

[Jenkins](https://www.jenkins.io/) has been the de facto CI system for most organizations I’ve worked for, and of course the big cloud hosting companies are getting in on the act as well.  For example, here is the [Amazon AWS CI offering](https://aws.amazon.com/devops/continuous-integration/) and [CircleCI](https://circleci.com/).

## Wrapping Up
Building out your SDP pipeline and then tying it to other organizational frameworks such as [SAFe](https://www.scaledagileframework.com/) or [Lean](https://www.lean.org/whatslean/) experiments doesn’t have to be a herculean, disruptive task if you take it in stages.  You can advance at a sustainable pace that meets your organizational needs, realize benefits at each step, and gain experience in applying these types of changes as you go along.

If you have any methods, strategies, or experiences building out your own SDP pipeline that worked for you, and you would like to share I would love to hear about them!  Please let me know either in the comments or via email.  

I’d also be more than happy to answer any questions you may have after reading this.

Thank you.
