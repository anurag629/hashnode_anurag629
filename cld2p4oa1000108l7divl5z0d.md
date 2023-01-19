# Introduction to machine learning

Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. There are three main types of machine learning: supervised learning, unsupervised learning and reinforcement learning.

1. **Supervised Learning**: In supervised learning, the algorithm is trained on a labeled dataset, where the correct output is already known. The goal is to learn a general rule that maps inputs to outputs, so that when a new input is encountered, the system can predict the correct output. Examples of supervised learning include regression and classification problems.
    
2. **Unsupervised Learning**: In unsupervised learning, the algorithm is not given any labeled data. Instead, the goal is to find patterns or relationships in the data, such as grouping similar data points together. Examples of unsupervised learning include clustering and dimensionality reduction.
    
3. **Reinforcement Learning**: In reinforcement learning, the algorithm learns by interacting with its environment and receiving feedback in the form of rewards or penalties. The goal is to learn a policy that maximizes the cumulative reward over time. Examples of reinforcement learning include playing games, controlling robots and autonomous vehicles.
    

Each of these types of machine learning have their own specific use cases, and are used to solve different types of problems.

![Traditional vs machine learning](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ub5lbgzhzo4v92wvff62.png align="left")

## What are machine learning models?

A machine learning model, like a piece of clay, can be molded into many different forms and serve many different purposes. A more technical definition would be that a machine learning model is a block of code or framework that can be modified to solve different but related problems based on the data provided.

A model is an extremely generic program (or block of code), made specific by the data used to train it. It is used to solve different problems.

### Example 1

Imagine you own a snow cone cart, and you have some data about the average number of snow cones sold per day based on the high temperature. You want to better understand this relationship to make sure you have enough inventory on hand for those high sales days.

![Snow covers sold](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jbu2gsgahbdhjcscwwsy.png align="left")

In the graph above, you can see one example of a model, a linear regression model (indicated by the solid line). You can see that, based on the data provided, the model predicts that as the high temperature for the day increases so do the average number of snow cones sold. Sweet!

### Example 2

Let's look at a different example that uses the same linear regression model, but with different data and a completely different question to answer. Imagine that you work in higher education and you want to better understand the relationship between the cost of enrollment and the number of students attending college. In this example, our model predicts that as the cost of tuition increases, the number of people attending college is likely to decrease.

![Srudent enrolment](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/tb6d6o2xcv6qqos8anyy.png align="left")

Using the same linear regression model (indicated by the solid line), you can see that the number of people attending college does go down as the cost increases.

Both examples showcase that a model is a generic program made specific by the data used to train it.

##Training and using a model

### How are model training algorithms used to train a model?

In the preceding section, we talked about two key pieces of information: a model and data. In this section, we show you how those two pieces of information are used to create a trained model. This process is called model training.

### Model training algorithms work through an interactive process

Let's revisit our clay teapot analogy. We've gotten our piece of clay, and now we want to make a teapot. Let's look at the algorithm for molding clay and how it resembles a machine learning algorithm:

* Think about the changes that need to be made. The first thing you would do is inspect the raw clay and think about what changes can be made to make it look more like a teapot. Similarly, a model training algorithm uses the model to process data and then compares the results against some end goal, such as our clay teapot.
    
* Make those changes. Now, you mold the clay to make it look more like a teapot. Similarly, a model training algorithm gently nudges specific parts of the model in a direction that brings the model closer to achieving the goal.
    
* Repeat. By iterating these steps over and over, you get closer and closer to what you want, until you determine that you’re close enough and then you can stop.
    

![red rock](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/m870b9p6n8qqerqeccyr.png align="left")

## Major steps in the machine learning process

![steps in machine learning](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/yydzpelvmv15kc0ay6h3.png align="left")

Here’s a quick recap of the terms introduced in this lesson:

* Clustering is an unsupervised learning task that helps to determine if there are any naturally occurring groupings in the data.
    
* A categorical label has a discrete set of possible values, such as "is a cat" and "is not a cat."
    
* A continuous (regression) label does not have a discrete set of possible values, which means there are potentially an unlimited number of possibilities.
    
* Discrete is a term taken from statistics referring to an outcome that takes only a finite number of values (such as days of the week).
    
* A label refers to data that already contains the solution.
    
* Using unlabeled data means you don't need to provide the model with any kind of label or solution while the model is being trained.
    

***Resource: Amazon AWS***