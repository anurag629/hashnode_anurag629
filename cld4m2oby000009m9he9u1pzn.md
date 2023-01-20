# Supervised Learning

Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.

Supervised learning can be separated into two types of problems when data mining—classification and regression:

* Classification uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined. Common classification algorithms are linear classifiers, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.
    
* Regression is used to understand the relationship between dependent and independent variables. It is commonly used to make projections, such as for sales revenue for a given business. Linear regression, logistical regression, and polynomial regression are popular regression algorithms.
    

### **How Supervised Learning Works?**

In supervised learning, models are trained using labelled dataset, where the model learns about each type of data. Once the training process is completed, the model is tested on the basis of test data (a subset of the training set), and then it predicts the output.

The working of Supervised learning can be easily understood by the below example and diagram:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674224136255/a1b9b9c1-80e0-4fd7-b125-e66dd296ab26.png align="center")

Suppose we have a dataset of different types of shapes which includes square, rectangle, triangle, and Polygon. Now the first step is that we need to train the model for each shape.

* If the given shape has four sides, and all the sides are equal, then it will be labelled as a **Square**.
    
* If the given shape has three sides, then it will be labelled as a **triangle**.
    
* If the given shape has six equal sides then it will be labelled as **hexagon**.
    

Now, after training, we test our model using the test set, and the task of the model is to identify the shape.

The machine is already trained on all types of shapes, and when it finds a new shape, it classifies the shape on the bases of a number of sides, and predicts the output.

### Types of supervised Machine learning Algorithms:

Supervised learning can be further divided into two types of problems:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1674224208383/2d9648b0-22aa-4ee5-bf64-33af670a32cd.png align="center")

**1\. Regression**

Regression algorithms are used if there is a relationship between the input variable and the output variable. It is used for the prediction of continuous variables, such as Weather forecasting, Market Trends, etc. Below are some popular Regression algorithms which come under supervised learning:

* Linear Regression
    
* Regression Trees
    
* Non-Linear Regression
    
* Bayesian Linear Regression
    
* Polynomial Regression
    

**2\. Classification**

Classification algorithms are used when the output variable is categorical, which means there are two classes such as Yes-No, Male-Female, True-false, etc.

Spam Filtering,

* Random Forest
    
* Decision Trees
    
* Logistic Regression
    
* Support vector Machines
    

### Advantages of Supervised learning:

* With the help of supervised learning, the model can predict the output on the basis of prior experiences.
    
* In supervised learning, we can have an exact idea about the classes of objects.
    
* Supervised learning model helps us to solve various real-world problems such as **fraud detection, spam filtering**, etc.
    

### Disadvantages of supervised learning:

* Supervised learning models are not suitable for handling the complex tasks.
    
* Supervised learning cannot predict the correct output if the test data is different from the training dataset.
    
* Training required lots of computation times.
    
* In supervised learning, we need enough knowledge about the classes of object.
    

## Supervised Machine Learning Examples

Here are some of supervised machine learning examples models used in different business applications:

### Image and object recognition

Supervised machine learning is used to locate, categorise and isolate objects from images or videos, which is useful when applied to different imagery analysis and vision techniques. The primary goal of image or object recognition is to identify the image accurately.

**Example:** *We use the ML to recognise the image precisely as if it is the image of the plane or a car or if the image is of a cat or a dog.*

### Predictive analytics

Supervised machine learning models are widely used in building predictive analytics systems, which provide in-depth insights into different business data points. This enables the organisations to predict certain results using the output given by the system. It also helps business leaders to make decisions for the betterment of the company.

**Example 1:** We *may use supervised learning to predict house prices. Data having details about the size of the house, price, the number of rooms in the house, garden and other features are needed. We need data about various parameters of the house for thousands of houses and it is then used to train the data. This trained supervised machine learning model can now be used to predict the price of a house.*

**Example 2:** *Spam detection is another area where most organisations use supervised machine learning algorithms. Data scientists classify different parameters to differentiate between official mail or spam mail. They use these algorithms to train the database such that the trained database recognise patterns in new data and classify them into spam and non-spam communication efficiently.*

### Sentiment analysis

Organisations can use supervised machine learning algorithms to predict customer sentiments. They use the algorithms to extract and categorise important information from large data sets like emotions, intent and context with little human interference. This model of supervised learning is also used to predict the sentiments of the text. This information is highly useful to gain insights about customer needs and help to improve brand-customer engagement efforts.

**Example:** Some o*rganisations, especially e-commerce stores, often try to identify the sentiments of their customer via product reviews posted on their applications or websites.*

  

**Reference:**

* **Javapoint**
    
* **IBM**
    
* **Indeed**