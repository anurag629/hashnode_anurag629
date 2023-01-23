# Linear Regression in Python: From Data to Model

## What is Linear Regression?

Linear regression is a statistical method used for modeling the relationship between a dependent variable (also known as the outcome or response variable) and one or more independent variables (also known as predictors or explanatory variables). The goal of linear regression is to find the best-fitting line through a set of data points, where the line is defined by an equation of the form y = mx + b, where y is the dependent variable, x is the independent variable, m is the slope of the line, and b is the y-intercept. Linear regression can be used for both simple linear regression (one independent variable) and multiple linear regression (more than one independent variable).

![Linear Regression.jpeg](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/aza0en7bdsz3djgldfbv.jpeg align="left")

## Importing Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
```

## Loding train and test dataset into pandas data frame

```python
train_df = pd.read_csv("/kaggle/input/random-linear-regression/train.csv")
#Drop null values
train_df = train_df.dropna() 
train_df.head()
```

|  | x | y |
| --- | --- | --- |
| 0 | 24.0 | 21.549452 |
| 1 | 50.0 | 47.464463 |
| 2 | 15.0 | 17.218656 |
| 3 | 38.0 | 36.586398 |
| 4 | 87.0 | 87.288984 |

```python
test_df = pd.read_csv("/kaggle/input/random-linear-regression/test.csv")
# Drop null values
test_df = test_df.dropna()
test_df.head()
```

|  | x | y |
| --- | --- | --- |
| 0 | 77 | 79.775152 |
| 1 | 21 | 23.177279 |
| 2 | 22 | 25.609262 |
| 3 | 20 | 17.857388 |
| 4 | 36 | 41.849864 |

## Selection of independent and and dependent variable

We selected the columns in your data frame that we want to use for the x and y axis. For example, if you have a column called 'x' that represents the independent variable and a column called 'y' that represents the dependent variable, you can select those columns like this:

```python
train_x = train_df['x']
train_y = train_df['y']

test_x = test_df['x']
test_y = test_df['y']
```

## Visualizing the training data

To draw a linear graph using your data frame, we use the popular data visualization library in Python called Matplotlib. We imported it above.

Now we use the `plt.scatter()` function to plot the data points, and the `plt.plot()` function to plot the line of best fit.

We also use the `numpy.polyfit()` function to fit a line to the data points and get the slope and y-intercept of the line of best fit.

```python
coefficients = np.polyfit(train_x, train_y, 1)
m, b = coefficients
plt.scatter(train_x, train_y)
plt.plot(train_x, m*train_x + b)
plt.xlabel('train_x')
plt.ylabel('train_y')
plt.show()
```

![train data](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ula8eznxr1srlkh1chbm.png align="left")

## Visualizing test data

```python
coefficients = np.polyfit(test_x, test_y, 1)
m, b = coefficients
plt.scatter(test_x, test_y)
plt.plot(test_x, m*test_x + b)
plt.xlabel('test_x')
plt.ylabel('test_y')
plt.show()
```

![test data](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/klt4n8awjt674sicadz2.png align="left")

## Model Creation, training, and testing

To create a linear regression model and train and test the data using your data frame, we can use the `scikit-learn` library in Python. The first step is to import the library and the specific model you want to use.

For example, we use the `LinearRegression` class from the `sklearn.linear_model` module:

```python
from sklearn.linear_model import LinearRegression
```

Create an instance of the model.

```python
model = LinearRegression()
```

Now, we use the `fit()` method to train the model on the training data:

```python
train_x = train_x.values.reshape(-1, 1)
test_x = test_x.values.reshape(-1, 1)
```

```python
model.fit(train_x, train_y)
```

LinearRegression()

Check the coefficients of the model and the intercept using following command:

```python
print("Coefficients: ",model.coef_)
print("Intercept: ",model.intercept_)
```

Coefficients: \[1.00065638\] Intercept: -0.10726546430097272

Our model is trained, now we can use the `predict()` method to make predictions on the test data:

```python
y_pred = model.predict(test_x)
```

## Evaluating model performance

We can evaluate the performance of the model by comparing the predicted values with the actual values. There are many evaluation metrics such as `mean_absolute_error`, `mean_squared_error` or `r2_score`.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean Absolute Error: ",mean_absolute_error(test_y, y_pred))
print("Mean Squared Error: ",mean_squared_error(test_y, y_pred))
print("R2 Score: ",r2_score(test_y, y_pred))
```

Mean Absolute Error: 2.415771850041258 Mean Squared Error: 9.432922192039305 R2 Score: 0.9888014444327563

## Visualizing model performance

We can also visualize the results by plotting the test data points and the predicted line using the same approach as before.

```python
plt.scatter(test_x, test_y)
plt.plot(test_x, y_pred, color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

![model visualization](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/lnr4i41k6x9v01flb4jv.png align="left")

## End! Hope you like this...

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)