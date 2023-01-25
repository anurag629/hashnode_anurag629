# Predicting Medical Costs using Multivariate Linear Regression in Python

## Multivariate Linear Regression

Multivariate linear regression is a statistical method used to model the relationship between multiple independent variables and a single dependent variable. It is an extension of simple linear regression, which only involves one independent variable. In multivariate linear regression, the goal is to find the equation that best predicts the value of the dependent variable based on the values of the independent variables. The equation is in the form of Y = a + b1X1 + b2X2 + ... + bnXn, where Y is the dependent variable, X1, X2, ..., Xn are the independent variables, a is the constant term, and b1, b2, ..., bn are the coefficients that represent the relationship between each independent variable and the dependent variable.

## What we do in this?

We accurately predict charges cost?

Columns present in dataset:

`age`: age of primary beneficiary

`sex`: insurance contractor gender, female, male

`bmi`: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.

`children`: Number of children covered by health insurance / Number of dependents

`smoker`: Smoking

`region`: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.

`charges`: Individual medical costs billed by health insurance

## Importing Important libreries

```python
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
```

## Reading files

Below code uses the `read_csv()` function from the pandas library to read in the medical insurence data from a csv file and assigns the resulting dataframe to a variable named `df`.

```python
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head()
```

|  | age | sex | bmi | children | smoker | region | charges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | female | 27.900 | 0 | yes | southwest | 16884.92400 |
| 1 | 18 | male | 33.770 | 1 | no | southeast | 1725.55230 |
| 2 | 28 | male | 33.000 | 3 | no | southeast | 4449.46200 |
| 3 | 33 | male | 22.705 | 0 | no | northwest | 21984.47061 |
| 4 | 32 | male | 28.880 | 0 | no | northwest | 3866.85520 |

## Feature engineering

Next, we applies one-hot encoding to the `sex`, `region`, and `smoker` columns of the dataframe and assigns the resulting dataframe to a new variable `df_encoded`.

```python
# Apply one-hot encoding to "color" column
df_encoded = pd.get_dummies(df, columns=['sex', 'region', 'smoker'])
df_encoded
```

|  | age | bmi | children | charges | sex\_female | sex\_male | region\_northeast | region\_northwest | region\_southeast | region\_southwest | smoker\_no | smoker\_yes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 19 | 27.900 | 0 | 16884.92400 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 1 | 18 | 33.770 | 1 | 1725.55230 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| 2 | 28 | 33.000 | 3 | 4449.46200 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| 3 | 33 | 22.705 | 0 | 21984.47061 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| 4 | 32 | 28.880 | 0 | 3866.85520 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 1333 | 50 | 30.970 | 3 | 10600.54830 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| 1334 | 18 | 31.920 | 0 | 2205.98080 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| 1335 | 18 | 36.850 | 0 | 1629.83350 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| 1336 | 21 | 25.800 | 0 | 2007.94500 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| 1337 | 61 | 29.070 | 0 | 29141.36030 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |

1338 rows × 12 columns

```python
df_encoded.columns
```

Index(\['age', 'bmi', 'children', 'charges', 'sex\_female', 'sex\_male', 'region\_northeast', 'region\_northwest', 'region\_southeast', 'region\_southwest', 'smoker\_no', 'smoker\_yes'\], dtype='object')

## Feature selection

Next, the code selects the relevant columns of the encoded dataframe to use as independent variables (X) and the dependent variable (y) for the linear regression model.

```python
X = df_encoded[['age', 'bmi', 'children', 'sex_female', 'sex_male',
       'region_northeast', 'region_northwest', 'region_southeast',
       'region_southwest', 'smoker_no', 'smoker_yes']]
y = df_encoded['charges']
```

## Preparing model

Below code splits the data into training and testing sets using the train\_test\_split function, fits the linear regression model using the training data and prints the MSE of the model.

```python
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

```python
# create a linear regression model
model = LinearRegression()
```

```python
# train the model on the training data
train_loss = []
test_loss = []

# train the model
for i in range(100):
    model.fit(X_train, y_train)
    train_loss.append(mean_squared_error(y_train, model.predict(X_train)))
    test_loss.append(mean_squared_error(y_test, model.predict(X_test)))
```

```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
```

```python
# predict the values for the training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

```python
# Plot the prediction line
plt.scatter(y_train, y_train_pred,label='train')
plt.scatter(y_test, y_test_pred,label='test')
plt.legend()
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Prediction line")
plt.show()
```

![prediction line](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/ysexj7r19ovn0b5t82hw.png align="left")

```python
# Plot the residuals
plt.scatter(y_train_pred, y_train_pred - y_train,label='train')
plt.scatter(y_test_pred, y_test_pred - y_test,label='test')
plt.legend()
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
```

![residuals](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zu6z8zfmg6pphr07b0g5.png align="left")

```python
# Plot the loss
plt.plot(train_loss, label='train')
plt.plot(test_loss, label='test')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Plot")
plt.show()
```

![Loss](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jlrchkbii5vy8ca2er2k.png align="left")

Overall, this code is performing a linear regression analysis on an insurance dataset. It begins by importing the necessary libraries for the analysis, then reads in the data from a csv file using pandas, applies one-hot encoding to certain columns, selects the relevant columns to use in the model, and finally splits the data into training and testing sets and fits a linear regression model to the training data. The last line prints the MSE of the model as a measure of performance.

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)