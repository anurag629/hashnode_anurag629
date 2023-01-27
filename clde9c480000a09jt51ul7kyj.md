# Evaluation Metrics for Classification and Regression: A Comprehensive Guide

Machine learning models are used to make predictions and classify data. However, it's essential to evaluate the performance of these models to ensure that they are working correctly. In this article, we will discuss the various evaluation metrics that can be used for classification and regression problems. We will also cover the mathematical formulas behind these metrics and provide examples of how to implement them in Python. By the end of this article, you will have a solid understanding of how to evaluate the performance of your machine learning models.

### Regression:

1. Mean Absolute Error (MAE)
    
2. Mean Squared Error (MSE)
    
3. R-squared (R²)
    
4. Root Mean Squared Error (RMSE)
    
5. Mean Absolute Percentage Error (MAPE)
    

### Classification:

1. Accuracy
    
2. Precision
    
3. Recall
    
4. F1 Score
    
5. AUC-ROC Curve
    
6. Confusion Matrix
    

## Evaluation metrics for regression

**1\. Mean Absolute Error (MAE)**: This metric calculates the average absolute difference between the predicted values and the actual values. The lower the MAE, the better the model's predictions. In Python, you can use the `mean_absolute_error()` function from the `sklearn.metrics` library to calculate the MAE.

![MAE](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/q1p6ttp0pqfo7nzaikmd.jpeg align="left")

This formula calculates the average absolute difference between the true values and the predicted values for all samples. The smaller the MAE, the better the model's performance.

```python
from sklearn.metrics import mean_absolute_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_error(y_true, y_pred))
```

**2\. Mean Squared Error (MSE)**: This metric calculates the average of the squared differences between the predicted values and the actual values. Like the MAE, the lower the MSE, the better the model's predictions. In Python, you can use the `mean_squared_error()` function from the `sklearn.metrics` library to calculate the MSE.

![MSE](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/79zscqxk0jx3pqk3kxss.jpeg align="left")

The mean squared error is a measure of the difference between the true and predicted values, where the difference is squared to penalize larger errors more heavily. The MSE is commonly used as a loss function in machine learning, and is often used to evaluate the performance of regression models.

```python
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))
```

**3\. R-squared (R²)**: This metric, also known as the coefficient of determination, ranges from 0 to 1 and represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). An R² of 1 indicates that the model perfectly predicts the target variable, while an R² of 0 indicates that the model does not predict the target variable at all. In Python, you can use the `r2_score()` function from the `sklearn.metrics` library to calculate the R².

![R squared](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/byid0iplunzqvhp3372n.jpeg align="left")

The numerator of the formula represents the residual sum of squares (RSS), which is the sum of the squares of the differences between the actual and predicted values of the dependent variable. The denominator of the formula represents the total sum of squares (TSS), which is the sum of the squares of the differences between each value of the dependent variable and the mean value of the dependent variable.

R-squared ranges between 0 and 1. A value of 0 indicates that the model does not explain any of the variance in the dependent variable, while a value of 1 indicates that the model explains all of the variance in the dependent variable.

```python
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))
```

**4\. Root Mean Squared Error (RMSE)**: This metric calculates the square root of the MSE. It is useful because it is in the same units as the target variable, so it is easier to interpret than the MSE. In Python, you can use the `mean_squared_error()` function from the `sklearn.metrics` library to calculate the MSE, and then take the square root of that value to get the RMSE.

![RMSE](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/rhklj0xkfl8x5itclb68.jpeg align="left")

In this formula, the RMSE is the square root of the average of the squared differences between the true values and the predicted values. It is used to measure the difference between the predicted values and the true values, and it is widely used in regression models. It gives an idea of the average difference between the predicted values and the true values. The smaller the RMSE, the better the model is.

```python
from sklearn.metrics import mean_squared_error
import numpy as np
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(np.sqrt(mean_squared_error(y_true, y_pred)))
```

**5\. Mean Absolute Percentage Error (MAPE)**: It represents the difference between actual value and predicted value in percentage. It is useful when you want to know the percentage of error from actual value.

![MAPE](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/fg1l6xa26inslzjkkyne.jpeg align="left")

In this formula, the absolute value is taken on the percentage error between the actual and predicted values to ensure that the error is always positive. The percentage error is then averaged across all samples to get the mean percentage error.

```python
def mean_absolute_percentage_error(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return np.mean(np.abs((y - y_pred) / y)) * 100
    
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_absolute_percentage_error(y_true, y_pred))
```

## Evaluation metrics for classification

**1\. Accuracy**: This metric represents the proportion of correctly classified instances out of the total number of instances. It is defined as (number of correct predictions) / (number of total predictions). In Python, you can use the `accuracy_score()` function from the `sklearn.metrics` library to calculate the accuracy.

![Accuracy](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/lybsf6gn0z2j7mxpsms3.jpeg align="left")

```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 1]
print(accuracy_score(y_true, y_pred))
```

**2\. Precision**: This metric represents the proportion of true positive predictions out of all positive predictions. It is defined as (number of true positives) / (number of true positives + number of false positives). In Python, you can use the `precision_score()` function from the `sklearn.metrics` library to calculate the precision.

![pRECISION](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/obncb4spttdw7iucn1q8.jpeg align="left")

```python
from sklearn.metrics import precision_score
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 1]
print(precision_score(y_true, y_pred))
```

**3\. Recall**: This metric represents the proportion of true positive predictions out of all actual positive instances. It is defined as (number of true positives) / (number of true positives + number of false negatives). In Python, you can use the `recall_score()` function from the `sklearn.metrics` library to calculate the recall.

![Recall](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/lud0w88narc332lm43q7.jpeg align="left")

```python
from sklearn.metrics import recall_score
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 1]
print(recall_score(y_true, y_pred))
```

**4\. F1 Score**: It is the harmonic mean of precision and recall. F1 score tries to find the balance between precision and recall. It gives a good idea of how precise and robust the classifier is.

![F1 score](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/2ytlyhjnr48zeev5fx90.jpeg align="left")

```python
from sklearn.metrics import f1_score
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 1]
print(f1_score(y_true, y_pred))
```

**5\. AUC-ROC Curve**: Receiver Operating Characteristic (ROC) curve is a graphical representation of the performance of a classification model. It plots the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. The area under the ROC curve (AUC) is a measure of how well a parameter can distinguish between two diagnostic groups. AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.

The formula for the area under the receiver operating characteristic (ROC) curve (AUC) is defined as the integral of the true positive rate (TPR) with respect to the false positive rate (FPR) over the range of possible threshold values. Mathematically, it can be represented as:

![AUC ROC curve](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/r0qlr5w1eneom4uamm7g.jpeg align="left")

```python
from sklearn.metrics import roc_auc_score
y_true = [0, 1, 1, 0]
y_pred = [0.1, 0.9, 0.8, 0.2]
print(roc_auc_score(y_true, y_pred))
```

**6\. Confusion Matrix**: It is a table that is used to define the performance of a classification algorithm. It is mostly used for binary classification. The top left value represents true negatives, top right value represents false positives, bottom left value represents false negatives and bottom right value represents true positives. It is a very useful tool for understanding the performance of a classification model.

![confusion matrix](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/jlnfamgvzhy2gxt0k3ue.jpeg align="left")

```python
from sklearn.metrics import confusion_matrix
y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 1]
print(confusion_matrix(y_true, y_pred))
```

## Summary:

In this article, we discussed the various evaluation metrics that can be used for classification and regression problems, including accuracy, precision, recall, F1 score, R-squared, and the confusion matrix. We also covered the mathematical formulas behind these metrics and provided examples of how to implement them in Python. By understanding these evaluation metrics, you can ensure that your machine learning models are working correctly and make informed decisions about how to improve their performance. Remember that the choice of metrics can be different depending on the problem and the data, so it's always a good idea to experiment with different metrics to find the one that best suits your needs.

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)