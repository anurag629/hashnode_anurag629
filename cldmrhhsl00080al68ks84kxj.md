# Predicting Diabetes Outcomes with Logistic Regression: A Hands-On Guide"

Logistic Regression is a powerful machine learning algorithm that is widely used in binary classification problems. In this blog, we will delve into the intricacies of Logistic Regression and understand why it is such a popular method.

## Definition:

Logistic Regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. The dependent variable in a Logistic Regression model is binary, meaning it can take on only two values (e.g. yes/no, true/false). The independent variables, on the other hand, can be continuous or categorical. Logistic Regression models the relationship between the dependent variable and independent variables by using the logistic function to produce a probability that the dependent variable is a certain value, which can then be thresholded to make a final binary prediction.

## Example:

Suppose you are tasked with predicting whether a customer will purchase a product based on their age and salary. In this case, the dependent variable is "purchase", which can be either "yes" or "no". The independent variables are "age" and "salary". The Logistic Regression model will use the logistic function to produce a probability that the customer will purchase the product based on their age and salary, which can then be thresholded to make a final binary prediction.

## Data:

To build a Logistic Regression model, you need a dataset that has a binary dependent variable and one or more independent variables. The data should be structured in a way that allows you to use the logistic function to model the relationship between the dependent variable and independent variables. The quality of the data is also critical, as a Logistic Regression model is only as good as the data it is trained on.

## Trained Model:

To train a Logistic Regression model, you need to fit the model to your training data by optimizing the coefficients of the independent variables. There are several optimization algorithms that can be used to fit a Logistic Regression model, including gradient descent, conjugate gradient, BFGS, and L-BFGS. Once the model is trained, it can be used to make predictions on new data.

## Advantages:

* Logistic Regression is a simple and easy-to-implement algorithm.
    
* It is fast and efficient for small datasets.
    
* It can handle both continuous and categorical independent variables.
    
* Logistic Regression models are easy to interpret, as the coefficients of the independent variables indicate their effect on the dependent variable.
    

## Disadvantages:

* Logistic Regression is not suitable for complex relationships between the dependent variable and independent variables.
    
* It assumes that the relationship between the dependent variable and independent variables is linear, which may not always be the case.
    
* Logistic Regression may not perform well on highly imbalanced datasets.
    

## Where to use:

* Logistic Regression is ideal for binary classification problems where the goal is to predict one of two possible outcomes.
    
* It is also a good starting point for more complex classification problems, as it provides a baseline for comparison.
    

## Where to not use:

* Logistic Regression is not suitable for complex non-linear relationships between the dependent variable and independent variables.
    
* It is also not recommended for multi-class classification problems, as it can only handle binary classification.
    

## Live model training for **Diabetics prediction using logistic regression**

## 1) Import necessary libraries:

We import pandas and numpy libraries for data processing and manipulation. train\_test\_split from scikit-learn library is used to split the data into training and testing sets. LogisticRegression from scikit-learn library is used to train the logistic regression model. confusion\_matrix and accuracy\_score from scikit-learn library are used to evaluate the model performance. Matplotlib library is used to plot the confusion matrix.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
```

## 2) Load the dataset:

We load the diabetes dataset using the pandas read\_csv function.

```python
df = pd.read_csv("/kaggle/input/diabetes-dataset/diabetes2.csv")
```

## 3) Data Visualization

```python
# Pair Plots
sns.pairplot(df, hue="Outcome")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1675320701733/1193dbcf-8b3a-464a-8d13-16a265083d85.png align="center")

## 4) Split the data into training and testing sets:

We split the data into training and testing sets using the train\_test\_split function from scikit-learn library. The training set consists of 75% of the data and the testing set consists of 25% of the data. The random\_state parameter is used to ensure reproducibility of the results.

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

## 5) Train the Logistic Regression model:

We create an instance of the Logistic Regression model and fit the model to the training data using the fit method.

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

LogisticRegression()

## 6) Make predictions on the test set:

We make predictions on the test set using the predict method.

```python
y_pred = clf.predict(X_test)
```

## 7) Evaluate the model performance:

We evaluate the model performance using the confusion matrix and accuracy score. The confusion matrix shows the number of true positive, true negative, false positive, and false negative predictions made by the model. The accuracy score indicates the percentage of correct predictions made by the model.

```python
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc*100))
```

Confusion Matrix: \[\[95 28\] \[24 45\]\] Accuracy: 72.92%

## 8) Plot the confusion matrix:

```python
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1675320736251/ef255380-7263-4322-b179-db159a628542.png align="center")

Finally, we plot the confusion matrix using matplotlib library. The confusion matrix is visualized as an image where the color intensity represents the count of predictions.

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)