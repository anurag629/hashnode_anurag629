# Uncovering the Best Email Spam Classifier: A Comparative Study of Machine Learning Algorithms

## Agenda/Aim:

**1) Preprocess the data:** Clean the data and remove any irrelevant information. As our data is already in numerical form so we don't need to convert it.

**2) Train the models:** Train several supervised classification models such as Logistic Regression, KNN, SVM, Naive Bayes, Decision Trees, Random Forest, and Gradient Boosting using the preprocessed data.

**3) Evaluate the models:** Evaluate the performance of the models using metrics such as accuracy, precision, recall, and F1-score.

**4) Choose the best model:** Based on the evaluation, choose the best model that provides the highest accuracy and has the best overall performance.

The overall aim of this project is to train a machine learning model on the given email data to predict whether an email is spam or not spam, and to choose the best model for this classification task.

## About the dataset

The dataset is from kagge [Link](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

The emails.csv file contains 5172 rows, each row for each email. There are 3002 columns. The first column indicates Email name. The name has been set with numbers and not recipients' name to protect privacy. The last column has the labels for prediction : 1 for spam, 0 for not spam. The remaining 3000 columns are the 3000 most common words in all the emails, after excluding the non-alphabetical characters/words. For each row, the count of each word(column) in that email(row) is stored in the respective cells. Thus, information regarding all 5172 emails are stored in a compact dataframe rather than as separate text files.

## Import libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
```

## Load and Preprocess the data

Below code will load the data from the csv file into a pandas dataframe, remove the first column which is the email number, replace all non-numeric characters with NaN values, fill the missing values with 0, convert the data into integer type, and store it as dataframe named 'df'.

```python
# Load the data from csv file into a pandas dataframe
df = pd.read_csv("/kaggle/input/email-spam-classification-dataset-csv/emails.csv")

# Remove the first column (Email name) as it is not relevant for the prediction
df = df.drop(columns=['Email No.'])

# Replace non-numeric characters with NaN values
df = df.replace(r'[^\d.]+', value=float('nan'), regex=True)

# Fill missing values with 0
df.fillna(0, inplace=True)

# Convert the data into integer type
df = df.astype(int)
df
```

|  | the | to | ect | and | for | of | a | you | hou | in | ... | connevey | jay | valued | lay | infrastructure | military | allowing | ff | dry | Prediction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 8 | 13 | 24 | 6 | 6 | 2 | 102 | 1 | 27 | 18 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 | 0 | 8 | 0 | 0 | 4 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 5 | 22 | 0 | 5 | 1 | 51 | 2 | 10 | 1 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 7 | 6 | 17 | 1 | 5 | 2 | 57 | 0 | 9 | 3 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5167 | 2 | 2 | 2 | 3 | 0 | 0 | 32 | 0 | 0 | 5 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5168 | 35 | 27 | 11 | 2 | 6 | 5 | 151 | 4 | 3 | 23 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 5169 | 0 | 0 | 1 | 1 | 0 | 0 | 11 | 0 | 0 | 1 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 5170 | 2 | 7 | 1 | 0 | 2 | 1 | 28 | 2 | 0 | 8 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 5171 | 22 | 24 | 5 | 1 | 6 | 5 | 148 | 8 | 2 | 23 | ... | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

5172 rows × 3001 columns

## Train the models

```python
# Split the data into features (X) and labels (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

```python
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

```python
# Define a dictionary to store the results of each model
results = {}
```

```python
# Train Logistic Regression model
model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)

# Predict the target values for test set
y_pred_LR = model_LR.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_LR = accuracy_score(y_test, y_pred_LR)
precision_LR = precision_score(y_test, y_pred_LR)
recall_LR = recall_score(y_test, y_pred_LR)
f1_LR = f1_score(y_test, y_pred_LR)

# Store the results of Logistic Regression model in the dictionary
results["Logistic Regression"] = {"accuracy": accuracy_LR, 
                                  "precision": precision_LR, 
                                  "recall": recall_LR, 
                                  "f1_score": f1_LR
                                 }
```

```python
# Train KNN model
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)

# Predict the target values for test set
y_pred_KNN = model_KNN.predict(X_test)

# Evaluate the KNN model
accuracy_KNN = accuracy_score(y_test, y_pred_KNN)
precision_KNN = precision_score(y_test, y_pred_KNN)
recall_KNN = recall_score(y_test, y_pred_KNN)
f1_KNN = f1_score(y_test, y_pred_KNN)

# Store the results of KNN model in the dictionary
results["KNN"] = {"accuracy": accuracy_KNN, 
                  "precision": precision_KNN, 
                  "recall": recall_KNN, 
                  "f1_score": f1_KNN
                 }
```

```python
# Train SVM model
model_SVM = SVC()
model_SVM.fit(X_train, y_train)

# Predict the target values for test set
y_pred_SVM = model_SVM.predict(X_test)

# Evaluate the SVM model
accuracy_SVM = accuracy_score(y_test, y_pred_SVM)
precision_SVM = precision_score(y_test, y_pred_SVM)
recall_SVM = recall_score(y_test, y_pred_SVM)
f1_SVM = f1_score(y_test, y_pred_SVM)

# Store the results of SVM model in the dictionary
results["SVM"] = {"accuracy": accuracy_SVM, 
                  "precision": precision_SVM, 
                  "recall": recall_SVM, 
                  "f1_score": f1_SVM
                 }
```

```python
# Train Naive Bayes model
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)

# Predict the target values for test set
y_pred_NB = model_NB.predict(X_test)

# Evaluate the Naive Bayes model
accuracy_NB = accuracy_score(y_test, y_pred_NB)
precision_NB = precision_score(y_test, y_pred_NB)
recall_NB = recall_score(y_test, y_pred_NB)
f1_NB = f1_score(y_test, y_pred_NB)

# Store the results of Naive Bayes model in the dictionary
results["Naive Bayes"] = {"accuracy": accuracy_NB, 
                          "precision": precision_NB, 
                          "recall": recall_NB, 
                          "f1_score": f1_NB
                         }
```

```python
# Train Decision Tree model
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)

# Predict the target values for test set
y_pred_DT = model_DT.predict(X_test)

# Evaluate the Decision Tree model
accuracy_DT = accuracy_score(y_test, y_pred_DT)
precision_DT = precision_score(y_test, y_pred_DT)
recall_DT = recall_score(y_test, y_pred_DT)
f1_DT = f1_score(y_test, y_pred_DT)

# Store the results of Decision Tree model in the dictionary
results["Decision Tree"] = {"accuracy": accuracy_DT, 
                            "precision": precision_DT, 
                            "recall": recall_DT, 
                            "f1_score": f1_DT
                           }
```

```python
# Train Random Forest model
model_RF = RandomForestClassifier()
model_RF.fit(X_train, y_train)

# Predict the target values for test set
y_pred_RF = model_RF.predict(X_test)

# Evaluate the Random Forest model
accuracy_RF = accuracy_score(y_test, y_pred_RF)
precision_RF = precision_score(y_test, y_pred_RF)
recall_RF = recall_score(y_test, y_pred_RF)
f1_RF = f1_score(y_test, y_pred_RF)

# Store the results of Random Forest model in the dictionary
results["Random Forest"] = {"accuracy": accuracy_RF, 
                            "precision": precision_RF, 
                            "recall": recall_RF, 
                            "f1_score": f1_RF
                           }
```

```python
# Train Logistic Regression model
model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)

# Predict the target values for test set
y_pred_LR = model_LR.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_LR = accuracy_score(y_test, y_pred_LR)
precision_LR = precision_score(y_test, y_pred_LR)
recall_LR = recall_score(y_test, y_pred_LR)
f1_LR = f1_score(y_test, y_pred_LR)

# Store the results of Logistic Regression model in the dictionary
results["Logistic Regression"] = {"accuracy": accuracy_LR, 
                                  "precision": precision_LR, 
                                  "recall": recall_LR, 
                                  "f1_score": f1_LR
                                 }
```

## Visualization of performance of all the models

```python
# Print the final results of all models
print(results)
```

**Output:**

> {'Logistic Regression': {'accuracy': 0.9652173913043478, 'precision': 0.9220338983050848, 'recall': 0.9543859649122807, 'f1\_score': 0.9379310344827585}, 'KNN': {'accuracy': 0.8541062801932368, 'precision': 0.7018072289156626, 'recall': 0.8175438596491228, 'f1\_score': 0.7552674230145866}, 'SVM': {'accuracy': 0.7951690821256039, 'precision': 0.8067226890756303, 'recall': 0.3368421052631579, 'f1\_score': 0.4752475247524753}, 'Naive Bayes': {'accuracy': 0.9584541062801932, 'precision': 0.8853503184713376, 'recall': 0.9754385964912281, 'f1\_score': 0.9282136894824708}, 'Decision Tree': {'accuracy': 0.9285024154589372, 'precision': 0.8625429553264605, 'recall': 0.8807017543859649, 'f1\_score': 0.8715277777777777}, 'Random Forest': {'accuracy': 0.9652173913043478, 'precision': 0.9368421052631579, 'recall': 0.9368421052631579, 'f1\_score': 0.9368421052631579}}

```python
# Plot the accuracy of all models
plt.figure(figsize=(10,5))
plt.bar(results.keys(), [result["accuracy"] for result in results.values()])
plt.title("Accuracy of Different Models")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
```

![Accuracy](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/n8b95qzy9yzwzsttumgk.png align="left")

```python
# Plot the precision of all models
plt.figure(figsize=(10,5))
plt.bar(results.keys(), [result["precision"] for result in results.values()])
plt.title("Precision of Different Models")
plt.xlabel("Models")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.show()
```

![Precision](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/oeshrkdwiu7a3kbgwptx.png align="left")

```python
# Plot the recall of all models
plt.figure(figsize=(10,5))
plt.bar(results.keys(), [result["recall"] for result in results.values()])
plt.title("Recall of Different Models")
plt.xlabel("Models")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.show()
```

![Recall](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/qbrxecp8b97vj6pgzckp.png align="left")

```python
# Plot the F1-score of all models
plt.figure(figsize=(10,5))
plt.bar(results.keys(), [result["f1_score"] for result in results.values()])
plt.title("F1-Score of Different Models")
plt.xlabel("Models")
plt.ylabel("F1-Score")
plt.ylim(0, 1)
plt.show()
```

![F1 Score](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/1mi4xl86o53bbshkng1m.png align="left")

```python
# extract metrics values
accuracies = [results[model]['accuracy'] for model in results]
precisions = [results[model]['precision'] for model in results]
recalls = [results[model]['recall'] for model in results]
f1_scores = [results[model]['f1_score'] for model in results]

# plot bar chart
bar_width = 0.2
index = np.arange(len(results))

plt.bar(index, accuracies, bar_width, label='Accuracy')
plt.bar(index + bar_width, precisions, bar_width, label='Precision')
plt.bar(index + 2 * bar_width, recalls, bar_width, label='Recall')
plt.bar(index + 3 * bar_width, f1_scores, bar_width, label='F1 Score')

plt.xticks(index + bar_width, list(results.keys()))
plt.legend()
plt.show()
```

![Bar Plot](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/iv8jg8uup8klj0v3he5x.png align="left")

**Based on the model\_performance result, the Logistic Regression model seems to be the best among all the models with highest accuracy (0.965), precision (0.922), recall (0.954) and f1\_score (0.938). This suggests that the Logistic Regression model has the highest ability to correctly identify emails as spam or not spam, with minimal false positives and false negatives.**

**It is worth noting that the performance of a model depends on the specific use case and the problem at hand. For example, if the cost of misclassifying an email as spam when it is not is higher, then a model with a high recall (even if its precision is lower) might be more desirable.**

### Any questions then comment please... I will be happy to answer them 😊

### If you want detailed knowledge of evaluation metrices then check out this article [Link](https://dev.to/anurag629/evaluation-metrics-for-classification-and-regression-a-comprehensive-guide-47hb)

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Hope you liked it...

## Sharing love and knowledge...♡ ♥💕❤😘

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)