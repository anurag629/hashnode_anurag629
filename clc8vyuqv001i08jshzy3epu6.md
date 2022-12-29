# Data detective: Tips and tricks for conducting effective exploratory data analysis

Exploratory data analysis (EDA) is an approach to analyzing and understanding data that involves summarizing, visualizing, and identifying patterns and relationships in the data. There are many different techniques and approaches that can be used in EDA, and the specific techniques used will depend on the nature of the data and the questions being asked. Here are some common techniques that are often used in EDA:

1. Visualization: Plotting the data in various ways can help reveal patterns and trends that may not be immediately apparent. Common types of plots include scatter plots, line plots, bar plots, and histograms.
    
2. Summary statistics: Calculating summary statistics such as mean, median, and standard deviation can provide useful information about the distribution and spread of the data.
    
3. Correlation analysis: Examining the relationships between different variables can help identify correlations and dependencies.
    
4. Data cleaning: Removing missing or incorrect values and ensuring that the data is in a consistent format is an important step in EDA.
    
5. Dimensionality reduction: Techniques such as principal component analysis (PCA) can be used to reduce the number of dimensions in the data, making it easier to visualize and analyze.
    
6. Anomaly detection: Identifying unusual or unexpected values in the data can be important in identifying errors or outliers.
    
7. Feature engineering: Creating new features or transforming existing features can improve the performance of machine learning models and facilitate analysis.
    

Overall, the goal of EDA is to gain a better understanding of the data, identify potential issues or problems, and develop hypotheses about the relationships and patterns in the data that can be further tested and refined.

Now we will study in more detail all the points mentioned above.

## 1\. Visualization

Here is a simple example using a sample dataset of weather data for a single location. The data includes the temperature, humidity, and wind speed for each day in a month.

| index | Date | Temperature | Humidity | Wind Speed | Month |
| --- | --- | --- | --- | --- | --- |
| 0 | 2022-01-01 | 45 | 65 | 10 | January |
| 1 | 2022-01-02 | 50 | 70 | 15 | January |
| 2 | 2022-01-03 | 55 | 75 | 20 | January |
| 3 | 2022-01-04 | 60 | 80 | 25 | January |
| 4 | 2022-01-05 | 65 | 85 | 30 | January |
| 5 | 2022-01-06 | 70 | 90 | 35 | January |
| 6 | 2022-01-07 | 75 | 95 | 40 | January |
| 7 | 2022-01-08 | 80 | 100 | 45 | January |
| 8 | 2022-01-09 | 85 | 95 | 50 | January |
| 9 | 2022-01-10 | 90 | 90 | 55 | January |

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read in the data from a CSV file
df = pd.read_csv('weather.csv')
```

Next, we can use various types of plots to visualize the data in different ways. Here are a few examples:

**Scatter plot**:

```python
# Scatter plot of temperature vs humidity
plt.scatter(df['Temperature'], df['Humidity'])
plt.xlabel('Temperature (°F)')
plt.ylabel('Humidity (%)')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672299731189/498fe074-e2e6-418e-baea-db1f5a20ab7c.png align="center")

**Line plot**:

```python
# Line plot of temperature over time
plt.plot(df['Date'], df['Temperature'])
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672299794099/d1bf4b90-4391-4b6b-aade-e50584920789.png align="center")

**Bar plot**:

```python
# Bar plot of average temperature by month
df.groupby('Month').mean()['Temperature'].plot(kind='bar')
plt.xlabel('Month')
plt.ylabel('Temperature (°F)')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672299858106/e744bc0f-18dd-45a7-9085-a64104472c79.png align="center")

**Histogram**:

```python
# Histogram of temperature
plt.hist(df['Temperature'], bins=20)
plt.xlabel('Temperature (°F)')
plt.ylabel('Frequency')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672299940277/36609947-241d-492d-b7a8-10b170a3fe4c.png align="center")

## 2\. Summary statistics:

From same above weather data, we can do the following statistics visualization.

**Mean**:

```python
# Calculate the mean temperature
mean_temp = df['Temperature'].mean()
print(f'Mean temperature: {mean_temp:.2f}°F')
```

> Mean temperature: 67.50°F

**Median**:

```python
# Calculate the median humidity
median_humidity = df['Humidity'].median()
print(f'Median humidity: {median_humidity:.2f}%')
```

> Median humidity: 87.50%

**Standard deviation**:

```python
# Calculate the standard deviation of wind speed
std_wind_speed = df['Wind Speed'].std()
print(f'Standard deviation of wind speed: {std_wind_speed:.2f} mph')
```

> Standard deviation of wind speed: 15.14 mph

**Minimum and maximum**:

```python
# Calculate the minimum and maximum temperature
min_temp = df['Temperature'].min()
max_temp = df['Temperature'].max()
print(f'Minimum temperature: {min_temp:.2f}°F')
print(f'Maximum temperature: {max_temp:.2f}°F')
```

> Minimum temperature: 45.00°F
> 
> Maximum temperature: 90.00°F

Now, I am not sure but I can read your mind. I am sure you thought that I forgets the pandas describe data frame function but don't worry it's here.

```python
df.describe()
```

Output:

| index | Temperature | Humidity | Wind Speed |
| --- | --- | --- | --- |
| count | 10.0 | 10.0 | 10.0 |
| mean | 67.5 | 84.5 | 32.5 |
| std | 15.138251770487457 | 11.654755824698059 | 15.138251770487457 |
| min | 45.0 | 65.0 | 10.0 |
| 25% | 56.25 | 76.25 | 21.25 |
| 50% | 67.5 | 87.5 | 32.5 |
| 75% | 78.75 | 93.75 | 43.75 |
| max | 90.0 | 100.0 | 55.0 |

I hope this helps! Let me know if you have any questions or if you would like to see examples of other summary statistics.

## 3\. Correlation analysis:

Here is an example using a sample dataset of student grades:

| index | Student | Midterm | Final |
| --- | --- | --- | --- |
| 0 | Alice | 80 | 85 |
| 1 | Bob | 75 | 70 |
| 2 | Charlie | 90 | 95 |
| 3 | Dave | 65 | 80 |
| 4 | Eve | 85 | 90 |
| 5 | Frank | 70 | 75 |
| 6 | Gary | 95 | 100 |
| 7 | Holly | 60 | 65 |
| 8 | Ivy | 80 | 85 |
| 9 | Jill | 75 | 80 |

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd
import seaborn as sns

# Read in the data from a CSV file
df = pd.read_csv('student_grades.csv')
```

To analyze the correlations between different variables, we can use a variety of techniques. Here are a few examples:

**Scatter plot**:

```python
# Scatter plot of midterm grades vs final grades
sns.scatterplot(x='Midterm', y='Final', data=df)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672301277560/4c812100-5abd-4586-bd0e-69a1f6c7b1ce.png align="center")

**Correlation matrix**:

```python
# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672301369630/991ff7c3-c37f-4718-a7c3-2d0d7bc3c896.png align="center")

**Linear regression**:

```python
# Linear regression of midterm grades vs final grades
sns.lmplot(x='Midterm', y='Final', data=df)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1672301528663/6f480679-c6a6-4201-bfc2-0d05c0365068.png align="center")

As you know it is a hard task and also time taking to cover any topic in detail but here I have provided a summary of the Correlation analysis.

Correlation analysis is a statistical method used to identify the strength and direction of the relationship between two variables. It is commonly used in exploratory data analysis to understand the relationships between different variables in a dataset and to identify patterns and trends.

There are several different measures of correlation, including Pearson's correlation coefficient, Spearman's rank correlation coefficient, and Kendall's tau. These measures range from -1 (perfect negative correlation) to 1 (perfect positive correlation), with 0 indicating no correlation.

To perform correlation analysis, you can use various techniques such as scatter plots, correlation matrices, and linear regression. Scatter plots can be used to visualize the relationship between two variables, and correlation matrices can be used to visualize the correlations between multiple variables. Linear regression can be used to fit a line to the data and assess the strength of the relationship between the variables.

It is important to note that correlation does not imply causation, meaning that the presence of a correlation between two variables does not necessarily mean that one variable causes the other. It is always important to consider other factors that may be influencing the relationship between the variables.

## 4\. Data cleaning:

Here is an example using a sample dataset of student grades with some missing and incorrect values:

| index | Student | Midterm | Final |
| --- | --- | --- | --- |
| 0 | Alice | 80.0 | 85.0 |
| 1 | Bob | 75.0 | 70.0 |
| 2 | Charlie | 90.0 | 95.0 |
| 3 | Dave | 65.0 | 80.0 |
| 4 | Eve | 85.0 | 90.0 |
| 5 | Frank | 70.0 | 75.0 |
| 6 | Gary | 95.0 | 100.0 |
| 7 | Holly | 60.0 | 65.0 |
| 8 | Ivy | 80.0 | 85.0 |
| 9 | Jill | 75.0 | 80.0 |
| 10 | Kim | 90.0 | NaN |
| 11 | Larry | 70.0 | 75.0 |
| 12 | Mandy | NaN | 80.0 |
| 13 | Nancy | 95.0 | 105.0 |

This dataset includes the names of students and their grades on a midterm and final exam. Some of the values are missing (indicated by empty cells) and some of the values are incorrect (e.g. a final grade of 105).

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd

# Read in the data from a CSV file
df = pd.read_csv('student_grades_with_errors.csv')
```

Here are a few examples of data cleaning techniques that can be used to address missing and incorrect values:

**Identifying missing values**:

```python
# Check for missing values
df.isnull().sum()
```

> Student 0
> 
> Midterm 1
> 
> Final 1
> 
> dtype: int64

**Dropping rows with missing values**:

```python
# Drop rows with missing values
df.dropna(inplace=True)
```

**Filling missing values with a placeholder value**:

```python
# Fill missing values with a placeholder value (-999)
df.fillna(-999, inplace=True)
```

**Replacing incorrect values**:

```python
# Replace incorrect values (e.g. grades above 100) with a placeholder value (-999)
df['Midterm'].mask(df['Midterm'] > 100, -999, inplace=True)
df['Final'].mask(df['Final'] > 100, -999, inplace=True)
```

There is much more in data cleaning but I have provided some general things.

Data cleaning is the process of identifying and addressing issues with the data, such as missing or incorrect values, inconsistent formats, and outliers. It is an important step in the data analysis process as it helps ensure that the data is accurate, consistent, and ready for analysis.

There are a variety of techniques that can be used for data cleaning, depending on the specific issues with the data and the desired outcome. Some common techniques include:

* Identifying missing values: Use functions such as `isnull()` or `notnull()` to identify cells that contain missing values.
    
* Dropping rows with missing values: Use the `dropna()` function to remove rows that contain missing values.
    
* Filling missing values: Use the `fillna()` function to fill missing values with a placeholder value (e.g. 0 or -999).
    
* Replacing incorrect values: Use functions such as `mask()` or `replace()` to replace incorrect values with a placeholder value.
    

It is important to carefully consider the appropriate approach for addressing missing or incorrect values, as simply dropping rows or filling missing values with a placeholder value may not always be the best solution. It is often helpful to investigate the cause of the missing or incorrect values and consider whether there may be other factors that need to be taken into account.

## 5\. Dimensionality reduction:

Here is a sample dataset of student grades with three variables (midterm grades, final grades, and attendance):

| index | Student | Midterm | Final | Attendance |
| --- | --- | --- | --- | --- |
| 0 | Alice | 80 | 85 | 90 |
| 1 | Bob | 75 | 70 | 85 |
| 2 | Charlie | 90 | 95 | 100 |
| 3 | Dave | 65 | 80 | 80 |
| 4 | Eve | 85 | 90 | 85 |
| 5 | Frank | 70 | 75 | 70 |
| 6 | Gary | 95 | 100 | 95 |
| 7 | Holly | 60 | 65 | 60 |
| 8 | Ivy | 80 | 85 | 80 |
| 9 | Jill | 75 | 80 | 75 |

This dataset includes the names of students, their grades on a midterm and final exam, and their attendance percentage. The grades are out of 100 and the attendance percentage is out of 100.

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd
from sklearn.decomposition import PCA

# Read in the data from a CSV file
df = pd.read_csv('student_grades_with_attendance.csv')
```

One common technique for dimensionality reduction is principal component analysis (PCA). PCA is a linear transformation technique that projects the data onto a lower-dimensional space, reducing the number of variables while still retaining as much of the variance as possible.

Here is an example of using PCA to reduce the dimensionality of the data from three variables to two:

```python
# Select only the numeric columns
data = df.select_dtypes(include='number')

# Perform PCA
pca = PCA(n_components=2)
pca.fit(data)

# Transform the data
transformed_data = pca.transform(data)

# Print the explained variance ratio for each principal component
print(pca.explained_variance_ratio_)
```

> \[0.90800073 0.06447863\]

Summary for the same for tips and note point:

Dimensionality reduction is the process of reducing the number of variables in a dataset while still retaining as much of the information as possible. It is often used in machine learning and data analysis to reduce the complexity of the data and improve the performance of algorithms.

There are a variety of techniques for dimensionality reduction, including principal component analysis (PCA), linear discriminant analysis (LDA), and t-distributed stochastic neighbor embedding (t-SNE). These techniques can be used to transform the data into a lower-dimensional space, typically by projecting the data onto a smaller number of orthogonal (uncorrelated) dimensions.

PCA is a linear transformation technique that projects the data onto a lower-dimensional space by finding the directions in which the data varies the most. LDA is a supervised learning technique that projects the data onto a lower-dimensional space by maximizing the separation between different classes. t-SNE is a nonlinear dimensionality reduction technique that projects the data onto a lower-dimensional space by preserving the local structure of the data.

It is important to carefully consider the appropriate dimensionality reduction technique for a given dataset, as the choice of technique can have a significant impact on the results.

## 6\. Anomaly detection:

Here is an example using a sample dataset of student grades with some anomalous values:

| index | Student | Midterm | Final |
| --- | --- | --- | --- |
| 0 | Alice | 80 | 85 |
| 1 | Bob | 75 | 70 |
| 2 | Charlie | 90 | 95 |
| 3 | Dave | 65 | 80 |
| 4 | Eve | 85 | 90 |
| 5 | Frank | 70 | 75 |
| 6 | Gary | 95 | 100 |
| 7 | Holly | 60 | 65 |
| 8 | Ivy | 80 | 85 |
| 9 | Jill | 75 | 80 |
| 10 | Kim | 110 | 100 |
| 11 | Larry | 70 | 75 |
| 12 | Mandy | 50 | 60 |
| 13 | Nancy | 95 | 105 |

This dataset includes the names of students and their grades on a midterm and final exam. The grades are out of 100. The values for Kim's midterm grade (110) and Nancy's final grade (105) are anomalous, as they are much higher than the other values in the dataset.

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Read in the data from a CSV file
df = pd.read_csv('student_grades_with_anomalies.csv')
```

One common technique for anomaly detection is isolation forest, which is a type of unsupervised machine learning algorithm that can identify anomalous data points by building decision trees on randomly selected subsets of the data and using the number of splits required to isolate a data point as a measure of abnormality.

Here is an example of using isolation forest to detect anomalous values in the midterm grades:

```python
# Create an isolation forest model
model = IsolationForest(contamination=0.1)

# Fit the model to the data
model.fit(df[['Midterm']])

# Predict the anomalies
anomalies = model.predict(df[['Midterm']])

# Print the anomalies
print(anomalies)
```

> \[ 1 1 1 1 1 1 1 1 1 1 -1 1 -1 1 \]
> 
> /usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but IsolationForest was fitted with feature names warnings.warn(

The `contamination` parameter specifies the expected proportion of anomalous values in the data. In this example, we set it to 0.1, which means that we expect 10% of the values to be anomalous.

I hope this helps! Let me know if you have any questions or if you would like to see examples of other anomaly detection techniques.

More about it:

Anomaly detection, also known as outlier detection, is the process of identifying data points that are unusual or do not conform to the expected pattern of the data. It is often used in a variety of applications, such as fraud detection, network intrusion detection, and fault diagnosis.

There are a variety of techniques for anomaly detection, including statistical methods, machine learning algorithms, and data mining techniques. Statistical methods involve calculating statistical measures such as mean, median, and standard deviation, and identifying data points that are significantly different from the expected values. Machine learning algorithms such as isolation forests and one-class support vector machines can be trained on normal data and used to identify anomalies in new data. Data mining techniques such as clustering can be used to identify data points that are significantly different from the majority of the data.

It is important to carefully consider the appropriate technique for a given dataset, as the choice of technique can have a significant impact on the results. It is also important to consider the specific context and requirements of the application, as well as the cost of false positives and false negatives.

## 7\. Feature engineering

Feature engineering is the process of creating new features (variables) from the existing data that can be used to improve the performance of machine learning models. It is an important step in the data analysis process as it can help extract more meaningful information from the data and enhance the predictive power of models.

There are a variety of techniques for feature engineering, including:

* Combining multiple features: Creating new features by combining existing features using arithmetic operations or logical statements.
    
* Deriving new features from existing features: Creating new features by applying mathematical transformations or aggregations to existing features.
    
* Encoding categorical variables: Converting categorical variables into numerical form so that they can be used in machine learning models.
    

It is important to carefully consider the appropriate approach for feature engineering for a given dataset, as the choice of features can have a significant impact on the results. It is often helpful to explore the data and identify potential opportunities for feature engineering, such as combining or transforming variables to better capture relationships or patterns in the data.

Here is an example using a sample dataset of student grades:

| index | Student | Midterm | Final | Gender |
| --- | --- | --- | --- | --- |
| 0 | Alice | 80 | 85 | Female |
| 1 | Bob | 75 | 70 | Male |
| 2 | Charlie | 90 | 95 | Male |
| 3 | Dave | 65 | 80 | Male |
| 4 | Eve | 85 | 90 | Female |
| 5 | Frank | 70 | 75 | Male |
| 6 | Gary | 95 | 100 | Male |
| 7 | Holly | 60 | 65 | Female |
| 8 | Ivy | 80 | 85 | Female |
| 9 | Jill | 75 | 80 | Female |

First, we will import the necessary libraries and read in the data from a CSV file:

```python
import pandas as pd

# Read in the data from a CSV file
df = pd.read_csv('student_grades.csv')
```

Feature engineering is the process of creating new features (variables) from the existing data that can be used to improve the performance of machine learning models. There are a variety of techniques for feature engineering, including:

**Combining multiple features**:

```python
# Create a new feature by combining two existing features
df['Total'] = df['Midterm'] + df['Final']
```

**Deriving new features from existing features**:

```python
# Create a new feature by dividing one feature by another
df['Average'] = df['Total'] / 2

# Create a new feature by taking the square root of a feature
import numpy as np
df['Sqrt_Midterm'] = np.sqrt(df['Midterm'])
```

**Encoding categorical variables**:

```python
# One-hot encode a categorical feature
df = pd.get_dummies(df, columns=['Gender'])
```

After doing feature engineering data frame look like this:

| index | Student | Midterm | Final | Total | Average | Sqrt\_Midterm | Gender\_Female | Gender\_Male |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Alice | 80 | 85 | 165 | 82.5 | 8.94427190999916 | 1 | 0 |
| 1 | Bob | 75 | 70 | 145 | 72.5 | 8.660254037844387 | 0 | 1 |
| 2 | Charlie | 90 | 95 | 185 | 92.5 | 9.486832980505138 | 0 | 1 |
| 3 | Dave | 65 | 80 | 145 | 72.5 | 8.06225774829855 | 0 | 1 |
| 4 | Eve | 85 | 90 | 175 | 87.5 | 9.219544457292887 | 1 | 0 |
| 5 | Frank | 70 | 75 | 145 | 72.5 | 8.366600265340756 | 0 | 1 |
| 6 | Gary | 95 | 100 | 195 | 97.5 | 9.746794344808963 | 0 | 1 |
| 7 | Holly | 60 | 65 | 125 | 62.5 | 7.745966692414834 | 1 | 0 |
| 8 | Ivy | 80 | 85 | 165 | 82.5 | 8.94427190999916 | 1 | 0 |
| 9 | Jill | 75 | 80 | 155 | 77.5 | 8.660254037844387 | 1 | 0 |

> # Did you learn something new from this post? Let us know in the comments!