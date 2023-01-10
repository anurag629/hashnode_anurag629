# Mastering Pandas: A Comprehensive Guide with Exercises

# Day 5 of 100 Days Data Science Bootcamp from noob to expert.

# GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

# Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 4

Yesterday we have studied in detail about NumPy in Python.

# Let's Start

Pandas is a powerful data analysis and manipulation library in Python. It allows you to easily access, select, and manipulate data in your dataset. In this post, you'll learn how to use Pandas to create a gradebook for tracking student grades. You'll learn how to read in data from a CSV file, manipulate the data, and create a report of the grades. You'll also learn how to handle missing values and prepare your data for visualization. By the end of this course, you'll be able to efficiently use Pandas to manage and analyze your data.

**Let's say we have a dataset of student grades in a CSV file called "grades.csv". The first step in exploring this dataset with Pandas is to read it into a Pandas DataFrame. We can do this using the read\_csv() function:**

```python
import pandas as pd

df = pd.read_csv('grades.csv')
df
```

|  | name | grade |
| --- | --- | --- |
| 0 | John | 89.0 |
| 1 | Mary | 95.0 |
| 2 | Emily | 77.0 |
| 3 | Michael | 82.0 |
| 4 | Rachel | NaN |

* Now that we have our DataFrame, we can start exploring the data. Let's say we want to access the grades of a specific student. We can do this by selecting the row of the student and then selecting the 'grade' column:
    

```python
student_name = 'John'
grade = df[df['name'] == student_name]['grade']
print(grade)
```

0 89.0 Name: grade, dtype: float64

* We can also select a specific column by its label using the '\[\]' operator:
    

```python
grades = df['grade']
print(grades)
```

0 89.0 1 95.0 2 77.0 3 82.0 4 NaN Name: grade, dtype: float64

* If we want to select multiple columns, we can pass a list of column labels to the '\[\]' operator:
    

```python
student_info = df[['name', 'grade']]
print(student_info)
```

name grade 0 John 89.0 1 Mary 95.0 2 Emily 77.0 3 Michael 82.0 4 Rachel NaN

* Now let's say we have some missing values in our dataset. We can handle these missing values using the fillna() function:
    

```python
df = df.fillna(-1)
df
```

|  | name | grade |
| --- | --- | --- |
| 0 | John | 89.0 |
| 1 | Mary | 95.0 |
| 2 | Emily | 77.0 |
| 3 | Michael | 82.0 |
| 4 | Rachel | \-1.0 |

This will replace all missing values with -1.

#### This is basic overview of pandas. Now we will go deep dive into it.

## 1\. Importing and reading in data:

* read in data from a variety of sources, such as a CSV file:
    

```python
import pandas as pd
df = pd.read_csv('people_data.csv')
df
#Or a Excel file:

# df = pd.read_excel('data.xlsx')
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 0 | John | 20 | Male |
| 1 | Jane | 30 | Female |
| 2 | Bob | 40 | Male |
| 3 | Alice | 50 | Female |

## 2\. Inspecting data:

Once you have your data in a pandas DataFrame, you can use various methods to inspect it.

For example, you can view the first few rows of the data using the `head()` method:

```python
df.head()
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 0 | John | 20 | Male |
| 1 | Jane | 30 | Female |
| 2 | Bob | 40 | Male |
| 3 | Alice | 50 | Female |

You can also view the column names and data types using the `info()` method:

```python
df.info()
```

&lt;class 'pandas.core.frame.DataFrame'&gt; RangeIndex: 4 entries, 0 to 3 Data columns (total 3 columns): # Column Non-Null Count Dtype --- ------ -------------- ----- 0 Name 4 non-null object 1 Age 4 non-null int64 2 Gender 4 non-null object dtypes: int64(1), object(2) memory usage: 224.0+ bytes

## 3\. Selecting data:

You can select specific columns or rows of data using the `[]` operator or the `loc` and `iloc` attributes.

For example, to select the "Name" and "Age" columns, you can use the following code:

```python
df[['Name', 'Age']]
```

|  | Name | Age |
| --- | --- | --- |
| 0 | John | 20 |
| 1 | Jane | 30 |
| 2 | Bob | 40 |
| 3 | Alice | 50 |

To select rows with a specific value in a certain column, you can use the `loc` attribute:

```python
df.loc[df['Gender'] == 'Female']
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 1 | Jane | 30 | Female |
| 3 | Alice | 50 | Female |

## 4\. Manipulating data:

You can use various methods to manipulate data in a pandas DataFrame.

For example, you can add a new column by assigning a value to a new column name:

```python
df['County'] = ["India", "USA", "India", "Canada"]
df
```

|  | Name | Age | Gender | County |
| --- | --- | --- | --- | --- |
| 0 | John | 20 | Male | India |
| 1 | Jane | 30 | Female | USA |
| 2 | Bob | 40 | Male | India |
| 3 | Alice | 50 | Female | Canada |

You can also drop columns or rows using the `drop()` method:

```python
newdf = df.drop('County', axis=1)  # drop a column
newdf
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 0 | John | 20 | Male |
| 1 | Jane | 30 | Female |
| 2 | Bob | 40 | Male |
| 3 | Alice | 50 | Female |

```python
newdf1 = df.drop(df[df['Age'] < 35].index, inplace=True)  # drop rows with Age < 18
```

```python
df
```

|  | Name | Age | Gender | County |
| --- | --- | --- | --- | --- |
| 2 | Bob | 40 | Male | India |
| 3 | Alice | 50 | Female | Canada |

## 5\. Grouping and aggregating data:

You can group data by specific values and apply an aggregation function using the `groupby()` method and the `apply()` function:

```python
import numpy as np
groupdf = df.groupby('Gender')['Age'].apply(np.mean)  # group by Gender and calculate mean Age
groupdf
```

Gender Female 50.0 Male 40.0 Name: Age, dtype: float64

## 6\. Merging and joining data:

You can merge or join data from multiple DataFrames using the `merge()` function or the `concat()` function.

For example, to merge two DataFrames based on a common column, you can use the following code:

```python
df1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})
df1
```

|  | key | A | B |
| --- | --- | --- | --- |
| 0 | K0 | A0 | B0 |
| 1 | K1 | A1 | B1 |
| 2 | K2 | A2 | B2 |
| 3 | K3 | A3 | B3 |

```python
df2 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})
df2
```

|  | key | C | D |
| --- | --- | --- | --- |
| 0 | K0 | C0 | D0 |
| 1 | K1 | C1 | D1 |
| 2 | K2 | C2 | D2 |
| 3 | K3 | C3 | D3 |

```python
merged_df = pd.merge(df1, df2, on='key')
merged_df
```

|  | key | A | B | C | D |
| --- | --- | --- | --- | --- | --- |
| 0 | K0 | A0 | B0 | C0 | D0 |
| 1 | K1 | A1 | B1 | C1 | D1 |
| 2 | K2 | A2 | B2 | C2 | D2 |
| 3 | K3 | A3 | B3 | C3 | D3 |

To concatenate data horizontally (i.e. adding columns), you can use the `concat()` function:

```python
concat_df = pd.concat([df1, df2], axis=1)
concat_df
```

|  | key | A | B | key | C | D |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | K0 | A0 | B0 | K0 | C0 | D0 |
| 1 | K1 | A1 | B1 | K1 | C1 | D1 |
| 2 | K2 | A2 | B2 | K2 | C2 | D2 |
| 3 | K3 | A3 | B3 | K3 | C3 | D3 |

## 7\. Handling missing data:

It's common to encounter missing data in real-world datasets. Pandas provides various methods to handle missing data, such as filling missing values with a specific value or dropping rows with missing values.

To fill missing values with a specific value, you can use the `fillna()` method:

```python
data = {'Name': ['John', 'Jane', 'Bob', 'Alice'],
        'Age': [20, 30, 40, np.nan],
        'Gender': ['Male', 'Female', 'Male', 'Female']}
df = pd.DataFrame(data)
df
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 0 | John | 20.0 | Male |
| 1 | Jane | 30.0 | Female |
| 2 | Bob | 40.0 | Male |
| 3 | Alice | NaN | Female |

```python
df['Age'].fillna(value='22', inplace=True)
df
```

|  | Name | Age | Gender |
| --- | --- | --- | --- |
| 0 | John | 20.0 | Male |
| 1 | Jane | 30.0 | Female |
| 2 | Bob | 40.0 | Male |
| 3 | Alice | 22 | Female |

To drop rows with missing values, you can use the `dropna()` method:

```python
df.dropna(inplace=True)
```

## 8\. Working with dates and times:

Pandas has built-in support for working with dates and times.

You can convert a column of strings to datetime objects using the `to_datetime()` function:

```python
df['Date'] = pd.to_datetime(df['Date'])
```

You can then extract specific parts of the datetime, such as the year or month, using the `dt` attribute:

```python
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
```

## 9\. Advanced operations:

There are many more advanced operations that you can perform with pandas, such as pivot tables, time series analysis, and machine learning. Here are a few more code snippets to help you explore these topics:

To create a pivot table, you can use the `pivot_table()` function:

```python
pivot_table = df.pivot_table(index='Column1', columns='Column2', values='Column3', aggfunc=np.mean)
```

To perform time series analysis, you can use the `resample()` method to resample data at a different frequency:

```python
resampled_df = df.resample('D').mean()  # resample to daily frequency
```

## 9\. Visualizing data:

You can use the `plot()` method to create various types of plots, such as bar plots, scatter plots, and line plots:

```python
df.plot(x='X Column', y='Y Column', kind='scatter')  # scatter plot
df.plot(x='X Column', y='Y Column', kind='bar')  # bar plot
df.plot(x='X Column', y='Y Column')  # line plot
```

**In conclusion, pandas is a powerful and versatile library for data manipulation and analysis in Python. With its wide range of built-in functions and methods, pandas makes it easy to work with a variety of data sources, perform complex data operations, and visualize results. Whether you're a beginner or an experienced data scientist, pandas is an essential tool for any data-related project.**

# **Exercise Question you will find in the exercise notebook of Day 5 on GitHub.**

# If you liked it then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)