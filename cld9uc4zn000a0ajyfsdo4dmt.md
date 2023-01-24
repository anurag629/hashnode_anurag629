# Transforming Categorical Data: A Practical Guide to Handling Non-Numerical Variables for Machine Learning and Data Science Algorithms.

There are several ways to deal with categorical data, also known as label data, in data science:

1. One-hot encoding
    
2. Label encoding
    
3. Dummy encoding
    
4. Binning
    
5. Count Encoding
    
6. Frequency Encoding
    
7. Target Encoding
    

The appropriate technique will depend on the specific data and the goals of the analysis. It's important to note that some algorithms like decision trees and random forest can handle categorical variables directly, so encoding may not be necessary.

We will now go through all the above ways with some sample data-set and also learn how o make our data trainable.

**Let's Start**

## 1\. One-hot encoding

One-hot encoding is a technique used to convert categorical variables into numerical values by creating a binary column for each category. It is useful for handling categorical variables with multiple levels.

For example, let's say we have a dataset of hand bags with a column called "color" that contains the following values: "red", "green", and "blue".

| **color** | **price** | **units** |
| --- | --- | --- |
| red | 500 | 2 |
| green | 800 | 3 |
| blue | 300 | 1 |
| red | 400 | 1 |
| green | 600 | 1 |

One-hot encoding would create three new binary columns, one for each unique category, with a value of 1 indicating that the category is present and a value of 0 indicating that it is not. The resulting data might look like this:

| **color** | price | units | **color\_red** | **color\_green** | **color\_blue** |
| --- | --- | --- | --- | --- | --- |
| red | 500 | 2 | 1 | 0 | 0 |
| green | 800 | 3 | 0 | 1 | 0 |
| blue | 300 | 1 | 0 | 0 | 1 |
| red | 400 | 1 | 1 | 0 | 0 |
| green | 600 | 1 | 0 | 1 | 0 |

As you can see, the original "color" column has been replaced by three new binary columns, one for each unique category. Each row now has a value of 1 in exactly one of these new columns, indicating the presence of that category.

But wait, you should have one question ..... How to do it using python? So, let's do it using python.

In Python, You can use the `get_dummies()` function from the `pandas` library to apply one-hot encoding to the "color" column of your dataframe. Here is an example of how to do it:

```python
import pandas as pd

# Create example dataframe
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'green'],
                   'price': [500, 800, 300, 400, 600],
                   'units': [2, 3, 1, 1, 1]})

# Apply one-hot encoding to "color" column
df_encoded = pd.get_dummies(df, columns=['color'])

print(df_encoded)
```

Alternatively, you can use the `OneHotEncoder` class from the `sklearn.preprocessing` library to apply one-hot encoding.

```python
from sklearn.preprocessing import OneHotEncoder

# Create example dataframe
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'green'],
                   'price': [500, 800, 300, 400, 600],
                   'units': [2, 3, 1, 1, 1]})

# Create an instance of the encoder
encoder = OneHotEncoder(sparse=False)

# Fit and transform the "color" column
color_encoded = encoder.fit_transform(df[['color']])

# Create new dataframe with the encoded values
df_encoded = pd.concat([df.drop(columns=['color']), pd.DataFrame(color_encoded, columns=encoder.get_feature_names(['color']))], axis=1)

print(df_encoded)
```

The resulting dataframe will look the same as the previous one, but the columns will have a prefix 'color\_x0\_' rather 'color'.

## 2\. Label encoding

Label encoding is a technique used to convert categorical variables into numerical values by assigning a unique integer value to each category. It is useful for handling ordinal variables, where the order of the categories matters.

For example, let's say we have a dataset with a column called "size" that contains the following values: "small", "medium", "large". Label encoding would replace each category with an integer, such as: "small" = 0, "medium" = 1, "large" = 2. The resulting data might look like this:

| **size** | **encoded\_size** |
| --- | --- |
| small | 0 |
| medium | 1 |
| large | 2 |
| small | 0 |
| medium | 1 |

As you can see, the original "size" column has been replaced by "encoded\_size" column, each row now has a unique integer value representing the category.

You can use the `LabelEncoder` class from the `sklearn.preprocessing` library to apply label encoding to your data. Here is an example of how to do it:

```python
from sklearn.preprocessing import LabelEncoder

# Create example dataframe
df = pd.DataFrame({'size': ['small', 'medium', 'large', 'small', 'medium'],
                   'price': [500, 800, 300, 400, 600],
                   'units': [2, 3, 1, 1, 1]})

# Create an instance of the encoder
encoder = LabelEncoder()

# Fit and transform the "size" column
df['encoded_size'] = encoder.fit_transform(df['size'])

print(df)
```

The resulting dataframe, `df`, will have an new column "encoded\_size" representing the encoded values of size column. The resulting dataframe will look like this:

| **size** | **price** | **units** | **encoded\_size** |
| --- | --- | --- | --- |
| small | 500 | 2 | 0 |
| medium | 800 | 3 | 1 |
| large | 300 | 1 | 2 |
| small | 400 | 1 | 0 |
| medium | 600 | 1 | 1 |

It's important to note that label encoding changes the relationship between the categories. It assigns a unique number to each category, but it doesn't take into account the ordinal relationship between the categories. In this case, the encoded values of "small", "medium" and "large" are 0, 1 and 2 respectively, but it doesn't mean that small is half the size of medium or large is twice the size of medium.

## 3\. Dummy Encoding

Dummy encoding, also known as indicator encoding, is a technique used to convert categorical variables into numerical values by creating binary columns for each category, similar to one-hot encoding, but it doesn't remove any column. It is useful when working with categorical variables with many levels.

For example, let's say we have a dataset with a column called "color" that contains the following values: "red", "green", "blue". Dummy encoding would create three new binary columns, one for each unique category, with a value of 1 indicating that the category is present and a value of 0 indicating that it is not. The resulting data might look like this:

| **color** | **red** | **green** | **blue** |
| --- | --- | --- | --- |
| red | 1 | 0 | 0 |
| green | 0 | 1 | 0 |
| blue | 0 | 0 | 1 |
| red | 1 | 0 | 0 |
| green | 0 | 1 | 0 |

As you can see, the original "color" column is still present in the table, but three new binary columns, one for each unique category, has been added. Each row now has a value of 1 in exactly one of these new columns, indicating the presence of that category.

You can use the `pd.concat()` function from the `pandas` library to apply dummy encoding to the "color" column of your dataframe, here is an example of how to do it:

```python
# Create example dataframe
df = pd.DataFrame({'color': ['red', 'green', 'blue', 'red', 'green'],
                   'price': [500, 800, 300, 400, 600],
                   'units': [2, 3, 1, 1, 1]})

# Apply dummy encoding to "color" column
df_encoded = pd.concat([df, pd.get_dummies(df['color'])], axis=1)

print(df_encoded)
```

The resulting dataframe, `df_encoded`, will have three new binary columns, one for each unique category in the "color" column, with a value of 1 indicating that the category is present and a value of 0 indicating that it is not. The original "color" column is still present in the table. The resulting dataframe will look like this:

| **color** | **price** | **units** | **red** | **green** | **blue** |
| --- | --- | --- | --- | --- | --- |
| red | 500 | 2 | 1 | 0 | 0 |
| green | 800 | 3 | 0 | 1 | 0 |
| blue | 300 | 1 | 0 | 0 | 1 |
| red | 400 | 1 | 1 | 0 | 0 |
| green | 600 | 1 | 0 | 1 | 0 |

## 4\. Binning

Binning is a technique used to group numerical values into bins or ranges, it is used to handle numerical variables with a large number of unique values. Binning can be useful for creating categorical variables from numerical ones and for handling outliers in the data.

For example, let's say we have a dataset with a column called "age" that contains the following values: 18, 20, 25, 30, 35, 40, 45. To apply binning, we can divide the range of values into a pre-defined number of intervals or bins. For example, we can divide the range of ages into four bins: (18, 25\], (25, 35\], (35, 45\], (45, 50\]. This would group the ages into four categories: "young", "middle-aged", "old", and "very old". The resulting data might look like this:

| **age** | **age\_bin** |
| --- | --- |
| 18 | young |
| 20 | young |
| 25 | middle-aged |
| 30 | middle-aged |
| 35 | old |
| 40 | old |
| 45 | very old |

As you can see, the original "age" column is still present in the table, but a new column "age\_bin" has been added, which contains the binned values for each age. The rows in the "age\_bin" column now contain categorical values representing the age group.

You can use the `cut()` function from the `pandas` library to apply binning to the "age" column of your dataframe, here is an example of how to do it:

```python
# Create example dataframe
df = pd.DataFrame({'age': [18, 20, 25, 30, 35, 40, 45],
                   'price': [500, 800, 300, 400, 600, 700, 800],
                   'units': [2, 3, 1, 1, 1, 2, 3]})

# Apply binning to "age" column
df['age_bin'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 50], labels=['young', 'middle-aged', 'old', 'very old'])

print(df)
```

The resulting dataframe, `df`, will have an new column "age\_bin" representing the binned values of age column. The resulting dataframe will look like this:

| **age** | **price** | **units** | **age\_bin** |
| --- | --- | --- | --- |
| 18 | 500 | 2 | young |
| 20 | 800 | 3 | young |
| 25 | 300 | 1 | middle-aged |
| 30 | 400 | 1 | middle-aged |
| 35 | 600 | 1 | old |
| 40 | 700 | 2 | old |
| 45 | 800 | 3 | very old |

As you can see, the original "age" column is still present in the table, but a new column "age\_bin" has been added, which contains the binned values for each age. The rows in the "age\_bin" column now contain categorical values representing the age group.

## 5\. Count Encoding

Count encoding is a technique used to convert categorical variables into numerical values by counting the number of occurrences of each category in the dataset. It is used to handle categorical variables with many levels.

For example, let's say we have a dataset with a column called "product" that contains the following values: "apple", "orange", "banana", "apple", "orange", "apple", "banana". Count encoding would replace each category with the number of times it appears in the dataset. The resulting data might look like this:

| **product** | **count\_encoded** |
| --- | --- |
| apple | 3 |
| orange | 2 |
| banana | 2 |
| apple | 3 |
| orange | 2 |
| apple | 3 |
| banana | 2 |

As you can see, the original "product" column is still present in the table, but a new column "count\_encoded" has been added, which contains the count encoded values for each product. The rows in the "count\_encoded" column now contain unique integer values representing the number of times each product appears in the dataset.

You can use the `value_counts()` function from the `pandas` library to apply count encoding to the "product" column of your dataframe, here is an example of how to do it:

```python
# Create example dataframe
df = pd.DataFrame({'product': ['apple', 'orange', 'banana', 'apple', 'orange', 'apple', 'banana'],
                   'price': [500, 800, 300, 400, 600, 700, 800],
                   'units': [2, 3, 1, 1, 1, 2, 3]})

# Apply count encoding to "product" column
df['count_encoded'] = df['product'].map(df['product'].value_counts())

print(df)
```

The resulting dataframe, `df`, will have an new column "count\_encoded" representing the count encoded values of product column. The resulting dataframe will look like this:

| **product** | **price** | **units** | **count\_encoded** |
| --- | --- | --- | --- |
| apple | 500 | 2 | 3 |
| orange | 800 | 3 | 2 |
| banana | 300 | 1 | 2 |
| apple | 400 | 1 | 3 |
| orange | 600 | 1 | 2 |
| apple | 700 | 2 | 3 |
| banana | 800 | 3 | 2 |

## 6\. Frequency Encoding

Frequency encoding is a technique used to convert categorical variables into numerical values by representing each category as the proportion of occurrences of that category in the dataset. It is similar to count encoding, but it normalizes the count by dividing it by the total number of occurrences of all categories in the dataset. It is used to handle categorical variables with many levels.

For example, let's say we have a dataset with a column called "product" that contains the following values: "apple", "orange", "banana", "apple", "orange", "apple", "banana". Frequency encoding would replace each category with the proportion of times it appears in the dataset. The resulting data might look like this:

| **product** | **frequency\_encoded** |
| --- | --- |
| apple | 0.429 |
| orange | 0.286 |
| banana | 0.286 |
| apple | 0.429 |
| orange | 0.286 |
| apple | 0.429 |
| banana | 0.286 |

As you can see, the original "product" column is still present in the table, but a new column "frequency\_encoded" has been added, which contains the frequency encoded values for each product. The rows in the "frequency\_encoded" column now contain decimal values between 0 and 1 representing the proportion of times each product appears in the dataset.

You can use the `value_counts()` function from the `pandas` library to apply frequency encoding to the "product" column of your dataframe, here is an example of how to do it:

```python
# Create example dataframe
df = pd.DataFrame({'product': ['apple', 'orange', 'banana', 'apple', 'orange', 'apple', 'banana'],
                   'price': [500, 800, 300, 400, 600, 700, 800],
                   'units': [2, 3, 1, 1, 1, 2, 3]})

# Apply frequency encoding to "product" column
df['frequency_encoded'] = df['product'].map(df['product'].value_counts(normalize=True))

print(df)
```

The resulting dataframe, `df`, will have an new column "frequency\_encoded" representing the frequency encoded values of product column. The resulting dataframe will look like this:

| **product** | **price** | **units** | **frequency\_encoded** |
| --- | --- | --- | --- |
| apple | 500 | 2 | 0.428571 |
| orange | 800 | 3 | 0.285714 |
| banana | 300 | 1 | 0.285714 |
| apple | 400 | 1 | 0.428571 |
| orange | 600 | 1 | 0.285714 |
| apple | 700 | 2 | 0.428571 |
| banana | 800 | 3 | 0.285714 |

## 7\. Target Encoding

Target Encoding is a technique used to convert categorical variables into numerical values by representing each category as the mean of the target variable for that category. This technique is used when the categorical variable has a large number of levels and is also useful in situations where the data is highly imbalanced.

For example, let's say we have a dataset with a column called "product" and a target variable called "sales" that contains the following values:

| **product** | **sales** |
| --- | --- |
| apple | 100 |
| orange | 200 |
| banana | 50 |
| apple | 150 |
| orange | 300 |
| apple | 50 |
| banana | 20 |

Target encoding would replace each category in the "product" column with the mean of the "sales" column for that category. The resulting data might look like this:

| **product** | **sales** | **target\_encoded** |
| --- | --- | --- |
| apple | 100 | 83.333 |
| orange | 200 | 250.0 |
| banana | 50 | 35.0 |
| apple | 150 | 83.333 |
| orange | 300 | 250.0 |
| apple | 50 | 83.333 |
| banana | 20 | 35.0 |

As you can see, the original "product" column is still present in the table, but a new column "target\_encoded" has been added, which contains the target encoded values for each product. The rows in the "target\_encoded" column now contain decimal values representing the mean of the "sales" column for each product.

You can use the `groupby()` function from the `pandas` library to apply target encoding to the "product" column of your dataframe, here is an example of how to do it:

```python
# Create example dataframe
df = pd.DataFrame({'product': ['apple', 'orange', 'banana', 'apple', 'orange', 'apple', 'banana'],
                   'sales': [100, 200, 50, 150, 300, 50, 20]})

# Apply target encoding to "product" column
df['target_encoded'] = df.groupby('product')['sales'].transform('mean')

print(df)
```

The resulting dataframe, `df`, will have an new column "target\_encoded" representing the mean of sales column for each product. The resulting dataframe will look like this:

| **product** | **sales** | **target\_encoded** |
| --- | --- | --- |
| apple | 100 | 83.333 |
| orange | 200 | 250.0 |
| banana | 50 | 35.0 |
| apple | 150 | 83.333 |
| orange | 300 | 250.0 |
| apple | 50 | 83.333 |
| banana | 20 | 35.0 |

### This blog is a part of a #100daysdatascience series. If you want to follow the whole series then go to the below links:

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

### If you liked the post and wanted me to support then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)