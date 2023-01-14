# Statistics for data science with practice

## Day 9 of 100 Days Data Science Bootcamp from noob to expert.

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 8

Yesterday we have studied in detail about statistics Python.

## Let's start

In today's data-driven world, understanding and working with data is an essential skill for businesses, researchers, and professionals of all backgrounds. One of the most powerful tools for data analysis is the Python programming language. With its powerful libraries, such as NumPy and pandas, Python provides a wide range of statistical and data analysis capabilities. In this article, we will explore some of the most important concepts in statistics, such as mean, median, mode, variance, and standard deviation, and learn how to use Python to calculate these values. We will also look at more advanced topics, such as percentiles, quartiles, and z-scores, and learn how to fill missing values and create new columns in a dataset. Whether you are a beginner or an experienced data analyst, this article will provide you with the knowledge and tools you need to work with data in Python.

## Mean:

The mean is the average value of a set of data. It is calculated by adding all the values in a set of data and then dividing by the number of values in the set. For example, if we have a set of data {1, 2, 3, 4, 5}, the mean would be (1 + 2 + 3 + 4 + 5) / 5 = 3.

```python
# Importing libraries
import numpy as np

# Creating a sample data set
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Mean
mean = np.mean(data)
print("Mean:", mean)
```

Mean: 5.5

## Median:

The median is the middle value of a set of data when it is arranged in numerical order. If the set has an odd number of values, the median is the middle value. If the set has an even number of values, the median is the average of the two middle values. For example, if we have a set of data {1, 2, 3, 4, 5}, the median would be 3.

```python
# Median
median = np.median(data)
print("Median:", median)
```

Median: 5.5

## Mode:

The mode is the value that appears most frequently in a set of data. A set of data can have multiple modes or no mode at all. For example, if we have a set of data {1, 2, 2, 3, 4, 5}, the mode would be 2.

```python
# Mode
import statistics as st
mode = st.mode(data)
print("Mode:", mode)
```

Mode: 1

## Range:

The range is the difference between the highest and lowest values in a set of data. For example, if we have a set of data {1, 2, 3, 4, 5}, the range would be 5 - 1 = 4.

```python
# Range
range = np.ptp(data)
print("Range:", range)
```

Range: 9

## Variance:

The variance is a measure of how much the values in a set of data deviate from the mean. It is calculated by taking the sum of the squares of the differences between each value and the mean, and then dividing by the number of values in the set.

```python
# Variance
variance = np.var(data)
print("Variance:", variance)
```

Variance: 8.25

## Standard deviation:

The standard deviation is a measure of how spread out the values in a set of data are. It is calculated by taking the square root of the variance.

```python
# Standard deviation
std_dev = np.std(data)
print("Standard deviation:", std_dev)
```

Standard deviation: 2.8722813232690143

## Percentiles and quartiles:

Percentiles and quartiles are measures of the distribution of a set of data. A percentile is a value that separates a set of data into 100 equal parts. A quartile is a value that separates a set of data into 4 equal parts.

```python
# Percentiles
percentile = np.percentile(data, [25, 50, 75])
print("25th percentile:", percentile[0])
print("50th percentile (Median):", percentile[1])
print("75th percentile:", percentile[2])
```

25th percentile: 3.25 50th percentile (Median): 5.5 75th percentile: 7.75

## Z-scores:

A z-score is a measure of how far away a value is from the mean in terms of standard deviations. It is calculated by taking the difference between a value and the mean, and then dividing by the standard deviation.

```python
# Z-scores
z_scores = (data - mean) / std_dev
print("Z-scores:", z_scores)
```

Z-scores: \[-1.5666989 -1.21854359 -0.87038828 -0.52223297 -0.17407766 0.17407766 0.52223297 0.87038828 1.21854359 1.5666989 \]

## Summary:

The key concepts of statistics such as mean, median, mode, variance, standard deviation, percentiles, quartiles, and z-scores are explained in detail, along with examples of how to calculate these values using Python libraries such as NumPy and pandas. Additionally, the article also covers more advanced topics such as filling missing values and creating new columns in a dataset. The article is suitable for both beginners and experienced data analysts, providing them with the knowledge and tools they need to work with data in Python. The article provides sample data in CSV format which can be used to practice the concepts explained.

## Exercise Question you will find in the exercise notebook of Day 6 on GitHub.

## If you liked it then...[  
](https://www.buymeacoffee.com/anurag629)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)