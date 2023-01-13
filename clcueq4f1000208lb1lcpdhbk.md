# Linear Algebra for Data Science: Understanding and Applying Vectors, Matrices and their Operations using Numpy

## Day 8 of 100 Days Data Science Bootcamp from noob to expert.

## GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

## Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 7

Yesterday we have studied in detail about DBMS/SQL and also executed all quries using python Python.

## Linear Algebra Part 1

## Let's Start

Linear algebra is a branch of mathematics that deals with vectors, matrices, and linear transformations. It plays an important role in data science and machine learning, as many data science algorithms are based on linear algebra concepts.

### Some of the key concepts in linear algebra that are important for data science include:

### 1\. Vectors:

A vector is a mathematical object that has both magnitude and direction. It is often represented as an array of numbers, and can be thought of as a point in space. In data science, vectors are used to represent data points, feature values, and other quantities. In linear algebra, vectors can be added, subtracted, and multiplied by scalars (numbers).

**Go in more detail:** [**Vectors**](https://www.cuemath.com/geometry/vectors/)

### 2\. Matrices:

A matrix is a two-dimensional array of numbers. It is used to represent systems of linear equations, and can be thought of as a collection of vectors. Matrices can be added, subtracted, and multiplied by scalars and other matrices. In data science, matrices are used to represent data, such as a dataset with multiple features, and to perform linear algebraic operations.

**Go in more detail:** [**Matrices/ Transpose of matrics/ Inverse of matrix/ Determinant of matrix/ Trance of matrix/ Dot product/ Eigen values and eigen vectors**](https://www.cuemath.com/algebra/solve-matrices/)

### 3\. Transpose of a matrix:

The transpose of a matrix is obtained by flipping the matrix over its diagonal. This operation changes the rows of the matrix into columns and vice versa. The transpose of a matrix is denoted by the superscript T. It is useful in a variety of linear algebra operations, such as solving systems of linear equations and calculating dot products.

### 4\. Inverse of a matrix:

The inverse of a matrix is a matrix that, when multiplied with the original matrix, results in the identity matrix. Not all matrices have an inverse, but square matrices (matrices with the same number of rows and columns) that are non-singular (have a non-zero determinant) do have an inverse. The inverse of a matrix is denoted by the superscript -1. The inverse of a matrix is useful in a variety of linear algebra operations, such as solving systems of linear equations and calculating matrix inverses.

### 5\. Determinant of a matrix:

The determinant of a matrix is a scalar value that can be calculated from a matrix. The determinant is used in linear algebra to solve systems of linear equations, and it can also be used to calculate the inverse of a matrix. The determinant of a matrix can be calculated using a variety of methods, including the use of cofactors, Laplace expansions, or LU decomposition.

### 6\. Trace of a matrix:

The trace of a matrix is the sum of the diagonal elements of a matrix. It is a scalar value that can be used to calculate other matrix characteristics such as eigenvalues. The trace of a matrix is useful in a variety of linear algebra operations, such as calculating eigenvalues and diagonalizing matrices.

### 7\. Dot product:

The dot product is a mathematical operation that takes two vectors and returns a scalar value. It is calculated by multiplying the corresponding entries of the two vectors and then summing the results

### 8\. Eigenvalues:

Eigenvalues are scalar values that are used to understand the properties of a matrix. They are found by solving the equation det(A - λI) = 0, where A is the matrix, λ is the eigenvalue, and I is the identity matrix. Eigenvalues are used to determine the characteristics of a matrix, such as its stability, and they are also used in matrix decompositions, such as diagonalization and principal component analysis.

### 9\. Eigenvectors:

Eigenvectors are vectors that, when multiplied by a matrix, change only in scale (not direction). They are found by solving the equation Av = λv, where A is the matrix, λ is the eigenvalue, and v is the eigenvector. Eigenvectors are used to determine the characteristics of a matrix, such as its stability, and they are also used in matrix decompositions, such as diagonalization and principal component analysis.

In summary, these concepts are important in linear algebra and are commonly used in data science and machine learning. They provide a way to understand and manipulate data and are used in various operations such as data compression, dimensionality reduction, and optimization. Understanding these concepts and being able to apply them using python is an important skill for any data scientist.

## Visualize a 2D vector using the Matplotlib library in Python:

```python
import matplotlib.pyplot as plt
import numpy as np

# Define the vector and its initial point
v = np.array([1, 2])
origin = [0], [0]

# Plot the vector as an arrow
plt.quiver(*origin, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='blue')

# Set the x and y limits
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Show the plot
plt.show()
```

![Single vector](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/zira47mfbe03k94f68fr.png align="left")

This code will create a plot with a blue arrow pointing in the direction of the vector. The arrow starts at the point (0, 0) and the x and y limits are set to (-3, 3) to give some extra space around the vector.

## Visualize multiple vectors in a single plot by plotting each vector as an arrow and setting the x and y limits accordingly.

```python
# Define the vectors and their initial points
v1 = np.array([1, 2])
origin1 = [0], [0]
v2 = np.array([3, 1])
origin2 = [0], [0]

# Plot the vectors as arrows
plt.quiver(*origin1, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue')
plt.quiver(*origin2, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red')

# Set the x and y limits
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Show the plot
plt.show()
```

![Two vectors](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/65zxi4v8m23qtipvqyym.png align="left")

This code will create a plot with two arrows, one blue and one red, starting at the point (0, 0) and pointing in the direction of the two defined vectors. The x and y limits are set to (-3, 3) to give some extra space around the vectors.

## Getting handy in matrices and its operation using python

The NumPy library in Python provides a variety of functions and methods for working with matrices, including calculating the transpose, inverse, determinant, trace, dot product, eigenvalues, and eigenvectors.

Here are some examples of how you can use NumPy to perform these operations:

* Transpose of a matrix:
    

```python
import numpy as np

# Create a matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix: \n", matrix)

# Calculate the transpose of the matrix
matrix_transpose = matrix.T
print("\nTranspose:\n", matrix_transpose)
```

Matrix: \[\[1 2 3\] \[4 5 6\] \[7 8 9\]\]

Transpose: \[\[1 4 7\] \[2 5 8\] \[3 6 9\]\]

* Inverse of a matrix:
    

```python
# Create a matrix
matrix = np.array([[1, 2], [3, 4]])
print("Matrix: \n", matrix)
# Calculate the inverse of the matrix
matrix_inverse = np.linalg.inv(matrix)
print("Matrix Inverse: \n", matrix_inverse)
```

Matrix: \[\[1 2\] \[3 4\]\] Matrix Inverse: \[\[-2. 1. \] \[ 1.5 -0.5\]\]

* Determinant of a matrix:
    

```python
# Create a matrix
matrix = np.array([[1, 2], [3, 4]])
print("Matrix: \n", matrix)
# Calculate the determinant of the matrix
matrix_determinant = np.linalg.det(matrix)
print("Matrix Determinant: \n", matrix_determinant)
```

Matrix: \[\[1 2\] \[3 4\]\] Matrix Determinant: -2.0000000000000004

* Trace of a matrix:
    

```python
# Create a matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix: \n", matrix)
# Calculate the trace of the matrix
matrix_trace = np.trace(matrix)
print("Matrix Trace: \n", matrix_trace)
```

Matrix: \[\[1 2 3\] \[4 5 6\] \[7 8 9\]\] Matrix Trace: 15

* Dot product:
    

```python
# Create two vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
print("Vector 1: ", vector1)
print("Vector 2: ", vector2)
# Calculate the dot product of the vectors
dot_product = np.dot(vector1, vector2)
print("Dot Product: ", vector1)
```

Vector 1: \[1 2 3\] Vector 2: \[4 5 6\] Dot Product: \[1 2 3\]

* Eigenvalues and Eigenvectors:
    

```python
# Create a matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Matrix: \n", matrix)
# Calculate the eigenvalues and eigenvectors of the matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("Eigen values: ", eigenvalues)
print("Eigen vectors: \n", eigenvectors)
```

Matrix: \[\[1 2 3\] \[4 5 6\] \[7 8 9\]\] Eigen values: \[ 1.61168440e+01 -1.11684397e+00 -4.22209278e-16\] Eigen vectors: \[\[-0.23197069 -0.78583024 0.40824829\] \[-0.52532209 -0.08675134 -0.81649658\] \[-0.8186735 0.61232756 0.40824829\]\]

## Summary:

The important concepts of vectors, matrices, and linear algebra in the context of data science. It explains the use and importance of these concepts in understanding and manipulating data, as well as their applications in various data science operations. It also provides examples of how to perform various linear algebraic operations using the NumPy library in Python.

## Exercise Question you will find in the exercise notebook of Day 6 on GitHub.

## If you liked it then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)