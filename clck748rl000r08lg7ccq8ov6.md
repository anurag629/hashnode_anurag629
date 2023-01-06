# Master the Basics of Python: A Step-by-Step Guide

# Day 1 of **100 Days Data Science Bootcamp from noob to expert.**

# Introduction

* Python is a general purpose high level programming language.
    
* Python was developed by Guido Van Rossam in 1989 while working at National Research Institute at Netherlands.
    
* But officially Python was made available to public in 1991.
    

### Hello World in Python

Syntax : print("Hello World")

```python
print("Hello World")
```

# IDENTIFIERS

* A Name in Python Program is called Identifier.
    
* It can be Class Name OR Function Name OR Module Name OR Variable Name.
    
* a = 10
    

### Rules to define Identifiers in Python:

* alphabet symbols(either lower case or upper case), digits(0 to 9), and underscore symbol(\_) are allowed characters in python.
    
* Identifier should not start with numbers
    
* Identifiers are case sensitive.
    
* We cannot use reserved words as identifiers.
    

#### Note:

* If identifier starts with \_ symbol then it indicates that it is private
    
* If identifier starts with \_\_(Two Under Score Symbols) indicating that strongly private identifier.
    

```python
# valid identifiers in Python:
my_variable
_private_variable
PI
myFunction

# Invalid identifiers in Python include:
123abc (starts with a digit)
my-variable (contains hyphen)
def (a reserved word in Python)
```

# DATA TYPES

* Data Type represents the type of data present inside a variable.
    
* In Python we are not required to specify the type explicitly. Based on value provided, the type will be assigned automatically. Hence Python is dynamically Typed Language.
    

### Python contains the following inbuilt data types

1) Int 2) Float 3) Complex 4) Bool 5) Str 6) List 7) Tuple 8) Set 9) Frozenset 10) Dict 11) None

* type() method is used to check the type of variable
    

### 1) int Data Type:

* to represent whole numbers (integral values)
    
* Python directly supports arbitrary precision integers, also called infinite precision integers or bignums, as a top-level construct.
    
* On a 64-bit machine, it corresponds to $2^{64 - 1}$ = 9,223,372,036,854,775,807
    

```python
# Examples of Integer data Type
a = 10
b = -14
c = 43095728934729279823748345345345345453453456345634653434363
print(type(a))
print(type(b))
print(type(c))
```

### 2) Float Data Type:

* We can use float data type to represent floating point values (decimal values)
    
* We can also represent floating point values by using exponential form Eg: f = 1.2e3 (instead of 'e' we can use 'E')
    

```python
# Examples of float data type
a = 1.9
b = 904304.0
c = 1.2e3
print(a)
print(b)
print(c)
print(type(a))
print(type(b))
print(type(c))
```

### 3) Complex Data Type:

* A complex number is of the form : (a + bj) where j = $\\sqrt{-1}$.
    
* we can perform operations on complex type values.
    

```python
# Examples of complex data type
a = 5 + 7j
b = 19 - 3j
print('a = ',a)
print('b = ',b)
print(type(a))
print(type(b))
print(f'Subtraction is : {a - b}')
print(f'Addition is : {a + b}')
# Complex data type has inbuilt attributes imag and real type
print(a.real)
print(a.imag)
```

### 4) bool Data Type:

* We can use this data type to represent boolean values.
    
* The only allowed values for this data type are: True and False
    
* Internally Python represents True as 1 and False as 0
    

```python
# Examples of bool data type
a = True
b = False
print(type(a))
print(type(b))
```

### 5) str Data Type:

* str represents String data type.
    
* A String is a sequence of characters enclosed within single quotes or double quotes.
    

```python
# Examples
a = "Lets code..."
print(a)
```

Lets code...

* By using single quotes or double quotes we cannot represent multi line string literals.
    
* For this requirement we should go for triple single quotes(''') or triple double quotes(""")
    

```python
b = "Lets
    code"
```

File "/tmp/ipykernel\_19/3804588111.py", line 1 b = "Lets ^ SyntaxError: EOL while scanning string literal

```python
b = '''Lets
    code'''
print(b)
```

#### Slicing of Strings:

1) slice means a piece 2) \[ \] operator is called slice operator, which can be used to retrieve parts of String. 3) In Python Strings follows zero based index. 4) The index can be either +ve or -ve. 5) +ve index means forward direction from Left to Right 6) -ve index means backward direction from Right to Left

```python
a = "Let's code great and change the world"
print("a[0] : ", a[0])
print("a[15] : ", a[15])
print("a[-1] : ", a[-1])
print("a[:5]", a[:5])
print("a[7:14] : ", a[7:14])

b = "Lets Code"
print(b*3)
print(len(b))
```

## TYPE CASTING

* We can convert one type value to another type. This conversion is called Typecasting.
    
* The following are various inbuilt functions for type casting. 1) int() 2) float() 3) complex() 4) bool() 5) str()
    

```python
print(int(3434.554))
```

```python
print(int(6 + 6j))
```

```python
print(str(32423))
```

```python
print(int("five"))
```

```python
print(int("2345"))
```

```python
print(float(244))
```

```python
print(bool(0))
print(bool(2324))
print(bool(-34))
print(bool(0.0))
print(bool(7.8))
print(bool("False"))
print(bool("True"))
print(bool("Lets code"))
```

### 6) List Data Type:

* Lists are used to store multiple items in a single variable.
    
* Lists are created using square brackets
    
* List items are ordered, changeable, and allow duplicate values.
    
* List items are indexed, the first item has index \[0\], the second item has index \[1\] etc.
    
* When we say that lists are ordered, it means that the items have a defined order, and that order will not change.
    
* If you add new items to a list, the new items will be placed at the end of the list.
    

```python
listt = [5, 6, 'hello', 5.76]
print(listt)
print(listt[0])
print(listt[2])

listt.append(376)

print(listt)


# Iterating over list
for i in listt:
    print(i)
```

### 7) Tuple Data Type:

* Tuples are used to store multiple items in a single variable.
    
* A tuple is a collection which is ordered and unchangeable.
    
* Tuples are written with round brackets.
    
* Tuple items are ordered, unchangeable, and allow duplicate values.
    
* Tuple items are indexed, the first item has index \[0\], the second item has index \[1\] etc.
    
* Tuples are unchangeable, meaning that we cannot change, add or remove items after the tuple has been created.
    

```python
tuple1 = ("abc", 34, True, 40, "male")
print(tuple1)
print(tuple[0])
print(tuple[3])
```

### 8) set Data Type:

* Sets are used to store multiple items in a single variable.
    
* A set is a collection which is unordered, unchangeable\*, and unindexed.
    
* Sets are written with curly brackets.
    
* Sets are unordered, so you cannot be sure in which order the items will appear.
    
* Set items are unchangeable, but you can remove items and add new items.
    
* Set items can appear in a different order every time you use them, and cannot be referred to by index or key.
    
* Sets cannot have two items with the same value.
    

```python
set1 = {"abc", 34, True, 40, "male", 40}
print(set1)
set1.add(67)
print(set1)
set1.remove("abc")
print(set1)
```

### 9) frozenset Data Type:

* It is exactly same as set except that it is immutable.
    
* Hence we cannot use add or remove functions.
    

```python
s={10,20,30,40}
frozen_set=frozenset(s)
print(type(frozen_set))
print(frozen_set)
```

### 10) dict Data Type:

* Dictionaries are used to store data values in key:value pairs.
    
* A dictionary is a collection which is ordered\*, changeable and do not allow duplicates.
    
* Dictionaries are written with curly brackets, and have keys and values.
    
* Dictionary items are ordered, changeable, and does not allow duplicates.
    
* Dictionary items are presented in key:value pairs, and can be referred to by using the key name.
    
* Dictionaries cannot have two items with the same key.
    

```python
dict1 = {
  "brand": "Ford",
  "electric": False,
  "year": 1964,
  "colors": ["red", "white", "blue"]
}

print(dict1)
print(dict1["colors"])
print(dict1["colors"][1])
```

### 11) None Data Type:

* None means nothing or No value associated.
    
* If the value is not available, then to handle such type of cases None introduced.
    

```python
def sum(a,b):
    c = a + b

s = sum(5,7)
print(s)
```

> None

# Operators

Operators in Python are special symbols that perform specific operations on one or more operands (values or variables). Operators are used to perform mathematical or logical operations on variables and values.

There are various types of operators in Python, including:

### Arithmetic operators:

These operators perform basic mathematical operations such as addition, subtraction, multiplication, and division.

1. Addition (+): The addition operator adds two operands and returns the sum.
    
2. Subtraction (-): The subtraction operator subtracts the second operand from the first and returns the difference.
    
3. Multiplication (\*): The multiplication operator multiplies two operands and returns the product.
    
4. Division (/): The division operator divides the first operand by the second and returns the quotient.
    
5. Modulus (%): The modulus operator returns the remainder of the division of the first operand by the second.
    

Example:

```python
x = 10
y = 5
z = x + y # Addition operator
print(z) # Output: 15
z1 = x - y
print(z1) # Output: 5
z2 = x * y
print(z2) # Output: 50
z3 = x / y
print(z3) # Output: 2.0
z4 = x % y
print(z4) # Output: 0
```

> 15 5 50 2.0 0

```python
x = 10
y = 5
z = (x + y) * 2 # Addition and multiplication are executed first
print(z) # Output: 30
```

> 30

### Comparison/Logical/Assignment operators:

These operators compare two operands and return a Boolean value (True or False) based on the comparison.

There are various comparison operators in Python, including:

1. Greater than (&gt;)
    
2. Less than (&lt;)
    
3. Greater than or equal to (&gt;=)
    
4. Less than or equal to (&lt;=)
    
5. Equal to (==)
    
6. Not equal to (!=)
    

Examples:

```python
x = 10
y = 5
z = x > y # Greater than operator
print(z) # Output: True
```

> True

```python
a = 10
b = 10
c = a < b # Less than operator
print(c) # Output: False
```

> False

```python
p = 5
q = 5
r = p == q # Equal to operator
print(r) # Output: True
```

> True

```python
s = 5
t = 6
u = s != t # Not equal to operator
print(u) # Output: True
```

> True

### Bitwise operators:

Bitwise operators in Python are special operators that perform bit-level operations on integers. These operators work by performing operations on the individual bits of an integer value.

There are several bitwise operators in Python, including:

1. AND (&): This operator performs a bit-level AND operation on two integers. It returns a new integer where each bit is set to 1 if both operand bits are 1, otherwise it is set to 0.
    

Example:

```python

x = 10 # Binary representation: 1010
y = 5 # Binary representation: 0101
z = x & y # Binary representation: 0000
print(z) # Output: 0
```

> 0

1. OR (|): This operator performs a bit-level OR operation on two integers. It returns a new integer where each bit is set to 1 if either operand bit is 1, otherwise it is set to 0.
    

Example:

```python
x = 10 # Binary representation: 1010
y = 5 # Binary representation: 0101
z = x | y # Binary representation: 1111
print(z) # Output: 15
```

> 15

1. XOR (^): This operator performs a bit-level XOR operation on two integers. It returns a new integer where each bit is set to 1 if one operand bit is 1 and the other operand bit is 0, otherwise it is set to 0.
    

Example:

```python
x = 10 # Binary representation: 1010
y = 5 # Binary representation: 0101
z = x ^ y # Binary representation: 1111
print(z) # Output: 15
```

> 15

1. NOT (~): This operator performs a bit-level NOT operation on an integer. It returns a new integer where each bit is set to 1 if the operand bit is 0, and vice versa.
    

Example:

```python

x = 10 # Binary representation: 1010
y = ~x # Binary representation: 0101
print(y) # Output: -11
```

> \-11

1. Left shift (&lt;&lt;): This operator shifts the bits of an integer value to the left by a specified number of places. The leftmost bits are lost and replaced with 0s.
    

Example:

```python
x = 10 # Binary representation: 1010
y = x << 2 # Binary representation: 101000
print(y) # Output: 40
```

> 40

1. Right shift (&gt;&gt;): This operator shifts the bits of an integer value to the right by a specified number of places. The rightmost bits are lost and replaced with 0s or 1s depending on the sign of the integer.
    

Example:

```python
x = 10 # Binary representation: 1010
y = x >> 2 # Binary representation: 0010
print(y) # Output: 2
```

> 2

# Exercise Question you will find in the exercise notebook of Day 1 on GitHub.

# If you like it then....;)

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)