# Take Your Python Skills to the Next Level with Built-in Data Structures

# Day 2 of 100 Days Data Science Bootcamp from noob to expert.

# GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

# Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 1

Yesterday we have studied in detail about basics of python.

## What we will study in this...

In this lesson, we will delve deeper into the various inbuilt data structures in Python and explore how they can be used in programs through detailed examples and exercises. We will cover data structures such as lists, tuples, sets, and dictionaries, and discuss their unique characteristics and uses. By the end of this lesson, you will have a solid understanding of these data structures and be able to confidently utilize them in your Python programs.

Python has several inbuilt data data structure or data types, including:

1. Integer (int) - used to represent whole numbers
    
2. Float - used to represent floating point values
    
3. Complex - used to represent complex numbers
    
4. Boolean (bool) - used to represent boolean values (True or False)
    
5. String (str) - used to represent sequences of characters
    
6. List - used to represent ordered sequences of elements
    
7. Tuple - used to represent ordered sequences of elements that cannot be modified
    
8. Set - used to represent unordered collections of unique elements
    
9. Frozen set - used to represent unordered collections of unique elements that cannot be modified
    
10. Dictionary - used to represent key-value pairs
    
11. None - used to represent the absence of a value.
    

**On day 1, we briefly discussed the basics of these topics, but now we will delve into them in more detail on** `List`, `Tuple`, `Set` and `Dictionary`.

## List

### Basics:

A list is a collection of items that are ordered and changeable. Lists are written with square brackets and the items are separated by commas.

```python
my_list1 = [1, 2, 3, 4] # this is a list with four integers
my_list2 = ['apple', 'banana', 'cherry'] # this is a list with three strings
my_list3 = [1, 'apple', 3.14, True] # this is a list with four items of different data types

print(my_list1)
print(my_list2)
print(my_list3)
```

There are several ways to create or declare a list in Python:

* Using square brackets: You can create a list by enclosing a comma-separated list of items in square brackets.
    

```python
my_list1 = [1, 2, 3, 4, 5] # this is a list of integers
my_list2 = ['apple', 'banana', 'cherry'] # this is a list of strings
my_list3 = [1, 'apple', 3.14, True] # this is a list of items with different data types
print(my_list1)
print(my_list2)
print(my_list3)
```

* Using the list() function: You can create a list by passing a sequence to the list() function.
    

```python
my_list1 = list(range(10)) # creates a list of integers from 0 to 9
my_list2 = list('abcdefg') # creates a list of characters from the string
print(my_list1)
print(my_list1)
```

* Using list comprehension: You can create a list using a single line of code with list comprehension.
    

```python
my_list1 = [x for x in range(10)] # creates a list of integers from 0 to 9
my_list2 = [x for x in 'abcdefg'] # creates a list of characters from the string
print(my_list1)
print(my_list1)
```

* Using the \* operator: You can create a list by duplicating an existing list using the \* operator.
    

```python
my_list1 = [1, 2, 3] * 3 # creates a list with three copies of [1, 2, 3]
my_list2 = [1] * 10 # creates a list with ten copies of the integer 1
my_list3 = ['hello'] * 5 # creates a list with five copies of the string 'hello'
print(my_list1)
print(my_list2)
print(my_list3)
```

### Accessing Items:

You can access the items in a list using their index. The index starts at 0 for the first item and goes up by 1 for each subsequent item.

```python
my_list = [1, 2, 3, 4]
print(my_list[0]) # prints 1 (the first item in the list)
print(my_list[2]) # prints 3 (the third item in the list)
```

### Changing Items:

You can change the value of an item in a list by assigning a new value to its index.

```python
my_list = [1, 2, 3, 4]
my_list[0] = 5 # changes the value of the first item in the list to 5
print(my_list) # prints [5, 2, 3, 4]
```

### Adding Items:

You can add items to a list using the `append()` method or the `insert()` method.

The `append()` method adds the item to the end of the list.

```python
my_list = [1, 2, 3, 4]
my_list.append(5) # adds 5 to the end of the list
print(my_list) # prints [1, 2, 3, 4, 5]
```

The `insert()` method adds the item at the specified index.

```python
my_list = [1, 2, 3, 4]
my_list.insert(1, 5) # adds 5 at index 1 (between 1 and 2)
print(my_list) # prints [1, 5, 2, 3, 4]
```

### Removing Items:

You can remove items from a list using the `remove()` method or the `pop()` method.

The `remove()` method removes the first occurrence of the item.

```python
my_list = [1, 2, 3, 4, 5, 5]
my_list.remove(5) # removes the first occurrence of 5
print(my_list) # prints [1, 2, 3, 4, 5]
```

The `pop()` method removes the item at the specified index and returns the item.

```python
my_list = [1, 2, 3, 4]
item = my_list.pop(1) # removes the item at index 1 (2) and assigns it to the variable "item"
print(my_list) # prints [1, 3, 4]
print(item) # prints 2
```

### Finding the Length of a List:

You can find the length of a list using the len() function.

```python
my_list = [1, 2, 3, 4]
print(len(my_list)) # prints 4
```

### Advanced:

Here are some more advanced features of lists in Python:

### Slicing:

You can slice a list to access a portion of it by specifying a range of indexes.

```python
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# access the first three items in the list
print(my_list[0:3]) # prints [1, 2, 3]

# access the middle four items in the list
print(my_list[3:7]) # prints [4, 5, 6, 7]

# access the last three items in the list
print(my_list[-3:]) # prints [8, 9, 10]

# access all items in the list
print(my_list[:]) # prints [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

### Sorting:

You can sort a list using the sorted() function or the sort() method.

The sorted() function returns a new sorted list, while the \`sort()

```python
my_list = [3, 5, 1, 4, 2]

#sort the list using the sorted() function
sorted_list = sorted(my_list)
print(sorted_list) # prints [1, 2, 3, 4, 5]
print(my_list) # prints [3, 5, 1, 4, 2]

#sort the list using the sort() method
my_list.sort()
print(my_list) # prints [1, 2, 3, 4, 5]
```

### Reverse:

You can reverse a list using the reverse() method.

```python
my_list = [1, 2, 3, 4, 5]

# reverse the list using the reverse() method
my_list.reverse()
print(my_list) # prints [5, 4, 3, 2, 1]
```

Reverse using slicing

```python
my_list = [1, 2, 3, 4, 5]

# reverse using slicing
print(my_list[::-1])
```

### List Comprehension:

List comprehension is a concise way to create a list using a single line of code.

```python
# create a list of squares of the numbers 0 to 9
squares = [x**2 for x in range(10)]
print(squares) # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# create a list of even numbers between 10 and 20
even_numbers = [x for x in range(10, 21) if x % 2 == 0]
print(even_numbers) # prints [10, 12, 14, 16, 18, 20]
```

### Loops:

You can loop through the items in a list using a for loop.

```python
# loop through the items in a list and print each one
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# loop through the items in a list and print their index and value
my_list = ['apple', 'banana', 'cherry']
for i, item in enumerate(my_list):
    print(i, item)

# output:
# 0 apple
# 1 banana
# 2 cherry
```

### Enumerate:

The enumerate() function returns a tuple with the index and value of each item in the list.

```python
my_list = ['apple', 'banana', 'cherry']
for i, item in enumerate(my_list):
    print(i, item)
```

### Filtering:

You can filter a list to only include items that meet certain criteria using a list comprehension with a condition.

```python
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# filter the list to only include even numbers
even_numbers = [x for x in my_list if x % 2 == 0]
print(even_numbers) # prints [2, 4, 6, 8, 10]

# filter the list to only include numbers greater than 5
greater_than_five = [x for x in my_list if x > 5]
print(greater_than_five) # prints [6, 7, 8, 9, 10]
```

### Mapping:

You can apply a function to each item in a list using a list comprehension.

```python
my_list = [1, 2, 3, 4, 5]

# multiply each item in the list by 2
doubled_list = [x * 2 for x in my_list]
print(doubled_list) # prints [2, 4, 6, 8, 10]

# convert each item in the list to a string
string_list = [str(x) for x in my_list]
print(string_list) # prints ['1', '2', '3', '4', '5']
```

### Max and Min:

You can find the maximum and minimum value in a list using the max() and min() functions.

```python
my_list = [1, 2, 3, 4, 5]

# find the maximum value in the list
max_value = max(my_list)
print(max_value) # prints 5

# find the minimum value in the list
min_value = min(my_list)
print(min_value) # prints 1
```

### Sum:

You can find the sum of all the values in a list using the sum() function.

```python
my_list = [1, 2, 3, 4, 5]

# find the sum of all the values in the list
total = sum(my_list)
print(total) # prints 15
```

### Multidimensional Lists:

Lists can also contain other lists, creating a multidimensional list.

```python
# create a 2D list with 3 rows and 4 columns
my_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

# access the first item in the first row
print(my_list[0][0]) # prints 1

# access the second item in the second row
print(my_list[1][1]) # prints 6

# access the fourth item in the third row
print(my_list[2][3]) # prints 12
```

### Conclusion:

Lists are a powerful and versatile data structure in Python. They allow you to store and manipulate a collection of items in an ordered manner. You can access, change, add, remove, and manipulate items in a list using various methods and functions. You can also use list comprehension, loops, and other advanced techniques to work with lists more efficiently.

## Tuple

A tuple is an immutable sequence type in Python. It is similar to a list in that it can contain multiple values, but unlike a list, the values in a tuple cannot be modified once created. This makes tuples more efficient for storing and manipulating data that does not need to be modified.

Here is an example of how to create a tuple

```python
my_tuple = (1, 2, 3)
print(my_tuple)
```

You can also create a tuple with a single element by including a comma after the element:

```python
my_tuple = (1,)
print(my_tuple)
```

Without the comma, Python will treat the parentheses as parentheses and not as the syntax for a tuple:

```python
my_tuple = (1)
print(my_tuple)
```

You can access the elements of a tuple using indexing, just like you would with a list.

For example:

```python
my_tuple = (1, 2, 3)
print(my_tuple[0])
print(my_tuple[1])
print(my_tuple[2])
```

You can also use slicing to access a range of elements in a tuple:

```python
my_tuple = (1, 2, 3, 4, 5)
print(my_tuple[1:3])
```

Tuples also support all of the common sequence operations, such as concatenation, repetition, and membership testing:

```python
my_tuple = (1, 2, 3)
print(my_tuple * 3) #Output: (1, 2, 3, 1, 2, 3, 1, 2, 3)
print(my_tuple + (4, 5, 6)) #Output: (1, 2, 3, 4, 5, 6)
print(3 in my_tuple) #Output: True
print(4 in my_tuple) #Output: False
```

Tuples are often used to store related pieces of data, such as the name and age of a person:

```python
person = ("John", 30)
name, age = person
print(name) #Output: "John"
print(age) # Output: 30
```

## Sets

A set is a collection of unique elements in Python. It is similar to a list or tuple, but it is unordered and does not allow duplicate values.

Here is an example of how to create a set in Python:

```python
# create an empty set
my_set = set()
print(my_set)

# create a set with values
my_set = {1, 2, 3, 4}
print(my_set)

# create a set from a list
my_list = [1, 2, 3, 4, 2]
my_set = set(my_list)  # {1, 2, 3, 4}
print(my_set)
```

Sets can be modified using various methods such as add(), update(), remove(), and discard().

```python
# create a set with values
my_set = {1, 2, 3, 4}
print(my_set)

# add an element to the set
my_set.add(5)
print(my_set)

# add multiple elements to the set
my_set.update([6, 7, 8])
print(my_set)

# remove an element from the set
my_set.remove(5)
print(my_set)

# remove an element from the set if it exists, otherwise do nothing
my_set.discard(5)
print(my_set)

# clear all elements from the set
my_set.clear()
print(my_set)
```

Sets can also be used to perform set operations such as union, intersection, and difference.

```python
set1 = {1, 2, 3}
set2 = {2, 3, 4}

# find the union of two sets
union = set1.union(set2)  # {1, 2, 3, 4}
print(union)

# find the intersection of two sets
intersection = set1.intersection(set2)  # {2, 3}
print(intersection)

# find the difference between two sets
difference = set1.difference(set2)  # {1}
print(difference)
```

Overall, sets are useful for storing and manipulating unique values in Python.

They are especially useful for performing set operations, such as finding the intersection or difference between two sets.

Here is an example of how to use sets to find the common elements between two lists:

```python
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]

# convert the lists to sets
set1 = set(list1)
set2 = set(list2)

# find the intersection of the sets
common = set1.intersection(set2)  # {4, 5}

# convert the intersection back to a list
common_list = list(common)

print(common_list)  # [4, 5]
```

Sets are also useful for removing duplicates from a list. Here is an example of how to do this:

```python
my_list = [1, 2, 3, 4, 4, 5, 5, 6, 6]

# convert the list to a set to remove duplicates
my_set = set(my_list)

# convert the set back to a list
unique_list = list(my_set)

print(unique_list)  # [1, 2, 3, 4, 5, 6]
```

### Conclusion

Sets are a powerful data structure in Python and have many uses. Some additional things to know about sets include:

* Sets are unordered, meaning that the elements are not stored in a specific order. This means that you cannot access elements in a set by index like you can with a list or tuple.
    
* Sets are mutable, meaning that you can add or remove elements from the set after it has been created.
    
* Sets are not indexed, meaning that you cannot reference elements in a set using an index like you can with a list or tuple.
    
* Sets are not sliceable, meaning that you cannot use the slice operator (\[:\]) to extract a portion of a set like you can with a list or tuple.
    
* Sets are not subscriptable, meaning that you cannot use the subscript operator (\[\]) to access elements in a set like you can with a list or tuple.
    
* Sets do not support concatenation, meaning that you cannot use the + operator to combine two sets like you can with lists or tuples.
    
* Sets do not support repetition, meaning that you cannot use the \* operator to repeat a set like you can with a list or tuple.
    

As you can see, sets have some limitations compared to other data structures in Python. However, they are still a useful tool to have in your toolkit, especially when working with unique values or performing set operations.

## Dictionary

A dictionary is a collection of key-value pairs in Python. It is similar to a list or tuple, but instead of using an index to access elements, you use a key.

Here is an example of how to create a dictionary in Python:

```python
# create an empty dictionary
my_dict = {}

# create a dictionary with values
my_dict = {'key1': 'value1', 'key2': 'value2'}
print(my_dict)

# create a dictionary from a list of tuples
my_list = [('key1', 'value1'), ('key2', 'value2')]
my_dict = dict(my_list)
print(my_dict)
```

Dictionaries can be modified using various methods such as update(), setdefault(), and pop().

```python
# create a dictionary with values
my_dict = {'key1': 'value1', 'key2': 'value2'}
print(my_dict)

# add a key-value pair to the dictionary
my_dict['key3'] = 'value3'
print(my_dict)

# update multiple key-value pairs in the dictionary
my_dict.update({'key4': 'value4', 'key5': 'value5'})
print(my_dict)

# set a default value for a key if it does not exist
my_dict.setdefault('key6', 'default value')
print(my_dict)

# remove a key-value pair from the dictionary
my_dict.pop('key3')
print(my_dict)
```

Dictionaries can also be used to perform dictionary operations such as merging and filtering.

```python
dict1 = {'key1': 'value1', 'key2': 'value2'}
dict2 = {'key3': 'value3', 'key4': 'value4'}

# merge two dictionaries
merged_dict = {**dict1, **dict2}  # {'key1': 'value1', 'key2': 'value2', 'key3': 'value3', 'key4': 'value4'}
print(merged_dict)

# filter a dictionary based on a condition
filtered_dict = {k: v for k, v in merged_dict.items() if v == 'value2'}  # {'key2': 'value2'}
print(filtered_dict)
```

Here is an example of how to use a dictionary to store and retrieve student grades:

```python
student_grades = {
    'John': {'math': 85, 'english': 90},
    'Mary': {'math': 95, 'english': 80},
    'Bob': {'math': 75, 'english': 70}
}
print(student_grades)

# retrieve John's math grade
john_math_grade = student_grades['John']['math']  # 85
print(john_math_grade)

# update Mary's english grade
student_grades['Mary']['english'] = 95
print(student_grades)
```

### Conclusion

Dictionaries are a powerful data structure in Python and have many uses. Some additional things to know about dictionaries include:

* Dictionaries are mutable, meaning that you can add or remove key-value pairs from the dictionary after it has been created.
    
* Dictionaries are unordered, meaning that the key-value pairs are not stored in a specific order.
    
* Dictionaries do not support slicing, meaning that you cannot use the slice operator (\[:\]) to extract a portion of a dictionary like you can with a list or tuple.
    
* Dictionaries do not support concatenation, meaning that you cannot use the + operator to combine two dictionaries like you can with lists or tuples.
    
* Dictionaries do not support repetition, meaning that you cannot use the \* operator to repeat a dictionary like you can with a list or tuple.
    

## Function in Python

In Python, a function is a block of code that performs a specific task and can be called by other code. Functions can take arguments (also known as parameters) and return a result.

Here is an example of a simple function in Python:

```python
def greet(name):
    print("Hello, " + name)

greet("John")  # prints "Hello, John"
```

In this example, the function greet takes one argument, name, and prints a greeting with it. The function is called with the argument "John", so it prints "Hello, John".

Functions can also return a value instead of printing it. For example:

```python
def add(x, y):
    return x + y

result = add(3, 4)  # stores 7 in result
print(result)  # prints 7
```

In this example, the function add takes two arguments, x and y, and returns their sum. When the function is called with the arguments 3 and 4, it returns 7, which is then stored in the variable result and printed.

Functions can have default values for their arguments. For example:

```python
def greet(name, greeting="Hello"):
    print(greeting + ", " + name)

greet("John")  # prints "Hello, John"
greet("John", "Hi")  # prints "Hi, John"
```

In this example, the function greet has a default value of "Hello" for the greeting argument, so if no value is provided for greeting when the function is called, it will use "Hello" as the default.

Functions can take an arbitrary number of arguments using the \*args syntax. For example:

```python
def sum_all(*args):
    result = 0
    for num in args:
        result += num
    return result

print(sum_all(1, 2, 3))  # prints 6
print(sum_all(1, 2, 3, 4, 5))  # prints 15
```

In this example, the function sum\_all takes an arbitrary number of arguments and returns their sum. The arguments are treated as a tuple, so you can access them like any other tuple.

Functions can also take an arbitrary number of keyword arguments using the \*\*kwargs syntax. For example:

```python
def print_keyword_args(**kwargs):
    for key, value in kwargs.items():
        print(key + ": " + value)

print_keyword_args(name="John", age='30', city="New York")
```

In this example, the function print\_keyword\_args takes an arbitrary number of keyword arguments and prints them. The keyword arguments are treated as a dictionary, so you can access them like any other dictionary.

Functions can return multiple values using tuples. For example:

```python
def min_max(numbers):
    return (min(numbers), max(numbers))

(min_val, max_val) = min_max([1, 2, 3, 4, 5])
print(min_val)  # prints 1
print(max_val)  # prints 5
```

In this example, the function min\_max returns a tuple containing the minimum and maximum values from the input list. The tuple is then unpacked into the variables min\_val and max\_val, which are printed.

# Exercise Question you will find in the exercise notebook of Day 2 on GitHub.

# If you liked it then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)