# "Get Your Bits in Order: The Ultimate Guide to Bit Manipulation"

There are a few key topics that are important to understand when it comes to bit manipulation in competitive programming:

1. Bitwise operations: It is important to understand how to use the bitwise operators (e.g., `&`, `|`, `^`, `~`, `<<`, `>>`) to perform various operations on binary numbers.
    
2. Bitmasking: Bitmasking is a technique for setting, clearing, or checking individual bits in a number. It can be useful for representing a set of flags or for implementing certain algorithms efficiently.
    
3. Bit manipulation tricks: There are a number of "tricks" that involve using bit manipulation to solve problems in clever ways. For example, you can use the `x & (-x)` idiom to isolate the rightmost 1-bit in a number, or you can use the `x & (x - 1)` idiom to clear the rightmost 1-bit in a number.
    
4. Bit manipulation-based algorithms: There are a number of algorithms that make use of bit manipulation to solve problems efficiently. For example, you can use the divide and conquer approach to implement a fast algorithm for finding the median of a set of numbers, or you can use bit manipulation to implement a fast algorithm for multiplying two numbers.
    

Overall, it is important to have a strong foundation in the basics of bit manipulation and to be familiar with a variety of techniques and algorithms that make use of bit manipulation. Practice is also key to mastering these concepts, so it is a good idea to solve as many problems as you can that involve bit manipulation.

But don't worry here I have covered all topics in detail to "Become a Bit Manipulation Pro"

## Bitwise operations:

Bitwise operations allow you to perform operations on binary numbers at the bit level. In Python, you can use the following bitwise operators:

* `&`: Bitwise AND. This operator compares each bit of the first operand to the corresponding bit of the second operand, and if both bits are 1, the corresponding result bit is set to 1. Otherwise, the corresponding result bit is set to 0. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = 0b01010101  # The number 85 in binary
z = x & y  # The result is 0b00000000, or 0 in decimal
```

* `|`: Bitwise OR. This operator compares each bit of the first operand to the corresponding bit of the second operand, and if either bit is 1, the corresponding result bit is set to 1. Otherwise, the corresponding result bit is set to 0. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = 0b01010101  # The number 85 in binary
z = x | y  # The result is 0b11111111, or 255 in decimal
```

* `^`: Bitwise XOR. This operator compares each bit of the first operand to the corresponding bit of the second operand, and if the bits are different, the corresponding result bit is set to 1. Otherwise, the corresponding result bit is set to 0. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = 0b01010101  # The number 85 in binary
z = x ^ y  # The result is 0b11111111, or 255 in decimal
```

* `~`: Bitwise NOT. This operator flips all the bits of the operand. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = ~x  # The result is -171 in decimal
```

* `<<`: Left shift. This operator shifts the bits of the operand to the left by the number of positions specified by the second operand. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = x << 2  # The result is 0b1010101000, or 680 in decimal
```

* `>>`: Right shift. This operator shifts the bits of the operand to the right by the number of positions specified by the second operand. For example:
    

```python
x = 0b10101010  # The number 170 in binary
y = x >> 2  # The result is 0b00101010, or 42 in decimal
```

## Bitmasking:

Bitmasking is a technique for setting, clearing, or checking individual bits in a number. It involves using bitwise operators to manipulate the bits of a number in order to set or clear specific bits or to check the values of specific bits.

For example, suppose you have a number `x` and you want to set the third and fifth bits (counting from the right) to 1. You can do this using the following code:

```python
x = 0b00000000  # The number 0 in binary
mask = 0b00101000  # The mask with the third and fifth bits set to 1
x = x | mask  # The result is 0b00101000, or 40 in decimal
```

Alternatively, suppose you have a number `x` and you want to clear the fourth and sixth bits (counting from the right). You can do this using the following code:

```python
x = 0b11111111  # The number 255 in binary
mask = 0b11010111  # The mask with the fourth and sixth bits set to 0
x = x & mask  # The result is 0b11010111, or 215 in decimal
```

Bitmasking can also be used to check the values of specific bits in a number. For example, suppose you have a number `x` and you want to check if the seventh bit (counting from the right) is set to 1. You can do this using the following code:

```python
x = 0b11010111  # The number 215 in binary
mask = 0b10000000  # The mask with the seventh bit set to 1
if x & mask:
    print("The seventh bit is set to 1")
else:
    print("The seventh bit is not set to 1")
```

Bitmasking is often used in programming to represent a set of flags or to implement certain algorithms efficiently. For example, you might use bitmasking to represent a set of permissions (e.g., read, write, execute) or to implement a fast algorithm for finding the intersection of two sets.

Suppose you want to represent a set of flags that indicate whether a user has read, write, and execute permissions for a particular file. You could define the following constants:

```python
READ_PERMISSION = 0b001
WRITE_PERMISSION = 0b010
EXECUTE_PERMISSION = 0b100
```

These constants represent the flags for each of the permissions, with the rightmost bit representing the read permission, the second rightmost bit representing the write permission, and the third rightmost bit representing the execute permission.

To represent a set of permissions for a particular user, you could use a variable of type `int` and set the appropriate bits using bitmasking. For example:

```python
permissions = 0b000  # No permissions

# Grant the user read permission
permissions = permissions | READ_PERMISSION  # The result is 0b001

# Grant the user write permission
permissions = permissions | WRITE_PERMISSION  # The result is 0b011

# Grant the user execute permission
permissions = permissions | EXECUTE_PERMISSION  # The result is 0b111
```

To check whether a user has a particular permission, you can use bitmasking to check the value of the corresponding bit. For example:

```python
if permissions & READ_PERMISSION:
    print("The user has read permission")

if permissions & WRITE_PERMISSION:
    print("The user has write permission")

if permissions & EXECUTE_PERMISSION:
    print("The user has execute permission")
```

## Bit manipulation tricks:

Bit manipulation tricks are techniques that involve using bit manipulation to solve problems in clever ways. Here are a few examples of common bit manipulation tricks:

* Isolating the rightmost 1-bit: You can use the `x & (-x)` idiom to isolate the rightmost 1-bit in a number. For example:
    

```python
x = 0b01010101  # The number 85 in binary
y = x & (-x)  # The result is 0b00000001, or 1 in decimal
```

* Clearing the rightmost 1-bit: You can use the `x & (x - 1)` idiom to clear the rightmost 1-bit in a number. For example:
    

```python
x = 0b01010101  # The number 85 in binary
y = x & (x - 1)  # The result is 0b01010100, or 84 in decimal
```

* Checking if a number is a power of 2: You can use the `x & (x - 1) == 0` idiom to check if a number is a power of 2. For example:
    

```python
x = 8  # The number 8 is a power of 2
if x & (x - 1) == 0:
    print("x is a power of 2")

x = 9  # The number 9 is not a power of 2
if x & (x - 1) == 0:
    print("x is a power of 2")  # This line will not be executed
```

* Counting the number of 1-bits: You can use the `x &= (x - 1)` idiom to count the number of 1-bits in a number. For example:
    

```python
x = 0b01010101  # The number 85 in binary
count = 0
while x:
    count += 1
    x &= (x - 1)
print(count)  # The output is 4
```

* Swapping the values of two variables: You can use bit manipulation to swap the values of two variables without using a temporary variable. For example:
    

```python
x = 10
y = 20

# Swap the values of x and y
x ^= y
y ^= x
x ^= y

print(x)  # The output is 20
print(y)  # The output is 10
```

* Checking if a number is odd or even: You can use the `x & 1` idiom to check if a number is odd or even. For example:
    

```python
x = 10  # The number 10 is even
if x & 1:
    print("x is odd")
else:
    print("x is even")

x = 11  # The number 11 is odd
if x & 1:
    print("x is odd")
else:
    print("x is even")
```

* Determining the sign of a number: You can use the `x >> 31` idiom to determine the sign of a number. For example:
    

```python
x = 10  # The number 10 is positive
if x >> 31:
    print("x is negative")
else:
    print("x is positive")

x = -10  # The number -10 is negative
if x >> 31:
    print("x is negative")
else:
    print("x is positive")
```

* Calculating the absolute value of a number: You can use the `(x + (x >> 31)) ^ (x >> 31)` idiom to calculate the absolute value of a number. For example:
    

```python
x = 10  # The absolute value of 10 is 10
y = (x + (x >> 31)) ^ (x >> 31)
print(y)  # The output is 10

x = -10  # The absolute value of -10 is 10
y = (x + (x >> 31)) ^ (x >> 31)
print(y)  # The output is 10
```

* Reversing the bits of a number: You can use bit manipulation to reverse the bits of a number. For example:
    

```python
x = 0b01010101  # The number 85 in binary
y = 0

# Reverse the bits of x
for i in range(8):
    y = (y << 1) | (x & 1)
    x >>= 1

print(y)  # The output is 0b10101010, or 170 in decimal
```

* Calculating the logarithm of a number: You can use bit manipulation to calculate the logarithm of a number. For example:
    

```python
x = 256  # The logarithm of 256 is 8
y = 0
while x > 1:
    y += 1
    x >>= 1
print(y)  # The output is 8
```

* Calculating the square root of a number: You can use bit manipulation to calculate the square root of a number. For example:
    

```python
x = 256  # The square root of 256 is 16
y = 0
while x > 0:
    y += 1
    x >>= 2
print(y)  # The output is 16
```

* Calculating the greatest common divisor of two numbers: You can use bit manipulation to calculate the greatest common divisor (GCD) of two numbers. For example:
    

```python
def gcd(x, y):
    while y > 0:
        x, y = y, x & y
    return
```

* Calculating the least common multiple of two numbers: You can use bit manipulation to calculate the least common multiple (LCM) of two numbers. For example:
    

```python
def lcm(x, y):
    return (x * y) // gcd(x, y)  # gcd() is the function that calculates the GCD

x = 10
y = 15
z = lcm(x, y)  # The result is 30
```

* Calculating the parity of a number: You can use bit manipulation to calculate the parity of a number (i.e., whether the number has an even or odd number of 1-bits). For example:
    

```python
x = 0b01010101  # The number 85 in binary
y = 0

# Calculate the parity of x
while x:
    y ^= 1
    x &= (x - 1)

if y:
    print("The number has an odd number of 1-bits")
else:
    print("The number has an even number of 1-bits")
```

* Calculating the Hamming distance between two numbers: You can use bit manipulation to calculate the Hamming distance between two numbers (i.e., the number of bit positions in which the numbers differ). For example:
    

```python
x = 0b01010101  # The number 85 in binary
y = 0b10101010  # The number 170 in binary
z = 0

# Calculate the Hamming distance between x and y
while x or y:
    z += (x & 1) ^ (y & 1)
    x >>= 1
    y >>= 1

print(z)  # The output is 8
```

## Bit manipulation-based algorithms:

Certainly! Bit manipulation can be used to implement a variety of algorithms efficiently. Here are a few examples of algorithms that can be implemented using bit manipulation:

* Bitonic sort: Bitonic sort is an efficient sorting algorithm that can be implemented using bit manipulation. It works by dividing the input array into pairs of elements and comparing them using bit manipulation. The algorithm then recursively sorts the resulting arrays using the same process.
    
* Bitwise trie: A bitwise trie is a data structure that can be used to store and retrieve data efficiently. It works by using bit manipulation to store the data in a tree-like structure, with each bit of the data serving as a branch in the tree.
    
* Bitmap: A bitmap is a data structure that can be used to store a large number of boolean values efficiently. It works by using bit manipulation to store the values in an array of bits, with each bit representing a boolean value.
    
* Hash table: A hash table is a data structure that can be used to store and retrieve data efficiently. It works by using bit manipulation to compute the hash value of the data, which is then used to index into an array of buckets.
    

These are just a few examples of algorithms that can be implemented using bit manipulation. There are many other algorithms that can be implemented efficiently using bit manipulation, including algorithms for searching, sorting, and data compression. But are example code so you all can understand effectively.

**Bitonic sort:**

```python
def bitonic_sort(arr):
    # Base case: return the array if it has 1 or 0 elements
    if len(arr) <= 1:
        return arr

    # Divide the array into two halves
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    # Recursively sort the two halves
    left = bitonic_sort(left)
    right = bitonic_sort(right)

    # Compare the elements in the two halves using bit manipulation
    for i in range(mid):
        if (left[i] > right[i]) == (i & 1):
            left[i], right[i] = right[i], left[i]

    # Concatenate the sorted halves and return the result
    return left + right

arr = [3, 7, 4, 8, 6, 2, 1, 5]
sorted_arr = bitonic_sort(arr)
print(sorted_arr)  # The output is [1, 2, 3, 4, 5, 6, 7, 8]
```

**Bitwise trie:**

```python
class BitwiseTrie:
    def __init__(self):
        self.children = [None, None]

    def insert(self, num):
        node = self
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if not node.children[bit]:
                node.children[bit] = BitwiseTrie()
            node = node.children[bit]

trie = BitwiseTrie()
trie.insert(85)  # Insert the number 85 (0b01010101) into the trie
trie.insert(170)  # Insert the number 170 (0b10101010) into the trie
```

**Bitmap:**

```python
class Bitmap:
    def __init__(self, size):
        self.bits = [0] * ((size + 31) // 32)

    def set_bit(self, pos):
        self.bits[pos // 32] |= (1 << (pos % 32))

    def clear_bit(self, pos):
        self.bits[pos // 32] &= ~(1 << (pos % 32))

    def get_bit(self, pos):
        return self.bits[pos // 32] & (1 << (pos % 32))

bitmap = Bitmap(100)  # Create a bitmap with 100 bits
bitmap.set_bit(10)  # Set the 11th bit (counting from the right) to 1
bitmap.clear_bit(10)  # Clear the 11th bit (counting from the right)
if bitmap.get_bit(10):
    print("The 11th bit (counting from the right) is set to 1")
else:
    print("The 11th bit (counting from the right) is not set to 1")
```

**Hash table:**

```python
class HashTable:
    def __init__(self, size):
        self.size = size
        self.buckets = [None] * size

    def hash_function(self, key):
        # Compute the hash value using bit manipulation
        hash_value = 0
        for c in key:
            hash_value = (hash_value << 5) + hash_value + ord(c)
        return hash_value % self.size

    def insert(self, key, value):
        # Compute the hash value of the key
        hash_value = self.hash_function(key)

        # Insert the key-value pair into the appropriate bucket
        self.buckets[hash_value] = (key, value)

    def lookup(self, key):
        # Compute the hash value of the key
        hash_value = self.hash_function(key)

        # Look up the key-value pair in the appropriate bucket
        return self.buckets[hash_value]

hash_table = HashTable(100)  # Create a hash table with 100 buckets
hash_table.insert("apple", "red")  # Insert the key-value pair "apple"-"red"
hash_table.insert("banana", "yellow")  # Insert the key-value pair "banana"-"yellow"
print(hash_table.lookup("apple"))  # The output is ("apple", "red")
print(hash_table.lookup("banana"))  # The output is ("banana", "yellow")
```

"In conclusion, bit manipulation is a powerful tool that can help programmers solve problems efficiently and effectively. Whether you are competing in programming contests or working on real-world projects, a strong understanding of bit manipulation can give you a competitive edge.

To master bit manipulation, it is important to practice using it to solve problems. You can find many online resources, such as online judges and programming contests, that offer challenges that can help you hone your skills. You can also create your own problems to solve, or explore more advanced techniques such as bit manipulation-based algorithms and data structures.

There are many resources available for learning about bit manipulation, including books, online tutorials, and blogs. Take the time to explore these resources and continue learning about this important topic. With dedication and practice, you can become a bit manipulation pro and unlock the full potential of this powerful tool.

## <mark>I hope you liked this post...sharing love and knowledge...</mark>

# Happy coding...