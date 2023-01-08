# OOP in Python: A Practical Guide

# Day 3 of 100 Days Data Science Bootcamp from noob to expert.

# GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

# Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 2

Yesterday we have studied in detail about python buit it data struccture.

Object-Oriented Programming (OOP) is a programming paradigm that is based on the concept of "objects", which can contain data and code that manipulate that data.

In OOP, a "class" is a template for creating objects. It defines the properties (data) and methods (functions) that an object can have. An object is an instance of a class, created from the class template.

Here is an example of a simple class in Python:

## Major Python OOPs concept-

In this section, we will deep dive into the basic concepts of OOP. We will cover the following topics-

1. Class
    
2. Object
    
3. Method
    
4. Inheritance
    
5. Encapsulation
    
6. Polymorphism
    
7. Data Abstraction
    

### Class

A straight forward answer to this question is- A class is a collection of objects. Unlike the primitive data structures, classes are data structures that the user defines. They make the code more manageable.

```python
class Car:
    # class body
    pass
```

### Object

When we define a class only the description or a blueprint of the object is created. There is no memory allocation until we create its object. The objector instance contains real data or information.

Instantiation is nothing but creating a new object/instance of a class. Let’s create the object of the above class we defined-

```python
obj1 = Car()
```

```python
print(obj1)
```

&lt;[**main**.Car](http://main.Car) object at 0x7f50d9fc6280&gt;

Since our class was empty, it returns the address where the object is stored i.e 0x7fc5e677b6d8

You also need to understand the class constructor before moving forward.

**Class constructor**

Until now we have an empty class Car, time to fill up our class with the properties of the car. The job of the class constructor is to assign the values to the data members of the class when an object of the class is created.

There can be various properties of a car such as its name, color, model, brand name, engine power, weight, price, etc. We’ll choose only a few for understanding purposes.

```python
class Car:
    def __init__(self, name, color):
        self.name = name
        self.color = color
```

So, the properties of the car or any other object must be inside a method that we call **init**( ). This **init**() method is also known as the constructor method. We call a constructor method whenever an object of the class is constructed.

Now let’s talk about the parameter of the **init**() method. So, the first parameter of this method has to be self. Then only will the rest of the parameters come.

The two statements inside the constructor method are –

1. [self.name](http://self.name) = name
    
2. self.color = color:
    

This will create new attributes namely name and color and then assign the value of the respective parameters to them. The “self” keyword represents the instance of the class. By using the “self” keyword we can access the attributes and methods of the class. It is useful in method definitions and in variable initialization. The “self” is explicitly used every time we define a method.

Note: You can create attributes outside of this **init**() method also. But those attributes will be universal to the whole class and you will have to assign the value to them.

Suppose all the cars in your showroom are Sedan and instead of specifying it again and again you can fix the value of car\_type as Sedan by creating an attribute outside the **init**().

```python
class Car:
    car_type = "Sedan"                 #class attribute
    def __init__(self, name, color):
        self.name = name               #instance attribute   
        self.color = color             #instance attribute
```

Here, Instance attributes refer to the attributes inside the constructor method i.e [self.name](http://self.name) and self.color. And, Class attributes refer to the attributes outside the constructor method i.e car\_type.

### Methods

So far we’ve added the properties of the car. Now it’s time to add some behavior. Methods are the functions that we use to describe the behavior of the objects. They are also defined inside a class. Look at the following code-

```python
class Car:   
    car_type = "Sedan" 

    def __init__(self, name, mileage):
        self.name = name 
        self.mileage = mileage 

    def description(self):                 
        return f"The {self.name} car gives the mileage of {self.mileage}km/l"

    def max_speed(self, speed):
        return f"The {self.name} runs at the maximum speed of {speed}km/hr"

obj2 = Car("Honda City",24.1)
print(obj2.description())
print(obj2.max_speed(150))
```

The Honda City car gives the mileage of 24.1km/l The Honda City runs at the maximum speed of 150km/hr

The methods defined inside a class other than the constructor method are known as the instance methods. Furthermore, we have two instance methods here- description() and max\_speed(). Let’s talk about them individually-

* description()- This method is returning a string with the description of the car such as the name and its mileage. This method has no additional parameter. This method is using the instance attributes.
    
* max\_speed()- This method has one additional parameter and returning a string displaying the car name and its speed.
    

Notice that the additional parameter speed is not using the “self” keyword. Since speed is not an instance variable, we don’t use the self keyword as its prefix. Let’s create an object for the class described above.

```python
obj2 = Car("Honda City",24.1)
print(obj2.description())
print(obj2.max_speed(150))
```

The Honda City car gives the mileage of 24.1km/l The Honda City runs at the maximum speed of 150km/hr

What we did is we created an object of class car and passed the required arguments. In order to access the instance methods we use object\_name.method\_name().

The method description() didn’t have any additional parameter so we did not pass any argument while calling it.

The method max\_speed() has one additional parameter so we passed one argument while calling it.

Note: Three important things to remember are-

1. You can create any number of objects of a class.
    
2. If the method requires n parameters and you do not pass the same number of arguments then an error will occur.
    
3. Order of the arguments matters.
    

Let’s look at this one by one-

### 1\. Creating more than one object of a class

```python
class Car:
    def __init__(self, name, mileage):
        self.name = name 
        self.mileage = mileage 

    def max_speed(self, speed):
        return f"The {self.name} runs at the maximum speed of {speed}km/hr"
```

```python
Honda = Car("Honda City",21.4)
print(Honda.max_speed(150))

Skoda = Car("Skoda Octavia",13)
print(Skoda.max_speed(210))
```

The Honda City runs at the maximum speed of 150km/hr The Skoda Octavia runs at the maximum speed of 210km/hr

### 2\. Passing the wrong number of arguments.

```python
class Car:
    def __init__(self, name, mileage):
        self.name = name 
        self.mileage = mileage
        
Honda = Car("Honda City")
print(Honda)
```

\---------------------------------------------------------------------------

TypeError Traceback (most recent call last)

Cell In\[22\], line 6 3 [self.name](http://self.name) = name 4 self.mileage = mileage ----&gt; 6 Honda = Car("Honda City") 7 print(Honda)

TypeError: **init**() missing 1 required positional argument: 'mileage'

Since we did not provide the second argument, we got this error.

### 3\. Order of the arguments

```python
class Car:

    def __init__(self, name, mileage):
        self.name = name 
        self.mileage = mileage 

    def description(self):                
        return f"The {self.name} car gives the mileage of {self.mileage}km/l"
Honda = Car(24.1,"Honda City")
print(Honda.description())
```

The 24.1 car gives the mileage of Honda Citykm/l

Object Oriented Programming - Argument Order

Messed up! Notice we changed the order of arguments. Now, there are four fundamental concepts of Object-oriented programming – Inheritance, Encapsulation, Polymorphism, and Data abstraction. It is very important to know about all of these in order to understand OOPs. Till now we’ve covered the basics of OOPs, let’s dive in further.

### Inheritance in Python Class

Inheritance is the procedure in which one class inherits the attributes and methods of another class. The class whose properties and methods are inherited is known as Parent class. And the class that inherits the properties from the parent class is the Child class.

The interesting thing is, along with the inherited properties and methods, a child class can have its own properties and methods.

How to inherit a parent class? Use the following syntax:

```python
class parent_class:
    # body of parent class
    pass

class child_class( parent_class):
    # body of child class
    pass
```

Let’s see the implementation-

```python
class Car:          #parent class

    def __init__(self, name, mileage):
        self.name = name 
        self.mileage = mileage 

    def description(self):                
        return f"The {self.name} car gives the mileage of {self.mileage}km/l"

class BMW(Car):     #child class
    pass

class Audi(Car):     #child class
    def audi_desc(self):
        return "This is the description method of class Audi."
obj1 = BMW("BMW 7-series",39.53)
print(obj1.description())

obj2 = Audi("Audi A8 L",14)
print(obj2.description())
print(obj2.audi_desc())
```

The BMW 7-series car gives the mileage of 39.53km/l The Audi A8 L car gives the mileage of 14km/l This is the description method of class Audi.

We have created two child classes namely “BMW” and “Audi” who have inherited the methods and properties of the parent class “Car”. We have provided no additional features and methods in the class BMW. Whereas one additional method inside the class Audi.

Notice how the instance method description() of the parent class is accessible by the objects of child classes with the help of obj1.description() and obj2.description(). And also the separate method of class Audi is also accessible using [obj2.audi](http://obj2.audi)\_desc().

### Encapsulation

Encapsulation, as I mentioned in the initial part of the article, is a way to ensure security. Basically, it hides the data from the access of outsiders. Such as if an organization wants to protect an object/information from unwanted access by clients or any unauthorized person then encapsulation is the way to ensure this.

You can declare the methods or the attributes protected by using a single underscore ( *) before their names. Such as- self.*name or def \_method( ); Both of these lines tell that the attribute and method are protected and should not be used outside the access of the class and sub-classes but can be accessed by class methods and objects.

Though Python uses ‘ \_ ‘ just as a coding convention, it tells that you should use these attributes/methods within the scope of the class. But you can still access the variables and methods which are defined as protected, as usual.

Now for actually preventing the access of attributes/methods from outside the scope of a class, you can use “private members“. In order to declare the attributes/method as private members, use double underscore ( **) in the prefix. Such as – self.**name or def \_\_method(); Both of these lines tell that the attribute and method are private and access is not possible from outside the class.

```python
class car:

    def __init__(self, name, mileage):
        self._name = name                #protected variable
        self.mileage = mileage 

    def description(self):                
        return f"The {self._name} car gives the mileage of {self.mileage}km/l"
obj = car("BMW 7-series",39.53)

#accessing protected variable via class method 
print(obj.description())

#accessing protected variable directly from outside
print(obj._name)
print(obj.mileage)
```

The BMW 7-series car gives the mileage of 39.53km/l BMW 7-series 39.53

Notice how we accessed the protected variable without any error. It is clear that access to the variable is still public. Let us see how encapsulation works-

```python
class Car:
    def __init__(self, name, mileage):
        self.__name = name              #private variable        
        self.mileage = mileage 

    def description(self):                
        return f"The {self.__name} car gives the mileage of {self.mileage}km/l"
obj = Car("BMW 7-series",39.53)

#accessing private variable via class method 
print(obj.description())

#accessing private variable directly from outside
print(obj.mileage)
print(obj.__name)
```

### Encapsulation

When we tried accessing the private variable using the description() method, we encountered no error. But when we tried accessing the private variable directly outside the class, then Python gave us an error stating: car object has no attribute ‘\_\_name’.

You can still access this attribute directly using its mangled name. Name mangling is a mechanism we use for accessing the class members from outside. The Python interpreter rewrites any identifier with “\_\_var” as “\_ClassName\_\_var”. And using this you can access the class member from outside as well.

```python
class Car:

    def __init__(self, name, mileage):
        self.__name = name              #private variable        
        self.mileage = mileage 

    def description(self):                
        return f"The {self.__name} car gives the mileage of {self.mileage}km/l"
obj = Car("BMW 7-series",39.53)

#accessing private variable via class method 
print(obj.description())

#accessing private variable directly from outside
print(obj.mileage)
print(obj._Car__name)      #mangled name
```

The BMW 7-series car gives the mileage of 39.53km/l 39.53 BMW 7-series

Note that the mangling rule’s design mostly avoids accidents. But it is still possible to access or modify a variable that is considered private. This can even be useful in special circumstances, such as in the debugger.

### Polymorphism

This is a Greek word. If we break the term Polymorphism, we get “poly”-many and “morph”-forms. So Polymorphism means having many forms. In OOP it refers to the functions having the same names but carrying different functionalities.

```python
class Audi:
    def description(self):
        print("This the description function of class AUDI.")

class BMW:
    def description(self):
        print("This the description function of class BMW.")
audi = Audi()
bmw = BMW()
for car in (audi,bmw):
    car.description()
```

This the description function of class AUDI. This the description function of class BMW.

When the function is called using the object audi then the function of class Audi is called and when it is called using the object bmw then the function of class BMW is called.

### Data abstraction

We use Abstraction for hiding the internal details or implementations of a function and showing its functionalities only. This is similar to the way you know how to drive a car without knowing the background mechanism. Or you know how to turn on or off a light using a switch but you don’t know what is happening behind the socket.

Any class with at least one abstract function is an abstract class. In order to create an abstraction class first, you need to import ABC class from abc module. This lets you create abstract methods inside it. ABC stands for Abstract Base Class.

```python
from abc import ABC

class abs_class(ABC):
    # Body of the class
    pass
```

Important thing is– you cannot create an object for the abstract class with the abstract method. For example-

```python
from abc import ABC, abstractmethod

class Car(ABC):
    def __init__(self,name):
        self.name = name 

@abstractmethod
def price(self,x):
    pass
obj = Car("Honda City")
```

Now the question is how do we use this abstraction exactly. The answer is by using inheritance.

```python
from abc import ABC, abstractmethod

class Car(ABC):
    def __init__(self,name):
        self.name = name

    def description(self):
        print("This the description function of class car.")

    @abstractmethod
    def price(self,x):
        pass
class new(Car):
    def price(self,x):
        print(f"The {self.name}'s price is {x} lakhs.")
obj = new("Honda City")

obj.description()
obj.price(25)
```

This the description function of class car. The Honda City's price is 25 lakhs.

Car is the abstract class that inherits from the ABC class from the abc module. Notice how I have an abstract method (price()) and a concrete method (description()) in the abstract class. This is because the abstract class can include both of these kinds of functions but a normal class cannot. The other class inheriting from this abstract class is new(). This method is giving definition to the abstract method (price()) which is how we use abstract functions.

After the user creates objects from new() class and invokes the price() method, the definitions for the price method inside the new() class comes into play. These definitions are hidden from the user. The Abstract method is just providing a declaration. The child classes need to provide the definition.

But when the description() method is called for the object of new() class i.e obj, the Car’s description() method is invoked since it is not an abstract method.

**Taken most of content from:** [https://www.analyticsvidhya.com](https://www.analyticsvidhya.com)

# Exercise Question you will find in the exercise notebook of Day 2 on GitHub.

# If you liked it then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)