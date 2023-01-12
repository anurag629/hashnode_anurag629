# A Comprehensive Guide to Relational Database Management Systems and SQL

# Day 7 of 100 Days Data Science Bootcamp from noob to expert.

# GitHub link: [Complete-Data-Science-Bootcamp](https://github.com/anurag629/Complete-Data-Science-Bootcamp)

# Main Post: [Complete-Data-Science-Bootcamp](https://anurag629.hashnode.dev/complete-data-science-roadmap-from-noob-to-expert)

## Recap Day 6

Yesterday we have studied in detail about Data Visualization using matplotlib and seaborn in Python.

# Let's Start

A DBMS (Database Management System) is a software application that interacts with end users, other applications, and the database itself to capture and analyze the data. It is a tool that allows data scientists to store, manage, and retrieve large amounts of structured and unstructured data.

There are different types of DBMSs, such as relational databases, NoSQL databases, and graph databases.

* Relational databases, such as MySQL and PostgreSQL, use a structured query language (SQL) to manage data stored in tables with rows and columns. These databases are well-suited for data that can be easily organized into a tabular format and are useful for data warehousing, business intelligence, and other reporting applications.
    
* NoSQL databases, such as MongoDB and Cassandra, are designed to handle large amounts of unstructured data and support different data models, such as document, key-value, and graph. These databases are useful for big data and real-time applications, such as real-time analytics, social media, and gaming.
    
* Graph databases, such as Neo4j and Amazon Neptune, are used to store and query data in a graph format, where nodes and edges represent entities and relationships. These databases are useful for applications that involve complex relationships, such as fraud detection, recommendation systems, and social network analysis.
    

In data science, the choice of DBMS depends on the type and volume of data, the complexity of the relationships among data, and the performance and scalability requirements of the application.

**In this article we will study in detail Relational databases**

### Basic Understanding

A relational database is a type of database that organizes data into tables, with each table consisting of rows and columns.

* `Tables`: A table is a collection of related data that is organized into rows and columns. Each table has a unique name that identifies it within the database. For example, a table named "customers" might contain information about all the customers of a company, with each row representing a different customer and each column representing a different piece of information about that customer, such as name, address, and phone number.
    
* `Columns`: A column is a vertical set of values in a table and it has a name and a data type. Each column represents a specific attribute of the data in the table, such as "customer\_name" or "customer\_address". For example, in a table named "customers", the columns could be "customer\_id", "customer\_name", "customer\_address", "customer\_phone\_number", "customer\_email" etc.
    
* `Rows`: A row is a horizontal set of values in a table and it represents a single record or tuple in the table. Each row represents a unique instance of the data in the table, such as a specific customer. For example, in a table named "customers", a row could represent a single customer with values such as "1" for the "customer\_id", "John Doe" for the "customer\_name", "123 Main St" for the "customer\_address", "555-555-5555" for the "customer\_phone\_number" and "johndoe@gmail.com" for the "customer\_email"
    
* `Primary keys`: A primary key is a column or set of columns in a table that uniquely identify each row in the table. Each table can have only one primary key, and it is used to enforce the integrity of the data and create relationships with other tables. For example, in a table named "customers" the "customer\_id" column might be the primary key, because it is unique for each customer and can be used to identify a specific customer in the table.
    
* `Foreign keys`: A foreign key is a column or set of columns in a table that refers to the primary key of another table. It is used to create relationships between tables and enforce referential integrity. For example, in a table named "orders" that contains information about customer orders, the "customer\_id" column might be a foreign key that refers to the primary key "customer\_id" in the "customers" table.
    
* `Indexing`: Indexing is the process of creating a separate data structure that allows for faster searching and sorting of data in a table. Indexes can be created on one or more columns in a table to improve the performance of queries that search for specific data. For example, if you frequently search for customers by their last name, you might create an index on the "customer\_name" column in the "customers" table to speed up those searches.
    

**Here is an example of a "customers" table:**

| **customer\_id** | **customer\_name** | **customer\_address** | **customer\_phone\_number** | **customer\_email** |
| --- | --- | --- | --- | --- |
| 1 | John Doe | 123 Main St | 555-555-5555 | [**johndoe@gmail.com**](mailto:johndoe@gmail.com) |
| 2 | Jane Smith | 456 Park Ave | 555-555-5556 | [**janesmith@gmail.com**](mailto:janesmith@gmail.com) |
| 3 | Bob Johnson | 789 Elm St | 555-555-5557 | [**bobjohnson@gmail.com**](mailto:bobjohnson@gmail.com) |

In this example table, the "customer\_id" column is the primary key and it is used to uniquely identify each customer. The other columns such as "customer\_name", "customer\_address", "customer\_phone\_number" and "customer\_email" are non-key attributes and they provide additional information about each customer.

An index can be created on any column to speed up the query performance. In this example, an index can be created on the "customer\_name" column if we frequently need to search customer by their name.

It is important to note that the example is a very simple representation of the table and in real world scenario, tables can have multiple columns as primary key, multiple foreign keys and a complex data model with multiple tables, but the basic concept remains the same.

### SQL Quries

**We will execute all our queries using Python and SQLite**

#### Connecting to a database

To start interacting with the database we first we need to establish a connection.

```python
#!/usr/bin/python

import sqlite3

conn = sqlite3.connect('100daysofdatascience.db')

print("Opened database successfully");
```

Opened database successfully

#### Sample data

```python
# Customers table:
conn.execute('''CREATE TABLE customers (
                        id INT PRIMARY KEY,
                        first_name VARCHAR(255),
                        last_name VARCHAR(255),
                        city VARCHAR(255)
                    );''')

conn.execute('''INSERT INTO customers (id, first_name, last_name, city) 
                            VALUES (1, 'John', 'Smith', 'Paris'),
                                   (2, 'Mary', 'Johnson', 'London'),
                                   (3, 'Michael', 'Williams', 'Berlin'),
                                   (4, 'Brad', 'Brown', 'Rome');''')

print("Customer table created successfully!")
```

Customer table created successfully!

```python
# Products table:
conn.execute('''CREATE TABLE products (
                        id INT PRIMARY KEY,
                        product_name VARCHAR(255),
                        category VARCHAR(255),
                        price DECIMAL(10,2),
                        in_stock BOOLEAN
                    );''')

conn.execute('''INSERT INTO products (id, product_name, category, price, in_stock)
                        VALUES (1, 'MacBook Pro', 'electronics', 1500, true),
                               (2, 'iPhone', 'electronics', 1000, true),
                               (3, 'T-Shirt', 'clothing', 20, true),
                               (4, 'Jeans', 'clothing', 50, false);''')

print("Products table created successfully!")
```

Products table created successfully!

```python
# Orders table:
conn.execute('''CREATE TABLE orders (
                        id INT PRIMARY KEY,
                        customer_id INT,
                        order_date DATE,
                        total DECIMAL(10,2)
                    );''')

conn.execute('''INSERT INTO orders (id, customer_id, order_date, total)
                        VALUES (1, 1, '2021-01-01', 100),
                               (2, 2, '2021-01-02', 200),
                               (3, 3, '2021-01-03', 150),
                               (4, 2, '2021-01-04', 75);''')

print("Orders table created successfully!")
```

Orders table created successfully!

```python
# Create the employee table
conn.execute('''CREATE TABLE employees
                (id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                salary REAL);''')

# Insert data into the employee table
conn.execute("INSERT INTO employees (id, name, salary) VALUES (1, 'John Smith', 50000)")
conn.execute("INSERT INTO employees (id, name, salary) VALUES (2, 'Mary Johnson', 55000)")
conn.execute("INSERT INTO employees (id, name, salary) VALUES (3, 'Michael Williams', 60000)")
conn.execute("INSERT INTO employees (id, name, salary) VALUES (4, 'Brad Brown', 65000)")

print("Employees table created successfully!")
```

Employees table created successfully!

```python
conn.commit()
```

### Here are some examples of basic-to-advanced SQL queries that a data scientist might use to retrieve and manipulate data:

### SELECT:

The SELECT statement is used to retrieve data from one or more tables. For example, the following query retrieves all columns from a table called "customers":

```python
import pandas as pd
data = pd.read_sql_query("SELECT * FROM customers", conn)
data
```

|  | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- |
| 0 | 1 | John | Smith | Paris |
| 1 | 2 | Mary | Johnson | London |
| 2 | 3 | Michael | Williams | Berlin |
| 3 | 4 | Brad | Brown | Rome |

### WHERE:

The WHERE clause is used to filter the data returned by a SELECT statement. For example, the following query retrieves all columns from the "customers" table for customers whose city is "New York":

```python
data = pd.read_sql_query("SELECT * FROM customers WHERE city = 'London'", conn)
data
```

|  | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- |
| 0 | 2 | Mary | Johnson | London |

### JOIN:

The JOIN clause is used to combine rows from two or more tables based on a related column between them. For example, the following query retrieves all columns from the "orders" table and the "customers" table, where the customer\_id in the "orders" table matches the id in the "customers" table:

```python
data = pd.read_sql_query("SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id", conn)
data
```

|  | id | customer\_id | order\_date | total | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 2021-01-01 | 100 | 1 | John | Smith | Paris |
| 1 | 2 | 2 | 2021-01-02 | 200 | 2 | Mary | Johnson | London |
| 2 | 3 | 3 | 2021-01-03 | 150 | 3 | Michael | Williams | Berlin |
| 3 | 4 | 2 | 2021-01-04 | 75 | 2 | Mary | Johnson | London |

### UPDATE:

The UPDATE statement is used to modify data in a table. For example, the following query updates the salary of all employees in the "employees" table by 10%:

```python
conn.execute('''UPDATE employees SET salary = salary * 10;''')
conn.commit()
```

```python
data = pd.read_sql_query("SELECT * FROM employees", conn)
data
```

|  | id | name | salary |
| --- | --- | --- | --- |
| 0 | 1 | John Smith | 500000.0 |
| 1 | 2 | Mary Johnson | 550000.0 |
| 2 | 3 | Michael Williams | 600000.0 |
| 3 | 4 | Brad Brown | 650000.0 |

### DELETE:

The DELETE statement is used to remove data from a table. For example, the following query deletes all rows from the "employees" table where the salary is less than $50,000:

```python
conn.execute('''DELETE FROM employees WHERE salary < 550000;''')
conn.commit()
data = pd.read_sql_query("SELECT * FROM employees", conn)
data
```

|  | id | name | salary |
| --- | --- | --- | --- |
| 0 | 2 | Mary Johnson | 550000.0 |
| 1 | 3 | Michael Williams | 600000.0 |
| 2 | 4 | Brad Brown | 650000.0 |

### GROUP BY:

The GROUP BY clause is used to group rows in a SELECT statement based on one or more columns. For example, the following query retrieves the total in\_stock for each product category:

```python
data = pd.read_sql_query("SELECT category, SUM(in_stock) FROM products GROUP BY category;", conn)
data
```

|  | category | SUM(in\_stock) |
| --- | --- | --- |
| 0 | clothing | 1 |
| 1 | electronics | 2 |

### ORDER BY:

The ORDER BY clause is used to sort the data returned by a SELECT statement. For example, the following query retrieves all columns from the "customers" table, sorted by last name in ascending order:

```python
data = pd.read_sql_query("SELECT * FROM customers ORDER BY last_name;", conn)
data
```

|  | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- |
| 0 | 4 | Brad | Brown | Rome |
| 1 | 2 | Mary | Johnson | London |
| 2 | 1 | John | Smith | Paris |
| 3 | 3 | Michael | Williams | Berlin |

### LIMIT:

The LIMIT clause is used to limit the number of rows returned by a SELECT statement. For example, the following query retrieves the top 2 products by in\_stocks:

```python
data = pd.read_sql_query("SELECT * FROM products ORDER BY in_stock DESC LIMIT 2;", conn)
data
```

|  | id | product\_name | category | price | in\_stock |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | MacBook Pro | electronics | 1500 | 1 |
| 1 | 2 | iPhone | electronics | 1000 | 1 |

### LIKE:

The LIKE operator is used to search for a specific pattern in a column. For example, the following query retrieves all customers whose last name starts with "S":

```python
data = pd.read_sql_query("SELECT * FROM customers WHERE last_name LIKE 'S%';", conn)
data
```

|  | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- |
| 0 | 1 | John | Smith | Paris |

### INNER JOIN:

The INNER JOIN keyword is used to combine rows from two or more tables based on a related column between them. This will only return rows when there is at least one match in both tables. For example:

```python
data = pd.read_sql_query('''SELECT orders.id, customers.first_name
                            FROM orders
                            INNER JOIN customers ON orders.customer_id = customers.id;''', conn)
data
```

|  | id | first\_name |
| --- | --- | --- |
| 0 | 1 | John |
| 1 | 2 | Mary |
| 2 | 3 | Michael |
| 3 | 4 | Mary |

### OUTER JOIN:

The OUTER JOIN keyword is used to combine rows from two or more tables based on a related column between them. This will return all rows from one table and the matching rows from the other table. If there is no match, NULL values will be returned. For example:

```python
data = pd.read_sql_query('''SELECT orders.id, customers.first_name
                            FROM orders
                            LEFT OUTER JOIN customers ON orders.customer_id = customers.id;
                            ''', conn)
data
```

|  | id | first\_name |
| --- | --- | --- |
| 0 | 1 | John |
| 1 | 2 | Mary |
| 2 | 3 | Michael |
| 3 | 4 | Mary |

### UNION:

The UNION operator is used to combine the result-set of two or more SELECT statements. The UNION operator selects only distinct values by default. For example:

```python
data = pd.read_sql_query('''SELECT * FROM orders UNION SELECT * FROM customers;''', conn)
data
```

|  | id | customer\_id | order\_date | total |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 2021-01-01 | 100 |
| 1 | 1 | John | Smith | Paris |
| 2 | 2 | 2 | 2021-01-02 | 200 |
| 3 | 2 | Mary | Johnson | London |
| 4 | 3 | 3 | 2021-01-03 | 150 |
| 5 | 3 | Michael | Williams | Berlin |
| 6 | 4 | 2 | 2021-01-04 | 75 |
| 7 | 4 | Brad | Brown | Rome |

### INDEX:

Indexes are used to improve the performance of a database by allowing the database to find and retrieve specific rows much faster. For example, the following command creates an index on the "last\_name" column in the "customers" table:

```python
conn.execute('''CREATE INDEX last_name_indexes ON customers (last_name);''')
conn.commit()
```

### Date and Time Functions:

SQL provides a number of functions to work with date and time data types. Some examples include:

* CURDATE() - returns the current date
    
* NOW() - returns the current date and time
    
* YEAR() - returns the year of a given date
    
* MONTH() - returns the month of a given date
    
* DAY() - returns the day of a given date
    

As we are using SQLite. So, SQLite provides several built-in functions for working with date and time. Here is a list of some of the most commonly used date and time functions in SQLite:

* `date(timestring, modifier, modifier, ...)`: This function returns the date part of a date-time string.
    
* `time(timestring, modifier, modifier, ...)`: This function returns the time part of a date-time string.
    
* `datetime(timestring, modifier, modifier, ...)`: This function returns the date and time parts of a date-time string.
    
* `julianday(timestring, modifier, modifier, ...)`: This function returns the Julian day - the number of days since noon in Greenwich on November 24, 4714 B.C.
    
* `strftime(format, timestring, modifier, ...)`: This function returns a string representation of the date and time based on the specified format. The format string can contain various placeholders for different parts of the date and time, such as %Y for the year, %m for the month, %d for the day, %H for the hour, %M for the minute, and %S for the second.
    
* `date(timestring, '+' or '-', number, 'days' or 'months' or 'years')`: This function allows you to add or subtract days, months, or years from a date.
    
* `current_date`, `current_time`, `current_timestamp`: These functions return the current date, time, and timestamp respectively.
    
* `year(timestring)`, `month(timestring)`, `day(timestring)`, `hour(timestring)`, `minute(timestring)`, `second(timestring)`: These functions return the year, month, day, hour, minute, and second of the given timestring respectively.
    

For example, the following query retrieves the total sales for each month of the current year:

```python
data = pd.read_sql_query('''SELECT strftime('%m', order_date) as month, SUM(total) as totals
                                FROM orders
                                WHERE strftime('%Y', order_date) = strftime('%Y', 'now')
                                GROUP BY month;''', conn)
```

### Handling NULL values:

SQL provides several operators and functions to handle NULL values. Some examples include:

* IS NULL - used to check for NULL values
    
* IS NOT NULL - used to check for non-NULL values
    
* COALESCE() - returns the first non-NULL value in a list of expressions
    
* NULLIF() - returns NULL if two expressions are equal, otherwise it returns the first expression
    

For example, the following query retrieves all customers who have not placed any orders:

```python
data = pd.read_sql_query('''SELECT * FROM customers
                                WHERE id NOT IN (SELECT customer_id FROM orders)
                                ''', conn)
data
```

|  | id | first\_name | last\_name | city |
| --- | --- | --- | --- | --- |
| 0 | 4 | Brad | Brown | Rome |

### Other data types:

SQL also supports other data types such as BLOB (binary large object) for storing binary data, BOOLEAN for storing true/false values, and ENUM for storing predefined set of strings.

For example, the following query retrieves all products with image and only if the products are in stock:

```python
data = pd.read_sql_query('''SELECT product_name, in_stock FROM products
                                    WHERE product_name IS NOT NULL AND in_stock = TRUE;
                                    ''', conn)
data
```

|  | product\_name | in\_stock |
| --- | --- | --- |
| 0 | MacBook Pro | 1 |
| 1 | iPhone | 1 |
| 2 | T-Shirt | 1 |

## Summary

This article provides a comprehensive guide to relational database management systems (RDBMS) and SQL (Structured Query Language) for data science and analysis. It covers the basics and advanced concepts of RDBMS and SQL, including data types, queries, and advanced concepts such as joins and subqueries. The article also provides examples and code snippets to help illustrate the concepts and provide a hands-on learning experience. The main goal of this article is to help data scientists, analysts, and developers understand the importance of RDBMS and SQL in managing and analyzing large datasets.

## **Exercise Question you will find in the exercise notebook of Day 6 on GitHub.**

## If you liked it then...

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/default-orange.png align="left")](https://www.buymeacoffee.com/anurag629)