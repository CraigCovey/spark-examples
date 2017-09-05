# Spark Machine Learning Code Examples

*All the code in this repository is the original code of Craig Covey.*

### Purpose

The purpose of this repository is to:

1. Store all of my various Spark code in one location
2. Be a guide for others in learning Spark
3. Highlight my work in Spark

### Content

As of September 2017 I upgraded to Spark 2.1 from 1.6 where I had spent the previous year working. My Spark 1.6 Scala machine learning code can be found in the *spark-1_6* folder.

With Spark 2.1 I will begin with a different approach. Before I upload machine learning code I will first include documentation with step-by-step instructions on creating: 

* pom.xml file
* Creating applications using IntelliJ IDEA & Apache Maven

**Stay tuned!!**

***

##### Code Repository

In this repository I include code on: 

* Ingesting and transforming data
* Clustering using KMeans
* Multiple Linear Regression

Within each directory of machine learning code I typically break out the files with a suffix of "_main" and "_func". The _main file consists of machine learning code all in one *Main* function. The _func file uses functional programming
to break up the code into usable functions. This is the optimal way to write code and keeps your code tidy. The _func file will also be more advanced than the _main file. So if you are learning Spark or a particular machine learning technique I recommend starting with the _main file. Sometimes I include other files and the suffix should easily identify the purpose of the code.

The code is heavily commented explaining what I am doing and in some places why I chose to do that particular thing.

`Happy Coding!`  :smiley: