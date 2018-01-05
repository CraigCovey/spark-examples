# Spark Machine Learning Code Examples

*All the code in this repository is the original code of Craig Covey.*

### Purpose

The purpose of this repository is to:

1. Store all of my various Spark code in one location
2. Be a guide for others in learning Spark
3. Highlight my work in Spark

### Content

I have Spark 1.6 code and Spark 2.1 code, each in their respective folders. I will do my best to include code (with comments), document explaining my code, and pom.xml file on each topic.

**Stay tuned!!**

***

##### Code Repository

In this repository I include code on: 

* Ingesting and transforming data
* Saving data to Impala with HDFS and Kudu
* Clustering using KMeans
* Multiple Linear Regression

Within each directory of machine learning code I typically break out the files with a suffix of "_main" and "_func". The _main file consists of machine learning code all in one *Main* function. The _func file uses functional programming
to break up the code into usable functions. This is the optimal way to write code and keeps your code tidy. The _func file will also be more advanced than the _main file. So if you are learning Spark or a particular machine learning technique I recommend starting with the _main file. Sometimes I include other files and the suffix should easily identify the purpose of the code.

The code is heavily commented explaining what I am doing and in some places why I chose to do that particular thing.

`Happy Coding!`  :smiley: