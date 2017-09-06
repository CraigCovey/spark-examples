# Making a pom.xml File
***
### With IntelliJ IDEA & Apache Maven

A pom.xml (or just a pom file) is a file that is used by Apache Maven to compile together all the dependencies required to make a jar file. A jar file is needed to actually execute a Spark application. I know, it's already getting confusing. 

Think of it like this:

**pom.xml file + Spark code >> jar file >> Spark app >> Happiness!!**

### Assumptions:

* Installed [Java 8](http://www.oracle.com/technetwork/java/javase/downloads/index.html) JDK from Oracle
* Installed & set up [Apache Maven](https://maven.apache.org). Instructions [here](https://maven.apache.org/install.html)
* Installed [IntelliJ IDEA](https://www.jetbrains.com/idea/) Community Edition by JetBrains
* Used IntelliJ to create the skeleton of a Scala app. Instructions forthcoming...

# Background

Creating a Spark application requires a large amount of dependencies. Dependencies are source code files that are required to make an application run. In our instance, a Spark app requires to basic things: Spark + Scala (Before anyone objects, remember that Pyton & R code are compiled into Scala at runtime :stuck_out_tongue:) 

... more to come ...
