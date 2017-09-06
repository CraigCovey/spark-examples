# Making a pom.xml File

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

Creating a Spark application requires a large amount of dependencies. Dependencies are source code files that are required to make an application run. In our instance, a Spark app requires two basic things: Spark + Scala (before anyone objects, remember that Pyton & R code are compiled into Scala at runtime :stuck_out_tongue:). Both Spark & Scala require several main dependencies each that will be placed in your pom file. In the pom.xml file in this repository all the basic dependencies have been added for you.

Scala requires:

* scala-library
* scala-compiler
* scala-reflect
* scalap

Spark requires:

* spark-core_2.11
* spark-sql_2.11
* spark-mllib_2.11

Within these seven dependencies are even more dependencies. In total there are over 140 seperate dependencies that are needed to execute a Spark app on the cluster. The beauty of Apache Maven is that through the pom file it takes care of all the lesser dependencies. You just need to know the main ones.

Copy the pom.xml file in this repository into your project and change the *\<groupId>* & *\<artifactId>* at the top of the file to match your project structure. IntelliJ will dynamically download all the dependencies and you should see a progress bar at the bottom of the app.

# How the pom.xml Works

The pom.xml file, written in the IntelliJ app and executed by Maven, reads all the dependencies and downloads the source code and stores them on your actual computer. Maven stores all the source code in the *".m2"* folder that was created when you installed and configured Maven. For example, I am on a Windows machine and my folder is at *"C:\Users\\{my_username}\\.m2\\"*. Within the .m2 folder is another folder called "repository" which contains a directory structure of every dependency that you have ever loaded. 

Let's inspect the scala-library dependency. From the *repository* directory, navigate to *\\org\\scala-lang\\scala-library\\2.11.8\\* directory. You should see at least the following four files:

* scala-library-2.11.8.jar
* scala-library-2.11.8.javadoc.jar
* scala-library-2.11.8.sources.jar
* scala-library-2.11.8.pom

These four types of files are required for a dependency to work. 

Maven downloads these files from the internet from the Maven Central repository @ [https://repo.maven.apache.org/maven2/](https://repo.maven.apache.org/maven2/). You can browse for other dependencies using [The Central Repository](http://search.maven.org) or the [MVNRepository](https://mvnrepository.com).

... more to come ...

# Troubleshooting

... more to come ...
