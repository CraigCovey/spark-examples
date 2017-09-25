# Exporting Spark Dataframe to an Impala Table

_By Craig Covey - 2017_

Instructions for quickly exporting a dataframe in Spark to an Impala table so the user can visualize data.

1. In Spark Scala app:

	* Make a dataframe of prediction results: 

		```
		...
		val pipelineModel = pipeline.fit(df_training)
		val predictionDF = pipelineModel.transform(df_testing)
		...
		```
	* Print schema of dataframe to determine which columns to remove and get a list of all column names and types to be used for Impala create table script.
	
		```
		println(predictionDF.printSchema())
		```
		Execute Spark app to retrieve `printSchema` output
	* From output of `printSchema` drop columns from dataframe that are vectors or arrays and column names that are Impala stopwords, like _"location"_

		```
		val dropColsDF = predictionDF.drop("vec_column_1", "vec_column_2", "vec_column_3", "location")
		```
		
	* Execute application again
2. Impala create table script

	* Copy the output from `printSchema` to a text editor with macros like Notepad++
	* Create macro that removes all characters and spaces before and after column name and data types. Every row should look like `columnName dataType, `
	* In a new text editor page, create the Impala Create Table script: 

		```
		CREATE EXTERNAL TABLE databaseName.tableName (
		
		)
		ROW FORMAT DELIMITED
		FIELDS TERMINATED BY ','
		LOCATION '/hdfs/table/location';
		```
		The `LOCATION` commands specifies the table location in HDFS where the data will be stored.
		
	* Copy and paste the column names and data types in between the parantheses and indent once:

		```
		CREATE EXTERNAL TABLE databaseName.tableName (
			columnName1 string,
			columnName2 integer,
			columnName3 double,
			columnName4 string
		)
		ROW FORMAT DELIMITED
		FIELDS TERMINATED BY ','
		LOCATION '/hdfs/table/location';
		```
	* Make sure that the columns in the final dataframe are the exact same as delimited in the create external table script!
	* In one of the nodes on the Hadoop cluster login to the impala shell, if needed create a Impala database, and then paste the entire create external table script into the console. Use the following commands:

		```
		$ impala-shell
		$ use default;
		$ create database testDB;
		$ show databases;
		$ use testDB;
		$ [paste create external table script]
		```
	* If no errors then the table was created in the specified database
3. Back in Spark Scala app:

	* At the end of the Spark code save the final dataframe as a CSV file in HDFS:

		```
		dropColsDF
			.write
			.format("csv")
			.mode(SaveMode.Overwrite)
			.save("/hdfs/table/location")
		```
	* Execute Spark app. The dataframe will be saved in same HDFS directory that Impala is pointing to. 
	* Be sure to _Invalidate Metadata_ for Impala to see data in Impala
4. Now, in HUE navigate to Impala and query table. **Additionaly, point a visualization tool like Spotfire or Microsoft Power BI to Impala table to visualize data.**


