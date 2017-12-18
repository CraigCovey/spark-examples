# Intergrating Spark and Kudu
Saving a Spark dataframe to a Kudu table

_By Craig Covey - 2017_

_Note - This application assumes a Cloudera CDH cluster with Spark 2.1 and Kudu 1.3 properly installed and configured_

The complete Spark code, pom.xml file, and spark2-submit script can be found in ..... 

All of the code below is within the `def main(args: Array[String]) : Unit = {` function.

1. ### Create the Spark Session
	
	The entry point to Spark is the SparkSession class. The SparkSession creates a SparkContext, if one doesn't already exist. The SparkContext tells Spark how to access a cluster.
	
	Helpful references: 
	* [Spark 2.1 docs](https://spark.apache.org/docs/2.1.0/sql-programming-guide.html#starting-point-sparksession)
	* [databricks - The SparkSession](https://docs.databricks.com/spark/latest/gentle-introduction/sparksession.html)
	
	```
	// Create SparkSession
	val spark = SparkSession
	  .builder()
	  .getOrCreate()
	// Create Spark Context
	val context = spark.sparkContext
	```

2. ### Create the KuduContext

	The KuduContext, like the SparkContext, is the entry point to access Kudu in Spark. This must be present in each Kudu app.
	
	To get the kude master node name (server name): 
	* Open **Cloudera Manager** and click on the **Kudu** service
	* Under the **Instances** tab there is a table. Inside the table, find the **Host** with the **Role Type** of _Master_
	* Copy the Kudu master node name

	The defult port number to Kudu is `7051`. Add a `:` after the node name and then the port number `7051`.
	
	```
	// Kudu Master Node
	val kuduMasterNode = "master03-tst.example.com:7051"
	
	// Kudu Context
	// Provide server name of Kudu Master
	// Provide SparkSession.sparkContext
	val kuduContext = new KuduContext(kuduMasterNode, context)
	```

3. ### Name the Kudu Table
	
	Kudu has a database and table system similar to Impala. But you do not have to specifiy a database name to create a Kudu table. It just makes it easier to include a database name so you can easily group similar tables together. 
	
	The database name and table name are concatenated by a "." just like Impala. Also, notice that a prepend _"impala::"_ before the database name. This is because Kudu was designed to integrate with Impala. From Impala interface the user can insert, query, update, and delete data in a Kudu table. For more information see the [Kudu Impala docs](https://kudu.apache.org/docs/kudu_impala_integration.html). In step **7** we will show you how to link the Kudu table to Impala.
	
	Lastly, we check if the table name is already present in Kudu. If so we delete it so we can create it again in step **4**.
	
	```
	// Kudu Table Name
    val kuduDatabase = "spark"
    val kuduTable = "new_kudu_table"
    val kuduTableName = "impala::" + kuduDatabase + "." + kuduTable

    // Check if the table exists, and drop it if it does
    if (kuduContext.tableExists(kuduTableName)) {
        kuduContext.deleteTable(kuduTableName)
    }
	```

4. Create Table in Kudu
	
	This is the step where we actually create the table in Kudu. In the `createTable` function we specify the:
	* Table name
	* Schema from the dataframe
	* Column to use as the Kudu tables ID
	* Number of partitions
	* Number of replicas
	
	For more information on partitioning see the [docs](https://kudu.apache.org/docs/schema_design.html#partitioning).
	
	If the code stopped after the `createTable` function, the user could navigate to the Kudu Web UI and in the **Tables* tab could see the newly created table.
	
	```
    // Create Kudu Table
	kuduContext.createTable(
	    kuduTableName,
	    finalDF.schema,
	    Seq("id_column_name"),
	    new CreateTableOptions()
	      .addHashPartitions(List("id_column_name").asJava, 3)
	      .setNumReplicas(1)
	)
	```

5. Upsert Data into Kudu Table
	
	This is where data is physically uploaded into the Kudu table from the Spark dataframe. Kudu allows the user to `insert`, `upsert`, `update`, and even `delete` data in a Kudu table. 
	
	It is preferrable to `upsert` data instead of using `insert`. Insert only inserts new rows of data that aren't in the Kudu table already. Upsert inserts new rows as well as updates existing data rows. Also, if you try to insert a row that has already been inserted, the insertion will fail because the primary key would be duplicated. For more information on the various ways of adding data to a Kudu table see the Kudu [docs](https://kudu.apache.org/docs/kudu_impala_integration.html#_inserting_data_into_kudu_tables).
	
	```
	// Upsert rows from existing dataframe into Kudu table
	// Upsert will insert new rows and update existing rows
	kuduContext.upsertRows(finalDF, kuduTableName)
	```

6. Display Data in Kudu Table
	
	Now that the data is present in the Kudu table we can display results to the console. To do this we create a new dataframe of data that is loaded from the Kudu table by mapping the Kudu name and the Kudu server name. Finaly we display the results with the `.show()` function like any other dataframe.
	
	```
	// Load results of Kudu table
	// Include Impala and database references: https://stackoverflow.com/a/44245705
	val results = spark.read.options(Map("kudu.table" -> kuduTableName,
	                                    "kudu.master" -> kuduMasterNode))
	  .format("org.apache.kudu.spark.kudu")
	  .load()
	
	// Display results
	results.show(10)
	```

7. Working with Kudu is much easier than HDFS. Enjoy!


