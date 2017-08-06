package com.github.craig.ingest_data

import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.{concat_ws, round}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.functions._

/* =============================================================================
The purpose of this Scala script is to ingest data from several Impala tables
inside an Impala database. Filter and transform that data into one wide dataframe in Spark.
Finally save this dataframe as parquet file in HDFS that will be used by
other Scala scripts to perform machine learning.
================================================================================
*/

object cleanData {

    //// Data Ingestion ==================================================================================================
    // Use Impala query to retrieve data in Impala datbase: analysis_database
    // Combine all the tables and return one wide and flat dataframe where one row equals one well
    // Dataframe has no duplicate wells
    def getData(hc: HiveContext) : DataFrame = {

        val select_stmt =
            "select ".concat(
            "    a.column1 ").concat(
            "    , a.column1 ").concat(
            "    , a.column2 ").concat(
            "    , a.column3 ").concat(
            "    , b.column4 ").concat(
            "    , b.column5 ").concat(
            "    , b.column6 ").concat(
            "    , c.column7 ").concat(
            "    , c.column8 ").concat(
            "    , c.column9 ").concat(
            "from table_1 a ").concat(
            "left join table_2 b ").concat(
            "    on a.column1 = b.column4 ").concat(
            "left join table_3 ").concat(
            "    on a.column1 = c.column7")

        val df = hc.sql(select_stmt)

        df
    }

    //// Data Cleaning ===================================================================================================
    // Remove nulls, remove rows with zero, filter data, create calculated columns, etc.
    def transformData(df: DataFrame) : DataFrame = {

        // Filter data
        val filteredDF = df
            .filter(df("column2").isNotNull)
            .filter(df("column3").isNotNull)
            .filter(df("column5") === "zoom")
            .filter(df("column6") === "foo" || df("column6") === "bar")
            .filter(df("column7") > 0)
            .filter(df("column8") > 0 && df("column8") < 400)
            .orderBy("column1")

        // Create new columns
        val modifiedDF = filteredDF
            .withColumn("column7", filteredDF("column7").cast(DoubleType))
            .withColumn("column8", filteredDF("column8").cast(IntegerType))
            .withColumn("column7_thousands", filteredDF("column7") * 1000)
            .withColumn("column10", round(filteredDF("column7_thousands") / filteredDF("column9")))
            .withColumn("column11", when(filteredDF("column3").isNull, 0).otherwise(filteredDF("column3")))

        // Selective filtering
        val finalDF : DataFrame = {
          // In Scala an if statement creates its own scope

          if(modifiedDF("column6") == "foo") {
            // do something
            // ...
            modDF2
        } else if (modifiedDF("column5") == "zoom" && modifiedDF("column8") < 150) {
              // do something else
              // ...
              modDF2
          } else {
            // do nothing
            modDF2
          }

        }

        finalDF
    }

    //// Main ===================================================================================================
    def main(args: Array[String]) : Unit = {

        // Start time (real) of code execution
        val t1 = System.nanoTime()

        /* =============================================================================
        In order to use Spark a SparkContext must be created. The SparkContext allows your
        Spark driver application to access the cluster through a resource manager, namely YARN.

        The HiveContext is part of SparkSQL. SparkSQL is a module of Spark. SparkSQL can be
        used to process structured data and has a SQL query engine. The HiveContext is needed
        to use dataframe and query data.
        ================================================================================
        */
        val sc = new SparkContext()
        val hc = new HiveContext(sc)

        //// ###############################################################################################################

        // Ingest raw data
        val rawData = getData(hc)

        println("Raw rows: " + rawData.count)

        // Clean raw data
        val cleanData = transformData(rawData)

        println("Clean rows: " + cleanData.count)

        // Save dataframe as a parquet file in HDFS
        cleanData.write
            .format("parquet")
            .mode(SaveMode.Overwrite)
            .save("/results/dev/clean_data")


        //// ###############################################################################################################

        // Print time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
