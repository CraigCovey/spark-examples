package com.github.craig.clustering

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.functions._

object KmeansClusteringFunc {

    def kmeans_clustering(df: DataFrame, setK: Int, predictors: Array[String]) : DataFrame = {

        val assembler = new VectorAssembler()
            .setInputCols(predictors)
            .setOutputCol("clusteringFeatures")

        // Normalizing each feature to have unit standard deviation and/or zero mean
        val scaler = new StandardScaler()
            .setInputCol("clusteringFeatures")
            .setOutputCol("scaledClustFeatures")
            .setWithStd(true)
            .setWithMean(false)

        val kmeansAlgorithm = new KMeans()
            .setK(setK)
            .setSeed(1069)
            .setMaxIter(40)         // <-- hyperparameter
            .setTol(1.0e-05)        // <-- hyperparameter
            .setFeaturesCol("scaledClustFeatures")
            .setPredictionCol("columnCluster")

        val pipeline = new Pipeline()
            .setStages(Array(assembler, scaler, kmeansAlgorithm))

        // Train model
        val pipelineModel = pipeline.fit(df)

        // Apply model to dataframe
        val kmeansPrediction = pipelineModel.transform(df)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
        val cost = kmeansModel.computeCost(kmeansPrediction)
        println("Cost: " + cost)

        // Print cluster centers
        val centers = kmeansModel.clusterCenters
        println("Cluster Centers:")
        centers.foreach(println)

        // Add new wellCluster column to orginal dataframe
        val results: DataFrame = kmeansPrediction
            .withColumn("columnCluster", kmeansPrediction("columnCluster"))

        //// SparkSQL
        // Print dataframe of avg column per cluster
        results.groupBy("columnCluster")
            .agg(
                count(results("row_id_column")).as("row_id_column"),
                avg(results("pred_col_1")).as("pred_col_1"),
                avg(results("pred_col_2")).as("pred_col_2"),
                avg(results("pred_col_3")).as("pred_col_3")
            ).show(setK)

        results
    }

    def main(args: Array[String]) : Unit = {

        // Record clock time of code execution
        val t1 = System.nanoTime()

        /* =============================================================================
        In order to use Spark a SparkContext must be created. The SparkContext allows your
        Spark driver application to access the cluster through a resource manager, namely YARN.

        The HiveContext is part of SparkSQL. SparkSQL is a module of Spark. SparkSQL can be
        used to process structured data and has a SQL query engine. The HiveContext is needed
        to use dataframe and query data.
        ================================================================================ */

        val sc = new SparkContext()
        val hc = new HiveContext(sc)

        //// Start machine learning code ###################################################################################

        // Get clean data that is stored in HDFS as a parquet file
        val df = hc.read.parquet("/results/dev/clean_data")

        // Filter dataframe
        val filter_1 : String = "ABCDEF"
        val filter_2 : String = "GHIJK"
        val filteredDF = df
            .filter(df("filter_col_1") === filter_1)
            .filter(df("filter_col_2") === filter_2)

        println("Rows in DF: " + filteredDF.count)

        val predictorVariables = Array("pred_col_1", "pred_col_2", "pred_col_3")

        // Execute Kmeans clustering
        val results = kmeans_clustering(filteredDF, 21, predictorVariables)

        //// Save Dataframe in HDFs ===========================================================================
        // Save datframe as one text file
        // !!This only works if the dataframe isn't large!!
        results.repartition(1).write
            .format("com.databricks.spark.csv")
            .option("header", "true")
            .mode(SaveMode.Overwrite)
            .save("/user/results/dev/spark_ml/clustering/kmeans")


        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
