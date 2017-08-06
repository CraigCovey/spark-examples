package com.github.craig.clustering

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, SaveMode}

object KmeansClusteringMain {

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
        // --- Create dataframe ---
        val df = hc.read.parquet("/results/dev/clean_data")

        // predictor/regressor variables
        // --- Create Array of column names ---
        val predictorVariables : Array[String] = Array("pred_col_1", "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")

        //// Assumes all predictor variables are numeric!!!

        val assembler = new VectorAssembler()
            .setInputCols(predictorVariables)
            .setOutputCol("clusteringFeatures")

        val scaler = new StandardScaler()
            .setInputCol("clusteringFeatures")
            .setOutputCol("scaledClusteringFeatures")
            .setWithMean(true)
            .setWithStd(true)

        val kmeansAlgorithm = new KMeans()
            .setK(10)                   // <-- number of clusters
            .setSeed(Random.nextLong())
            .setMaxIter(20)                 // <-- hyperparameter
            .setTol(1.0e-05)                // <-- hyperparameter
            .setFeaturesCol("scaledClusteringFeatures")
            .setPredictionCol("columnCategory")     // <-- create your own column name

        val pipeline = new Pipeline()
            .setStages(Array(assembler, scaler, kmeansAlgorithm))

        // Train model
        val pipelineModel = pipeline.fit(df)

        // Apply model to dataframe
        val kmeansPrediction = pipelineModel.transform(df)

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
        val cost = kmeansModel.computeCost(kmeansPrediction)
        println("Clustering Cost: " + cost)

        // Print cluster centers
        val centers = kmeansModel.clusterCenters
        println("Cluster Centers:")
        centers.foreach(println)

        // Print clustering prediction dataframe
        val kmeansResultDF = kmeansPrediction
            .select("row_id_column", "columnCategory", "pred_col_1", "pred_col_2", "pred_col_3",
                "pred_col_4", "pred_col_5", "pred_col_6")

        kmeansResultDF.show


        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
