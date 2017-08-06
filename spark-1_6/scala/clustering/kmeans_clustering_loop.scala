package com.github.craig.clustering

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.hive.HiveContext
import scala.collection.mutable.ArrayBuffer

/* =============================================================================
Purpose: Help decide which K to use on your clustering algorithm.
Use a loop to iterate of K and record the cost. This is called the "Elbow method".
============================================================================= */

object KmeansClusteringLoop {

    // Function that returns just the cost of the kmeans clustering. The cost
    // will be returned and recorded.
    def clusterWellsKmeans(hc: HiveContext, df: DataFrame, k: Int, predictors: Array[String], iter: Int = 40, tol: Double = 1.0e-05) : Double = {

        val assembler = new VectorAssembler()
            .setInputCols(predictors)
            .setOutputCol("clusteringFeatures")

        // Normalizing each feature to have unit standard deviation and/or zero mean
        val scaler = new StandardScaler()
            .setInputCol("clusteringFeatures")
            .setOutputCol("scaledClustFeatures")
            .setWithStd(true)
            .setWithMean(true)

        val kmeansAlgorithm = new KMeans()
            .setK(k)
            .setSeed(1069)
            .setMaxIter(iter)         // <-- hyperparameter
            .setTol(tol)              // <-- hyperparameter
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

        cost
    }

    case class Row(col1: String, col2: String)

    def twoArrayBufToDF(sc: SparkContext, hc: HiveContext, firstBuf: ArrayBuffer[Double], secondBuf: ArrayBuffer[Int]): DataFrame = {

        import hc.implicits._

        val firstArr : Array[String] = firstBuf.toArray[Double].map(_.toString)
        val secondArr : Array[String] = secondBuf.toArray[Int].map(_.toString)

        val trans : Array[Array[String]] = Array(firstArr, secondArr).transpose
        val rdd = sc.parallelize(trans).map(ys => Row( ys(0), ys(1) ))
        val df = rdd.toDF("Cost", "K")

        df
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
        // Spark app: IngestCleanOFMData
        val df = hc.read.parquet("/results/dev/clean_data")

        // Filter dataframe
        val filter_1 : String = "ABCDEF"
        val filter_2 : String = "GHIJK"
        val filteredDF = df
            .filter(df("filter_col_1") === filter_1)
            .filter(df("filter_col_2") === filter_2)

        val predictorVariables : Array[String] = Array("pred_col_1", "pred_col_2", "pred_col_3")

        val kc = ArrayBuffer.empty[Double]
        val arrK = ArrayBuffer.empty[Int]

        // Create loop that iterates over K
        for (i <- 5 to 90 by 1) {

            val kmeansCost = clusterWellsKmeans(hc, filteredDF, i, predictorVariables)

            kc += kmeansCost
            arrK += i

        }

        // Produces a two column dataframe of results
        val kmeansDF = twoArrayBufToDF(sc, hc, kc, arrK)

        kmeansDF.show(100)

        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
