package com.github.craig.linear_regression

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HiveContext

/* =============================================================================
This Spark Scala 1.6 script uses multiple linear regression machine learning to
predict column "target_column_name" using six columns as predictor inputs.

This code is simple in that all the code lies within the main fuction. 
================================================================================
*/

object MultLinearRegressionMain {

    def main(args: Array[String]) : Unit = {

        // Record clock time of code execution
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

        //// Start machine learning code ###################################################################################

        // Get clean data that is stored in HDFS as a parquet file
        // --- Create dataframe ---
        val df = hc.read.parquet("/results/dev/clean_data")

        // target/response/outcome/result varaible
        // --- Enter column name ---
        val target_variable : String = "target_column_name"
        val target_prediction : String = "pred_" + target_column_name

        // predictor/regressor variables
        // --- Create Array of column names ---
        val predictorVariables = Array("pred_col_1", "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")

        // Split df into 70% training and 30% testing dataframes
        // "seed" can be any long integer
        val Array(training, testing) = df.randomSplit(Array(0.7, 0.3), seed = 1069L)

        //// Begin machine learning pipeline

        // A transformer that merges multiple columns into a vector column
        // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
        // "rawFeatures" is just a name; could be anything you like
        val vectorAssembler = new VectorAssembler()
          .setInputCols(predictorVariables)
          .setOutputCol("rawFeatures")

          // Normalizing each feature to have unit standard deviation and/or zero mean
          val scaler = new StandardScaler()
            .setInputCol("rawFeatures")
            .setOutputCol("scaledFeatures")
            .setWithStd(true)
            .setWithMean(false)

        // Multiple Linear Regression Algorithm
        val algorithm = new LinearRegression()
          .setLabelCol(target_variable)
          .setPredictionCol(target_prediction)
          .setFeaturesCol("scaledFeatures")
          .setMaxIter(100)         // <-- hyperparameter
          .setRegParam(0)         // <-- hyperparameter
          .setElasticNetParam(0)     // <-- hyperparameter
          .setStandardization(true)

        // Create machine learning pipeline
        val pipeline = new Pipeline()
            .setStages(Array(vectorAssembler, scaler, algorithm))

        // Apply pipeline to training dataframe
        val model = pipeline.fit(training)

        // Apply model to testing dataframe
        val predictionDF = model.transform(testing)

        //// Optional step - Evaluate model +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        // Use regression to evaluate model
        val evaluator = new RegressionEvaluator()
          //.setMetricName("rmse")
          .setLabelCol(target_variable)
          .setPredictionCol(target_prediction)

        /**
            * Param for metric name in evaluation. Supports:
            *  - `"rmse"` (default): root mean squared error
            *  - `"mse"`: mean squared error
            *  - `"r2"`: R^2^ metric
            *  - `"mae"`: mean absolute error
            *  https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/evaluation/RegressionEvaluator.scala
        **/

        val r2 = evaluator.setMetricName("r2").evaluate(predictionDF)
        val rmse = evaluator.setMetricName("rmse").evaluate(predictionDF)
        val mae = evaluator.setMetricName("mae").evaluate(predictionDF)

        println("r2: " + r2)
        println("rmse: " + rmse)
        println("mae: " + mae)

        //// End of Evaluation step +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        // Print prediction dataframe
        predictionDF
          .select("row_id_column", "target_column_name", "pred_target_column_name", "pred_col_1",
          "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")
          .show()

        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
