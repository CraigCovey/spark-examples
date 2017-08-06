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
 
This code is more complex because it uses functional programming to split up code into units.
================================================================================
*/

object MultLinearRegressionFunc {

    def linearRegressionPipeline(tv: String, tp: String, pred: Array[String]) : Pipeline = {
        // tv = target_variable
        // tp = target_prediction
        // pred = array of predictor variables

        // A transformer that merges multiple columns into a vector column
        // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
        // "rawFeatures" is just a name; could be anything you like
        val vectorAssembler = new VectorAssembler()
          .setInputCols(pred)
          .setOutputCol("rawFeatures")

          // Normalizing each feature to have unit standard deviation and/or zero mean
          val scaler = new StandardScaler()
            .setInputCol("rawFeatures")
            .setOutputCol("scaledFeatures")
            .setWithStd(true)
            .setWithMean(false)

        // Multiple Linear Regression Algorithm
        val algorithm = new LinearRegression()
          .setLabelCol(tv)
          .setPredictionCol(tp)
          .setFeaturesCol("scaledFeatures")
          .setMaxIter(100)         // <-- hyperparameter
          .setRegParam(0)         // <-- hyperparameter
          .setElasticNetParam(0)     // <-- hyperparameter
          .setStandardization(true)

        // Create machine learning pipeline
        val pipeline = new Pipeline()
            .setStages(Array(vectorAssembler, scaler, algorithm))

        pipeline
    }

    def regressionEvaluator(df: DataFrame, target_var: String, target_pred: String) : (Array[String], Array[Double]) = {

      // Use regression to evaluate model
      val evaluator = new RegressionEvaluator()
        .setLabelCol(target_var)
        .setPredictionCol(target_pred)

      /**
        * Param for metric name in evaluation. Supports:
        *  - `"rmse"` (default): root mean squared error
        *  - `"mse"`: mean squared error
        *  - `"r2"`: R^2^ metric
        *  - `"mae"`: mean absolute error
        *  https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/evaluation/RegressionEvaluator.scala
      **/

      val metricNames : Array[String] = Array("rsquared", "rmse", "mae", "mtv")

      // mtv = mean target variable - metric I created that is the mean of all target_variables in dataframe
      // Helps me insights into how far off "metrics" are from the mean

      val metricValues : Array[Double] = Array(
        evaluator.setMetricName("r2").evaluate(df),
        evaluator.setMetricName("rmse").evaluate(df),
        evaluator.setMetricName("mae").evaluate(df),
        df.agg(mean(df(target_var))).head(){0}.asInstanceOf[Double]
      )

      (metricNames, metricValues)
    }

    // Take results of evaluator and produces a one row dataframe
    // Take an Array[Double} and convert it into a dataframe
    def resultsRowToDF(sc: SparkContext, hc: HiveContext, tar: String, pred: String, results: Array[Double], metrics: Array[String]) : DataFrame = {
      // http://stackoverflow.com/a/40801637
      import hc.implicits._   // Needed to convert rdd to df
      val strResults : Array[String] = results.map(_.toString)
      // Make one Row of data
      val rows : Row = Row(tar, pred, strResults(0), strResults(1), strResults(2), strResults(3))
      // Must Seq the Row object in order to work
      // .getString() only works on Rows (not arrays)
      val df = sc.parallelize(Seq(rows)).map(rows => (rows.getString(0), rows.getString(1), rows.getString(2),
        rows.getString(3), rows.getString(4), rows.getString(5) ))
        .toDF("target", "prediction", metrics(0), metrics(1), metrics(2), metrics(3))

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

        // Call multiple linear regression pipeline
        val pipeline = linearRegressionPipeline(target_variable, target_prediction, predictorVariables)

        // Apply pipeline to training dataframe
        val model = pipeline.fit(training)

        // Apply model to testing dataframe
        val predictionDF = model.transform(testing)

        // Evaluate Model
        val (metricNames, metricValues) = regressionEvaluator(predictionDF, target_variable, target_prediction)

        // Return metrics as a dataframe
        val resultsDF = resultsRowToDF(sc, hc, target_variable, target_prediction, metricValues, metricNames)

        // Print prediction dataframe
        predictionDF
          .select("row_id_column", "target_column_name", "pred_target_column_name", "pred_col_1",
          "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")
          .show()

        // Print metrics dataframe
        resultsDF.show()

        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
