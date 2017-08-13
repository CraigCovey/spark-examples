package com.github.craig.regression

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.functions.mean
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}

/* =============================================================================
This Spark Scala 1.6 script uses random forest regression machine learning to
predict column "target_column_name" using six columns as predictor inputs.
================================================================================
*/

object RandomForest {

    def regressionEvaluator(df: DataFrame, target_var: String, target_pred: String) : (Array[String], Array[Double]) = {

        // Use regression to evaluate model
        val evaluator = new RegressionEvaluator()
            //.setMetricName("rmse")
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

        println("*********************************** Random Forest Regression ***********************************")

        // Get clean data that is stored in HDFS as a parquet file
        // Spark app: IngestCleanOFMData
        val df = hc.read.parquet("/results/dev/clean_data")

        // Filter dataframe
        val filter_1 : String = "ABCDEF"
        val filter_2 : String = "GHIJK"
        val filteredDF = df
            .filter(df("filter_col_1") === filter_1)
            .filter(df("filter_col_2") === filter_2)

        println("Rows in DF: " + filteredDF.count)

        // Split into training and testing dataframes
        val Array(training, testing) = filteredDF.randomSplit(Array(0.7, 0.3), seed = 1069)

        val predictorVariables = Array("pred_col_1", "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")

        // A transformer that merges multiple columns into a vector column
        // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
        val vectorAssembler = new VectorAssembler()
            .setInputCols(predictorVariables)
            .setOutputCol("rawFeatures")

        // Random Forest Algorithm
        val tree = new RandomForestRegressor()
            .setLabelCol("target_column_name")
            .setPredictionCol("pred_target_column_name")
            .setFeaturesCol("rawFeatures")

        println("Max bins: " + tree.getMaxBins)
        println("Max depth: " + tree.getMaxDepth)

        // Create pipeline
        val pipeline = new Pipeline()
            .setStages(Array(vectorAssembler, tree))

        // Apply pipeline to training dataframe
        val model = pipeline.fit(training)

        // Apply model to testing dataframe
        val predictionDF = model.transform(testing)

        // Evaluate Model
        val (metricNames, metricValues) = regressionEvaluator(predictionDF, "target_column_name", "pred_target_column_name")

        // Return metrics as a dataframe
        val resultsDF = resultsRowToDF(sc, hc, "target_column_name", "pred_target_column_name", metricValues, metricNames, geo_area,
            reservoir, testing.count)

        // Testing prediction dataframe
        val testDF = predictionDF
            .select("row_id_column", "target_column_name", "pred_target_column_name", "pred_col_1",
            "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")


        println("===== Testing Set Prediction Results =====")
        // Print Testing prediction results dataframe - top 20
        testDF.show()

        // Print Testing metrics dataframe
        println("===== Testing Set Metrics =====")
        resultsDF.show()

        println("##### Training Set Metrics #####")
        val treeModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
//        println("Learned regression forest model:\n" + treeModel.toDebugString)

        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
