package com.datascience.craig

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PCA, StandardScaler, VectorAssembler}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.mean
import org.apache.spark.sql.functions._
import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object FeatureReduction {

    // Prints correlation between every column combination pair in a dataframe
    def corr_or_covMatrix(df: DataFrame, statType: String = "corr") : Unit = {

        if (statType == "corr") {
            println("--- Pearson Correlation Matrix ---")
        } else if (statType == "cov") {
            println("+++ Covariance Matrix +++")
        } else {
            System.exit(1)
        }

        // Array of dataframe column names
        val names = df.columns
        // Split column names into unique pairs as an array of arrays
        val paired_cols = names.mkString(",").split(",").combinations(2).toArray

        // Loop that prints the perason correlation between every column pair in the dataframe
        for (i <- paired_cols.indices) {

            // First column name of ith pair
            val p_0 = paired_cols(i)(0)
            // Second column name of ith pair
            val p_1 = paired_cols(i)(1)

            if (statType == "corr") {

                // Pearson Correlation between two columns
                val cor = df.stat.corr(p_0, p_1)
                println(s"$p_0 vs $p_1 : corr = $cor")

            } else if (statType == "cov") {

                // Covariance between two columns
                val cov = df.stat.cov(p_0, p_1)
                println(s"$p_0 vs $p_1 : cov = $cov")

            }

        }

    }

    def principalComponentAnalysis(df: DataFrame, input_cols: Array[String], k: Int) : Unit = {

        println("********************************* Principal Component Analysis (PCA) *********************************")
        print("Principal Component Analysis is a strategy for dimensionality reduction. " +
          "PCA can be interpreted as identifying the directions of maximum variance in high-dimensional data and " +
          "project it onto a smaller dimensional subspace while retaining most of the information." +
          "The eigenvectors form the axes. However, the eigenvectors only define the directions of the new axis, since they have all the same unit length 1." +

          "In order to decide which eigenvector(s) can dropped without losing too much information for the construction of " +
          "lower-dimensional subspace, we need to inspect the corresponding eigenvalues: The eigenvectors with the lowest " +
          "eigenvalues bear the least information about the distribution of the data; those are the ones can be dropped." +

          "The next question is “how many principal components are we going to choose for our new feature subspace?” A " +
          "useful measure is the so-called “explained variance,” which can be calculated from the eigenvalues. The explained " +
          "variance tells us how much information (variance) can be attributed to each of the principal components.")
        println("  ")

        // A transformer that merges multiple columns into a vector column
        // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
        val pca_assembler = new VectorAssembler()
          .setInputCols(input_cols)
          .setOutputCol("rawFeatures")

        // Normalizing each feature to have unit standard deviation and/or zero mean
        val pca_scaler = new StandardScaler()
          .setInputCol("rawFeatures")
          .setOutputCol("scaledFeatures")
          .setWithStd(true)
          .setWithMean(true)

        // Create pipeline
        // Produces scaled data
        val pca_pipeline = new Pipeline()
          .setStages(Array(pca_assembler, pca_scaler))

        // Apply pipeline to training dataframe
        // Creates a PipelineModel
        val pca_pipelineModel = pca_pipeline.fit(df)

        // Scaled dataframe
        val scaledDF = pca_pipelineModel.transform(df)

        // Determine the optimum 'k' by printing variance for every principal component
        // Loop of PCA where k = 1 to input_cols.length
        for (i <- 1 to input_cols.length) {

            val loop_pca = new PCA()
              .setInputCol("scaledFeatures")
              .setOutputCol("reducedFeatures")
              .setK(i)
              .fit(scaledDF)

            // Explained variance
            val r : Double = loop_pca.explainedVariance(i - 1)
            // Sum of explained variance for each k
            val v : Double = loop_pca.explainedVariance.values.sum
            println(s"k = $i; var = $r; sum var = $v")
        }

        // Create Principal Component Analysis Spark transformer
        val pca = new PCA()
          .setInputCol("scaledFeatures")
          .setOutputCol("reducedFeatures")
          .setK(k)
          .fit(scaledDF)

        // A principal components Matrix. Each column is one principal component.
        println("--- Principal Components Matrix ---")
        println("i.e. eigenvectors -- give us the directions of maximal variance")
        println(pca.pc.toString(Int.MaxValue, Int.MaxValue))

        // Explained Variance
        // A vector of proportions of variance explained by each principal component.
        println("--- Variance of each principal component ---")
        println("The explained variance tells us how much information (variance) can be attributed to each of the principal components.")
        println("Note: this vector is NOT the eigenvalues but the explained variance which is computed FROM the eigenvalues.")
        println("The object is the find the minimum number of principal components that explain the maximum amount of variance.")
        val var_expl = pca.explainedVariance
        println(var_expl.toString())
        // Return as array
        val var_expl_arr : Array[Double] = var_expl.values
        // Sum the values of the array
        val sum_var_expl = var_expl_arr.sum
        println("Sum of Explained Variance")
        println(sum_var_expl)

        // PCA Projections of nth dimensional space down to k dimensional space
        val pcaDF = pca.transform(scaledDF).select("reducedFeatures")
        // pcaDF.show(false)

    }

    def allPossibleCombinationsRegression(ss: SparkSession, df: DataFrame, preds: Array[String], tv: String) : Unit = {

        println("********************************* All Possible Combinatioins Regression *********************************")
        println("... explaing what is going on ...")

        // Create training and testing dataframes
        val Array(comb_training, comb_testing) = df.randomSplit(Array(0.7, 0.3), seed = 1069)

        // target/response/outcome/result varaible
        val target_variable : String = tv
        val target_prediction : String = "pred_" + target_variable

        // Predictor variables
        val predictor_columns : Array[String] = preds

        // Filter dataframe to only predictor_columns
        val predDF : DataFrame = df.select(predictor_columns.head, predictor_columns.tail: _*)
        // List of column names
        val col_names = predDF.columns.toList

        // All possible column names combinations as an array of arrays
        // Source: https://stackoverflow.com/a/13109916
        val all_combins = col_names.toSet[String].subsets.map(x => x.toArray).toArray
        // val all_combins = col_names.toSet[String].subsets.map(x => x.toList).toList

        // Filter out the blank array
        val filtered_combins = all_combins.filter(x => x.length > 0)

        // Print column names combinations
        // println(filtered_combins.deep.mkString(", "))


        // Mutable array for Prediction Metrics
        val metsArrM = ArrayBuffer[Array[Double]]()

        // Mutable array for Prediction Columns
        val colArrM = ArrayBuffer[Array[String]]()

        // Loop over array of combinations of predictor variables
        for (i <- filtered_combins.indices) {

            // Select ith array of predictors
            val c = filtered_combins(i)

            // A transformer that merges multiple columns into a vector column
            // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
            val comb_assembler = new VectorAssembler()
              .setInputCols(c)
              .setOutputCol("rawFeatures")

            // Normalizing each feature to have unit standard deviation and/or zero mean
            val comb_scaler = new StandardScaler()
              .setInputCol("rawFeatures")
              .setOutputCol("scaledFeatures")
              .setWithStd(true)
              .setWithMean(true)

            // Multiple Linear Regression Algorithm
            val algorithmLR = new LinearRegression()
              .setLabelCol(target_variable)
              .setPredictionCol(target_prediction)
              .setFeaturesCol("scaledFeatures")
              .setMaxIter(100)          // default: 100
              .setRegParam(0)           // default: 0
              .setElasticNetParam(0)    // default: 0
              .setStandardization(true)

            // Create pipeline
            val comb_pipeline = new Pipeline()
              .setStages(Array(comb_assembler, comb_scaler, algorithmLR))

            // Apply pipeline to training dataframe
            // Creates a PipelineModel
            val comb_pipeModel = comb_pipeline.fit(comb_training)

            // Apply PipelineModel to testing dataframe
            // Creates a dataframe with new prediction column
            val combPredictionDF = comb_pipeModel.transform(comb_testing)

            // Use regression to evaluate model
            val combEvaluator = new RegressionEvaluator()
              .setLabelCol(target_variable)
              .setPredictionCol(target_prediction)

            // Apply regression evaluation to prediction dataframe
            val metricValues : Array[Double] = Array(
                c.length,
                combEvaluator.setMetricName("r2").evaluate(combPredictionDF),
                combEvaluator.setMetricName("rmse").evaluate(combPredictionDF),
                combEvaluator.setMetricName("mae").evaluate(combPredictionDF)
                // mean target value (mtv): combPredictionDF.agg(mean(combPredictionDF(target_variable))).head(){0}.asInstanceOf[Double]
            )

            // Display Mean Target Variable (MTV) --- one time
            // Use mtv as a metic to compare to RMSE and MAE
            if (i == 0) {
                val mtv : Double = combPredictionDF.agg(mean(combPredictionDF(target_variable))).head(){0}.asInstanceOf[Double]
                println("   ")
                println("   ")
                println(s"Mean Target Variable: $mtv")
            }

            //            println(s"Predictors: ${c.mkString(", ")}")
            //            println(s"Prediction Metrics: ${metricValues.mkString(", ")}")

            // Collect predictor columns in a nested array
            colArrM += c
            // Collect metric values in a nested array
            metsArrM += metricValues

        }

        import ss.implicits._
        // https://stackoverflow.com/a/42182950
        // List of string of predictors
        val predictors_list = colArrM.toArray.map(x => x.mkString(", ")).toList
        // Make an one column dataframe of predictors
        val tmpDF = predictors_list.toDF()
        // Create id column to join with other DFs
        val predictorsDF = tmpDF
          .withColumnRenamed("value", "preds")
          .withColumn("id", monotonically_increasing_id())  // requires import org.apache.spark.sql.functions._

        // Nested list of metric values
        val metrics_list = metsArrM.toArray.map(y => y.toList).toList
        // Make an one column dataframe of list of metric values
        val metsDF = metrics_list.toDF()
        // Convert to four column dataframe
        val metricsDF = metsDF
          .withColumn("num_preds", metsDF("value").getItem(0).cast("Double"))
          .withColumn("r2", metsDF("value").getItem(1).cast("Double"))
          .withColumn("rmse", metsDF("value").getItem(2).cast("Double"))
          .withColumn("mae", metsDF("value").getItem(3).cast("Double"))
          .withColumn("id", monotonically_increasing_id) // requires import org.apache.spark.sql.functions._
          .drop("value")

        // Join dataframes by id column
        // Without duplicating joining column
        val combinedDF = predictorsDF.join(metricsDF, Seq("id"))//.drop("id")
        combinedDF.show(combinedDF.count.toInt, false)

        // Find row with max r2
        println("*** Max R2 ***")
        val maxR2 : Double = combinedDF.agg(max(combinedDF("r2")).as("maxr2")).collect()(0).getDouble(0)
        combinedDF.filter($"r2" === maxR2).show(false)

        // Display row with min RMSE
        println("*** Min RMSE ***")
        val minRMSE : Double = combinedDF.agg(min($"rmse")).collect()(0).getDouble(0)
        combinedDF.filter($"rmse" === minRMSE).show(false)

        // Display row with min MAE
        println("*** Min MAE ***")
        val minMAE : Double = combinedDF.agg(min($"mae")).collect()(0).getDouble(0)
        combinedDF.filter($"mae" === minMAE).show(false)

    }


    def randomForestFeatureImportance(spark: SparkSession, df: DataFrame, preds: Array[String], tv: String, seed: Long) : Unit = {

        println("********************************* Random Forest Feature Importance *********************************")
        println("... explain algorithm ...")

        // Create training and testing dataframes
        val Array(rf_training, rf_testing) = df.randomSplit(Array(0.7, 0.3), seed = seed)

        // target/response/outcome/result varaible
        val target_variable : String = tv
        val target_prediction : String = "pred_" + target_variable

        // A transformer that merges multiple columns into a vector column
        // Consists of columns (predictor/regressor variables) that will be used in ml algorithm
        val rf_assembler = new VectorAssembler()
          .setInputCols(preds)
          .setOutputCol("rawFeatures")

        // Normalizing each feature to have unit standard deviation and/or zero mean
//        val rf_scaler = new StandardScaler()
//          .setInputCol("rawFeatures")
//          .setOutputCol("scaledFeatures")
//          .setWithStd(true)
//          .setWithMean(true)
        //// --- Scaling the data causes an error. I don't know why - 12/12/17 ---

        // Random Forest Regression
        // View all hyperparameter options in the docs: https://spark.apache.org/docs/2.1.0/api/scala/index.html#org.apache.spark.ml.regression.RandomForestRegressionModel
        val rf_algorithm = new RandomForestRegressor()
          .setLabelCol(target_variable)
          .setPredictionCol(target_prediction)
          .setFeaturesCol("rawFeatures")
          .setMaxBins(32)   // default: 32; Maximum number of bins used for discretizing continuous features and for choosing how to split on features at each node. More bins give higher granularity.
          .setMaxDepth(5)   // default: 5; Maximum depth of the tree (>= 0). E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.
          .setNumTrees(20)  // default: 20; Number of trees to train (>= 1). If 1, then no bootstrapping is used. If > 1, then bootstrapping is done. !!!ALWAYS DO BOOTSTRAPPING!!!
          .setFeatureSubsetStrategy("onethird")   // The number of features to consider for splits at each tree node. If numTrees > 1 (forest), set to "sqrt" for classification and to "onethird" for regression.
          .setSeed(seed)

        // Create pipeline
        val rf_pipeline = new Pipeline()
          .setStages(Array(rf_assembler, rf_algorithm))

        // Apply pipeline to training dataframe
        // Creates a PipelineModel
        val rf_pipelineModel = rf_pipeline.fit(rf_training)

        val rf_model = rf_pipelineModel.stages(1).asInstanceOf[RandomForestRegressionModel]

        println("### Sorted Relative Feature Importance ###")
        // https://stackoverflow.com/a/47046948
        val featImport = rf_model.featureImportances
        val srfi = preds.zip(featImport.toArray).sortBy(-_._2)
        // Convert to dataframe
        // https://stackoverflow.com/a/37154570
        import spark.implicits._
        // Zip both pairs of the array to form a dataframe
        val test = spark.sparkContext.parallelize(srfi).toDF("predictor", "score")
        test.show()
        // Print array[(string, double)]
        // srfi.foreach(println)
        //println(rf_model.featureImportances.toString)

        println(" ")
        println("### Num of Features ###")
        println(rf_model.numFeatures)

        // println("### Full Description of Model ###")
        //// Notice: this prints the entire random forest decision tree
        //// Long output
        // println(rf_model.toDebugString)

        // println("### Summary of the Model ###")
        // println(rf_model.toString())
        // No important information given

        println("### Total Num of Nodes ###") // summed over all trees in the ensemble
        println(rf_model.totalNumNodes)

        // println("### Tree Weights ###")
        // println(rf_model.treeWeights.mkString(", "))
        // No important information given

    }

    def main(args: Array[String]) : Unit = {

        // Record clock time of code execution
        val t1 = System.nanoTime()

        /* =============================================================================
        The entry point to Spark is the SparkSession class.
        The SparkSession creates a SparkContext, if one doesn't already exist.
        The SparkContext tells Spark how to access a cluster.
        ================================================================================
        */

        // Create Spark Session
        val spark = SparkSession
          .builder()
          .getOrCreate()

        // Reads parquet data saved in HDFS as a dataframe
        // This data has already been processed, combined, filtered, and cleaned and in pqruet format
        val df = spark.read.parquet("/data_lake/some/folder")


        //// Correlation & Covariance Analysis ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        // -- Note -- Spark doesn't have a native correlation or covariance matrix function. So I made the next best thing.

        /// List of all statistics functions that can be applied to a dataframe
        // https://spark.apache.org/docs/2.1.0/api/scala/index.html#org.apache.spark.sql.DataFrameStatFunctions

        // Features to correlate
        val feature_columns : Array[String] = Array("target_column_name", "pred_col_1", "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6")

        // Filter dataframe to only feature_columns
        val corr_data : DataFrame = df.select(feature_columns.head, feature_columns.tail: _*)

        // Display Correlation between varibles
        corr_or_covMatrix(corr_data, "corr")

        // Display Covariance between varibles
        corr_or_covMatrix(corr_data, "cov")





        // target/response/outcome/result varaible
        val target_variable : String = "target_column_name"

        val input_cols : Array[String] = Array("pred_col_1", "pred_col_2", "pred_col_3", "pred_col_4", "pred_col_5", "pred_col_6"

        //// Find Principal Components using PCA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        principalComponentAnalysis(df, input_cols, 5)



        //// All Possible Combinations Regression ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        allPossibleCombinationsRegression(spark, df, input_cols, "target_column_name")



        //// Random Forest Regression Feature Importance ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        randomForestFeatureImportance(spark, df, input_cols, "target_column_name", 1069)




        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }

}
