package com.github.craig.clustering

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.hive.HiveContext


object Gaussian_Mixture {

    def main(args: Array[String]) : Unit = {

        // Record clock time of code execution
        val t1 = System.nanoTime()

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

        println("Rows in DF: " + filteredDF.count)


        // Code from https://stackoverflow.com/questions/36168536/gaussian-mixture-model-in-scala-spark-1-5-1-weights-are-always-uniformly-distrib

        val input_cols : Array[String] = Array("outcome_col", "pred_col_1", "pred_col_2", "pred_col_3"))

        // Use transforms to take predictor columns normalize them and convert them to a RDD
        // RDD will be fed to Gaussian Mixture Model
        val assembler = new VectorAssembler()
            .setInputCols(input_cols)
            .setOutputCol("features")

        val output = assembler.transform(filteredDF)

        val normalizer = new Normalizer()
            .setInputCol("features")
            .setOutputCol("normFeatures")
            .setP(1.0)

        val normalizedOutput = normalizer.transform(output)

        val temp = normalizedOutput
            .select("normFeatures")

        val results = temp.rdd.map(_.getAs[Vector]("normFeatures"))


        // Cluster the data into two classes use GaussianMixture
        val gmm = new GaussianMixture()
            .setK(8)
            .setConvergenceTol(0.01)        // <-- hyperparameter
            .setMaxIterations(100)          // <-- hyperparameter
            .setSeed(1069)
            .run(results)

        // Save model @ HDFS: /user/scovey/myGMModel
//        gmm.save(sc, "myGMModel")
        // Loads model from HDFS
//        val sameModel = GaussianMixtureModel.load(sc, "myGMModel")

        // Print input columns
        println("Input columns: " + input_cols.mkString(", "))

        // Output parameters of max-likelihood model
        println("k (number of gaussians in mixture) = " + gmm.k)
        for (i <- 0 until gmm.k) {
            println("Gaussian #: %s\nweight (weight for gaussian %s) =%f\nmu (the mean vector of the distribution) =\n%s\nsigma (the covariance matrix of the distribution) =\n%s\n" format
                (i, i, gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma.toString(8, Int.MaxValue)))
        }
        // gmm.gaussians(i).mu is a vector
        // gmm.gaussians(i).sigma is a matrix


        //// End machine learning code ###################################################################################

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")

    }


}
