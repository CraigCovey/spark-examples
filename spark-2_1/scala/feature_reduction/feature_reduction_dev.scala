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

object FeatureReductionDev {

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


    // Add backticks to string
    def sanitize(input: String): String = s"`$input`"


    // Prints correlation between every column combination pair in a dataframe
    // Helps user decide if correlation is significant
    def correlationOutput(ss: SparkSession, df: DataFrame, alpha: Double, p: String = "two-sided") : DataFrame = {

        println("********************************* Pearson Correlation *********************************")
        println("  ")
        println("Correlation refers to a technique used to measure the relationship between two variables.")
        println("When two things are correlated, it means that they vary together. ")
        print("Correlation coefficients can vary numerically between 0.0 and 1.0. The closer the correlation is to 1.0, " +
          "the stronger the relationship between the two variables. A correlation of 0.0 indicates the absence of a relationship. " +
          "If the correlation coefficient is –0.80, which indicates the presence of a strong relationship. ")
        print("A positive correlation coefficient means that as variable 1 increases, variable 2 increases, and conversely, as variable 1 decreases, variable 2 decreases. " +
          "A negative correlation means that as variable 1 increases, variable 2 decreases and vice versa. ")
        // http://statisticalconcepts.blogspot.com/2010/04/interpretation-of-correlation.html
        println("  ")
        println("  ")
        print("For correlations, the effect size is called the coefficient of determination and is defined as r2. " +
          "The coefficient of determination can vary from 0 to 1.00 and indicates that the proportion of variation in the scores can be " +
          "predicted from the relationship between the two variables. For r  = -0.80 the coefficient of determination is 0.65, " +
          "which means that 65% of the variation in variable 1 can be predicted from the relationship between variable 2 and " +
          "variable 1. (Conversely, 35% of the variation in variable 1 cannot be explained.)")
        // http://statisticalconcepts.blogspot.com/2010/04/interpretation-of-correlation.html
        println("  ")
        println("  ")
        print("A test statistic is a standardized value that is calculated from sample data during a hypothesis test. " +
          "A test statistic measures the degree of agreement between a sample of data and the null hypothesis. Its observed " +
          "value changes randomly from one random sample to a different sample. A test statistic contains information about " +
          "the data that is relevant for deciding whether to reject the null hypothesis. ")
        // http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/basic-statistics/inference/supporting-topics/basics/what-is-a-test-statistic/
        println("   ")
        println("  ")
        println("In hypothesis testing, a critical value is a point on the test distribution that is compared to the test " +
          "statistic to determine whether to reject the null hypothesis.")
        // http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/basic-statistics/inference/supporting-topics/basics/what-is-a-critical-value/

        import ss.implicits._   // Needed for .toDF() & to use $"column_name" to reference a column in a dataframe

        // Mutable array for string column names
        val mArrCols = ArrayBuffer[String]()

        // Mutable array for correlation values
        val mArrCorrCov = ArrayBuffer[Double]()


        // Array of dataframe column names
        val names = df.columns
        // Split column names into unique pairs as an array of arrays
        val paired_cols = names.mkString(",").split(",").combinations(2).toArray

        // Loop that prints the Pearson correlation between every column pair in the dataframe
        for (i <- paired_cols.indices) {

            // First column name of ith pair
            val p_0 = paired_cols(i)(0)
            // Second column name of ith pair
            val p_1 = paired_cols(i)(1)


            // Pearson Correlation between two columns
            val cor = df.stat.corr(p_0, p_1)
            //println(s"$p_0 vs $p_1 : corr = $cor")

            // Collect columns in a mutable nested array
            mArrCols += s"$p_0 vs $p_1"
            // Collect correlation value in a mutable nested array
            mArrCorrCov += cor


        }

        println("   ")

        // Store all the values in a dataframe and sort in desc order
        val arrCorrCov = mArrCorrCov.toArray
        val arrCols = mArrCols.toArray


        // Create dataframe of two arrays
        val corrDF = ss.sparkContext.parallelize(arrCols zip arrCorrCov).toDF("Columns", "Corr")

        // Calculate the effect size of the correlation (Coefficient of Determination)
        // http://statisticalconcepts.blogspot.com/2010/04/interpretation-of-correlation.html
        // http://janda.org/c10/Lectures/topic06/L24-significanceR.htm

        // The coefficient of determination can vary from 0 to 1.00 and indicates that the proportion of
        // variation in the scores can be predicted from the relationship between the two variables.

        // Example, if r = 0.50 and Coefficient of determiination (r^2) = 0.25
        // Means that 25% of variance in political stability is "explained" by literacy rate

        val df_2 = corrDF
          .withColumn("CoefDet", pow($"Corr", 2.0))

        // Calculate the test statistic
        // http://statisticalconcepts.blogspot.com/2010/04/interpretation-of-correlation.html
        // http://janda.org/c10/Lectures/topic06/L24-significanceR.htm

        // n = number of rows in original dataframe
        val n : Double = df.count.toDouble
        val degFreedom = n - 2.0
        println("Degrees of Freedom = " + degFreedom)
        println("Alpha = " + alpha)
        println("Test: " + p)
        var prob : Double = 0.0

        if (p == "two-sided") {
            prob = 1.0 - alpha / 2.0
            println("Probability P = " + prob)
        } else if (p == "upper one-sided") {
            prob = 1.0 - alpha
            println("Probability P = " + prob)
        } else if (p == "lower one-sided") {
            prob = 1.0 - alpha
            println("Probability P = " + prob)
        } else {
            System.exit(1)
        }


        // If degrees of freedom > 100 then use the infinity critical values.
        // If degrees of freedom <= 100 then use the critical values from 1 to 100.
        val t : DataFrame = if (degFreedom > 100) {
            // Creates a dataframe of critical values with degrees of freedom equal to infinity
            // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
            Seq(
                (1.282, 1.645, 1.960, 2.326, 2.576, 3.090)
            ).toDF("df", "0.9", "0.95", "0.975", "0.99", "0.995", "0.999")
        } else {
            // Creates a dataframe of critical values with degrees of freedome from 1 to 100
            students_t_Table(ss)
        }


        // Must wrap column names in backticks that have a period in the name
        // https://stackoverflow.com/a/42698510
        // https://stackoverflow.com/a/42671494

//        t.filter($"df" === degFreedom)
//          .select(sanitize(prob.toString))
//          .show()

        val critValue : Double = t
          .filter($"df" === degFreedom)
          .select(sanitize(prob.toString))
          .head().getDouble(0)

        println("Critical Value (from t Table): " + critValue)

        println("   ")
        println("1. For a two-sided test, find the column corresponding to probability 1-α/2 and reject the null hypothesis " +
          "if the absolute value of the test statistic is greater than the critical value.")
        println("   ")
        println("2. For an upper, one-sided test, find the column corresponding to probability 1-α and reject the null hypothesis " +
          "if the test statistic is greater than the critical value.")
        println("   ")
        println("3. For a lower, one-sided test, find the column corresponding to probability 1-α and reject the null hypothesis " +
          "if the test statistic is less than the negative of the critical value.")
        println("   ")

        // Calculate test statistic and absolute value of test statistic
        val df_3 = df_2
          .withColumn("tValue", $"Corr" * sqrt( (lit(n) - lit(2)) / (lit(1.0) - $"CoefDet") ) )
          .withColumn("abs_tValue", abs($"Corr" * sqrt( (lit(n) - lit(2)) / (lit(1.0) - $"CoefDet") )) )
          // Sort dataframe by Corr
          .orderBy($"Corr".desc)

        // Hypothesis Testing
        val df_4 : DataFrame = if (p == "two-sided") {
            df_3.withColumn("Significance", when($"abs_tValue" > critValue, "Reject Null Hyp").otherwise("Fail to Reject"))
        } else if (p == "upper one-sided") {
            df_3.withColumn("Significance", when($"tValue" > critValue, "Reject Null Hyp").otherwise("Fail to Reject"))
        } else {
            df_3.withColumn("Significance", when($"tValue" < critValue, "Reject Null Hyp").otherwise("Fail to Reject"))
        }

        df_4

    }



    def students_t_Table(spark: SparkSession) : DataFrame = {

        //// Critical Values of the Student's t Distribution

        // http://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm

        import spark.implicits._    // for .toDf() method

        // t table
        val df : DataFrame = Seq(
            (1.0 , 3.078, 6.314, 12.706, 31.821, 63.657, 318.313),
            (2.0 , 1.886, 2.92, 4.303, 6.965, 9.925, 22.327),
            (3.0 , 1.638, 2.353, 3.182, 4.541, 5.841, 10.215),
            (4.0 , 1.533, 2.132, 2.776, 3.747, 4.604, 7.173),
            (5.0 , 1.476, 2.015, 2.571, 3.365, 4.032, 5.893),
            (6.0 , 1.44, 1.943, 2.447, 3.143, 3.707, 5.208),
            (7.0 , 1.415, 1.895, 2.365, 2.998, 3.499, 4.782),
            (8.0 , 1.397, 1.86, 2.306, 2.896, 3.355, 4.499),
            (9.0 , 1.383, 1.833, 2.262, 2.821, 3.25, 4.296),
            (10.0 , 1.372, 1.812, 2.228, 2.764, 3.169, 4.143),
            (11.0 , 1.363, 1.796, 2.201, 2.718, 3.106, 4.024),
            (12.0 , 1.356, 1.782, 2.179, 2.681, 3.055, 3.929),
            (13.0 , 1.35, 1.771, 2.16, 2.65, 3.012, 3.852),
            (14.0 , 1.345, 1.761, 2.145, 2.624, 2.977, 3.787),
            (15.0 , 1.341, 1.753, 2.131, 2.602, 2.947, 3.733),
            (16.0 , 1.337, 1.746, 2.12, 2.583, 2.921, 3.686),
            (17.0 , 1.333, 1.74, 2.11, 2.567, 2.898, 3.646),
            (18.0 , 1.33, 1.734, 2.101, 2.552, 2.878, 3.61),
            (19.0 , 1.328, 1.729, 2.093, 2.539, 2.861, 3.579),
            (20.0 , 1.325, 1.725, 2.086, 2.528, 2.845, 3.552),
            (21.0 , 1.323, 1.721, 2.08, 2.518, 2.831, 3.527),
            (22.0 , 1.321, 1.717, 2.074, 2.508, 2.819, 3.505),
            (23.0 , 1.319, 1.714, 2.069, 2.5, 2.807, 3.485),
            (24.0 , 1.318, 1.711, 2.064, 2.492, 2.797, 3.467),
            (25.0 , 1.316, 1.708, 2.06, 2.485, 2.787, 3.45),
            (26.0 , 1.315, 1.706, 2.056, 2.479, 2.779, 3.435),
            (27.0 , 1.314, 1.703, 2.052, 2.473, 2.771, 3.421),
            (28.0 , 1.313, 1.701, 2.048, 2.467, 2.763, 3.408),
            (29.0 , 1.311, 1.699, 2.045, 2.462, 2.756, 3.396),
            (30.0 , 1.31, 1.697, 2.042, 2.457, 2.75, 3.385),
            (31.0 , 1.309, 1.696, 2.04, 2.453, 2.744, 3.375),
            (32.0 , 1.309, 1.694, 2.037, 2.449, 2.738, 3.365),
            (33.0 , 1.308, 1.692, 2.035, 2.445, 2.733, 3.356),
            (34.0 , 1.307, 1.691, 2.032, 2.441, 2.728, 3.348),
            (35.0 , 1.306, 1.69, 2.03, 2.438, 2.724, 3.34),
            (36.0 , 1.306, 1.688, 2.028, 2.434, 2.719, 3.333),
            (37.0 , 1.305, 1.687, 2.026, 2.431, 2.715, 3.326),
            (38.0 , 1.304, 1.686, 2.024, 2.429, 2.712, 3.319),
            (39.0 , 1.304, 1.685, 2.023, 2.426, 2.708, 3.313),
            (40.0 , 1.303, 1.684, 2.021, 2.423, 2.704, 3.307),
            (41.0 , 1.303, 1.683, 2.02, 2.421, 2.701, 3.301),
            (42.0 , 1.302, 1.682, 2.018, 2.418, 2.698, 3.296),
            (43.0 , 1.302, 1.681, 2.017, 2.416, 2.695, 3.291),
            (44.0 , 1.301, 1.68, 2.015, 2.414, 2.692, 3.286),
            (45.0 , 1.301, 1.679, 2.014, 2.412, 2.69, 3.281),
            (46.0 , 1.3, 1.679, 2.013, 2.41, 2.687, 3.277),
            (47.0 , 1.3, 1.678, 2.012, 2.408, 2.685, 3.273),
            (48.0 , 1.299, 1.677, 2.011, 2.407, 2.682, 3.269),
            (49.0 , 1.299, 1.677, 2.01, 2.405, 2.68, 3.265),
            (50.0 , 1.299, 1.676, 2.009, 2.403, 2.678, 3.261),
            (51.0 , 1.298, 1.675, 2.008, 2.402, 2.676, 3.258),
            (52.0 , 1.298, 1.675, 2.007, 2.4, 2.674, 3.255),
            (53.0 , 1.298, 1.674, 2.006, 2.399, 2.672, 3.251),
            (54.0 , 1.297, 1.674, 2.005, 2.397, 2.67, 3.248),
            (55.0 , 1.297, 1.673, 2.004, 2.396, 2.668, 3.245),
            (56.0 , 1.297, 1.673, 2.003, 2.395, 2.667, 3.242),
            (57.0 , 1.297, 1.672, 2.002, 2.394, 2.665, 3.239),
            (58.0 , 1.296, 1.672, 2.002, 2.392, 2.663, 3.237),
            (59.0 , 1.296, 1.671, 2.001, 2.391, 2.662, 3.234),
            (60.0 , 1.296, 1.671, 2.00, 2.39, 2.66, 3.232),
            (61.0 , 1.296, 1.67, 2.00, 2.389, 2.659, 3.229),
            (62.0 , 1.295, 1.67, 1.999, 2.388, 2.657, 3.227),
            (63.0 , 1.295, 1.669, 1.998, 2.387, 2.656, 3.225),
            (64.0 , 1.295, 1.669, 1.998, 2.386, 2.655, 3.223),
            (65.0 , 1.295, 1.669, 1.997, 2.385, 2.654, 3.22),
            (66.0 , 1.295, 1.668, 1.997, 2.384, 2.652, 3.218),
            (67.0 , 1.294, 1.668, 1.996, 2.383, 2.651, 3.216),
            (68.0 , 1.294, 1.668, 1.995, 2.382, 2.65, 3.214),
            (69.0 , 1.294, 1.667, 1.995, 2.382, 2.649, 3.213),
            (70.0 , 1.294, 1.667, 1.994, 2.381, 2.648, 3.211),
            (71.0 , 1.294, 1.667, 1.994, 2.38, 2.647, 3.209),
            (72.0 , 1.293, 1.666, 1.993, 2.379, 2.646, 3.207),
            (73.0 , 1.293, 1.666, 1.993, 2.379, 2.645, 3.206),
            (74.0 , 1.293, 1.666, 1.993, 2.378, 2.644, 3.204),
            (75.0 , 1.293, 1.665, 1.992, 2.377, 2.643, 3.202),
            (76.0 , 1.293, 1.665, 1.992, 2.376, 2.642, 3.201),
            (77.0 , 1.293, 1.665, 1.991, 2.376, 2.641, 3.199),
            (78.0 , 1.292, 1.665, 1.991, 2.375, 2.64, 3.198),
            (79.0 , 1.292, 1.664, 1.99, 2.374, 2.64, 3.197),
            (80.0 , 1.292, 1.664, 1.99, 2.374, 2.639, 3.195),
            (81.0 , 1.292, 1.664, 1.99, 2.373, 2.638, 3.194),
            (82.0 , 1.292, 1.664, 1.989, 2.373, 2.637, 3.193),
            (83.0 , 1.292, 1.663, 1.989, 2.372, 2.636, 3.191),
            (84.0 , 1.292, 1.663, 1.989, 2.372, 2.636, 3.19),
            (85.0 , 1.292, 1.663, 1.988, 2.371, 2.635, 3.189),
            (86.0 , 1.291, 1.663, 1.988, 2.37, 2.634, 3.188),
            (87.0 , 1.291, 1.663, 1.988, 2.37, 2.634, 3.187),
            (88.0 , 1.291, 1.662, 1.987, 2.369, 2.633, 3.185),
            (89.0 , 1.291, 1.662, 1.987, 2.369, 2.632, 3.184),
            (90.0 , 1.291, 1.662, 1.987, 2.368, 2.632, 3.183),
            (91.0 , 1.291, 1.662, 1.986, 2.368, 2.631, 3.182),
            (92.0 , 1.291, 1.662, 1.986, 2.368, 2.63, 3.181),
            (93.0 , 1.291, 1.661, 1.986, 2.367, 2.63, 3.18),
            (94.0 , 1.291, 1.661, 1.986, 2.367, 2.629, 3.179),
            (95.0 , 1.291, 1.661, 1.985, 2.366, 2.629, 3.178),
            (96.0 , 1.29, 1.661, 1.985, 2.366, 2.628, 3.177),
            (97.0 , 1.29, 1.661, 1.985, 2.365, 2.627, 3.176),
            (98.0 , 1.29, 1.661, 1.984, 2.365, 2.627, 3.175),
            (99.0 , 1.29, 1.66, 1.984, 2.365, 2.626, 3.175),
            (100.0 , 1.29, 1.66, 1.984, 2.364, 2.626, 3.174)
        ).toDF("df", "0.9", "0.95", "0.975", "0.99", "0.995", "0.999")

        df

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

        // Display top Predictors ordered by r2
        println("*** Top R2 ***")
        combinedDF
          .orderBy($"r2".desc)
          .show()

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
        // corr_or_covMatrix(corr_data, "corr")

        // Display Covariance between varibles
        corr_or_covMatrix(corr_data, "cov")

        // Prints correlation between every column combination pair in a dataframe
        val c = correlationOutput(spark, corr_data, 0.05, "two-sided")
        c.show(c.count.toInt, false)

        println("Note:")
        println("   A relationship can be strong and yet not significant.")
        println("   Conversely, a relationship can be weak but significant.")
        println("      The key factor is the size of the sample.")
        print("   For small samples, it is easy to produce a strong correlation by chance and one must pay attention to signficance to keep from jumping to conclusions. " +
          "For large samples, it is easy to achieve significance, and one must pay attention to the strength of the correlation to determine if the relationship explains very much.")
        println("   ")




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
