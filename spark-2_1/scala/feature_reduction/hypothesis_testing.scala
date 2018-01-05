package com.datascience.craig

import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg._
import breeze.numerics._


object StatsPackage {

    // Function to convert dataframe to Breeze DenseMatrix
    def dfToMatrix(df: DataFrame) : DenseMatrix[Double] = {

        // ALL data in dataframe must be double!!!!

        /*
        Articles:
            https://stackoverflow.com/a/44082786
            https://stackoverflow.com/a/40003312
            https://stackoverflow.com/a/44533409
            https://stackoverflow.com/a/41943635
         */

        // Number of rows of dataframe
        val r = df.count().toInt

        // Number of columns of dataframe
        val c = df.columns.length

        // Convert entire dataframe into one long array
        // Uses RDD, so it caputres data by rows (not by columns)
        val arr = df.rdd.collect()
          .map(_.toSeq)
          .array
          .map(x => x.toArray).flatten
          .map(y => y.toString).map(z => z.toDouble)

        // Create Breeze DenseMatrix
        // DenseMatrix(rows, columns, data)
        // Switched the rows and columns because data is brought in by rows
        val mat = new DenseMatrix(c, r, arr)

        // Transpose matrix
        // Matrix will now be origional shape as dataframe
        val mtx = mat.t

        mtx
    }

    def matrixAppendDouble(d: Double, m: DenseMatrix[Double]) : DenseMatrix[Double] = {

        // Rows of DenseMatrix
        val rows = m.rows

        // Create DenseVector of Double
        val vec = DenseVector.fill(rows){d}

        // Convert DenseVector into DenseMatrix of n rows and 1 column
        val col = new DenseMatrix(rows, 1, vec.toArray)

        // Append DenseMatrix of one column to orginal DenseMatrix
        val X = DenseMatrix.horzcat(col, m)

        X
    }


    def HatMatrix(X: DenseMatrix[Double]) : DenseMatrix[Double] = {

        //// H = hat matrix
        // H = X * (X' * X)^-1 * X'
        // H = n X n matrix, where n is the number of rows in the data set
        // H dimensions = [m x n] * ([n x m] * [m x n])^-1 * [n x m]
        // H dimensions = [m x n] * ([n x n])^-1 * [n x m]
        // H dimensions = [m x n] * [n x n] * [n x m]
        // H dimensions = [m x n] * [n x m]
        // H dimensions = [m x m]

        val H = X * inv(X.t * X) * X.t

        H
    }

    def SS_R(y: DenseMatrix[Double], H: DenseMatrix[Double]) : Double = {

        // SS_R = Sum of Squares Regression

        // SS_R = y' * (H-(1/n) * J) * y
        // where H = X * (X' * X)^-1 * X'
        // and n = number of rows in dataset
        // and J = n X n square matrix of ones

        // SS_R = y'*(H-(1/n)*J)*y
        // SS_R dimensions = [1 x n] * ([n x n] - (1/n) * [n x n]) * [n x 1]
        // SS_R dimensions = [1 x n] * ([n x n] - [n x n]) * [n x 1]
        // SS_R dimensions = [1 x n] * [n x n] * [n x 1]
        // SS_R dimensions = [1 x n] * [n x 1]
        // SS_R dimensions = [1 x 1]

        // Number of rows in dataframe
        val n = y.rows

        val J : DenseMatrix[Double] = DenseMatrix.ones[Double](n, n)

        // You must divide with a double (not integer)
        val ssr = y.t * (H - ((1.0 / n) * J)) * y

        // ssr is a DenseMatrix of 1 x 1
        // Select the first (and only) element to return a double
        ssr(0, 0)

    }


    def SS_E(y: DenseMatrix[Double], H: DenseMatrix[Double]) : Double = {

        // SS_E = Sum of Squares Error

        // SS_E = y' * (I - H) * y
        // where I = the identity matrix of order n
        // where H is the Hat matrix
        // H = X * (X' * X)^-1 * X'
        // and n = number of rows in dataset

        // SS_E dimensions = [1 x n] * ( [n x n] - [n x n] ) * [n x 1]
        // SS_E dimensions = [1 x n] * [n x n] * [n x 1]
        // SS_E dimensions = [1 x n] * [n x 1]
        // SS_E dimensions = [1 x 1]

        // Number of rows in dataframe
        val n = y.rows

        // The Identity Matrix
        val I = DenseMatrix.eye[Double](n)

        val sse = y.t * (I - H) * y

        // sse is a DenseMatrix of 1 x 1
        // // Select the first (and only) element to return a double
        sse(0, 0)

    }

    def dof_SS_E(df: DataFrame) : Double = {

        // Degrees of Freedom for SS_E

        // dof_SS_E = n - (k + 1)
        // where n is the number of rows in the dataframe
        // and k is the number of predictors (columns) in the dataframe

        val n = df.count.toDouble

        val k = df.columns.length.toDouble

        val dof = n - (k + 1)

        dof
    }


    // ========================================================================================
    def least_Squares_Regression(xDF: DataFrame, yDF: DataFrame) : DenseMatrix[Double] = {

        //// Estimating Regression Models Using Least Squares ==========================================================
        // Returns coefficients that are used to construct fitted regression model

        // Least Squares Estimate = BHat
        // BHat = ( X' * X )^-1 * X' * y
        // where X' = X transpose

        // BHat dimensions = ([n x m] * [m x n])^-1 * [n x m] * [m x 1]
        // BHat dimensions = ([n x n])^-1 * [n x m] * [m x 1]
        // BHat dimensions = [n x n] * [n x m] * [m x 1]
        // BHat dimensions = [n x m] * [m x 1]
        // BHat dimensions = [n x 1]

        // Convert dataframe of predictor columns to a DenseMatrix
        val matX = dfToMatrix(xDF)

        // Convert dataframe of outcome columns to a DenseMatrix
        val Y = dfToMatrix(yDF)

        // Add column of ones to predictors matrix
        val X = matrixAppendDouble(1.0, matX)

        // Least Squares Estimate
        val BHat = inv(X.t * X) * X.t * Y
        // results in a n by 1 matrix
        // with the first number equal to the estimated regression coefficients BHat_0,
        // the second number BHat_1,
        // etc.

        BHat
    }





    // ========================================================================================
    def test_Regression_Significance(xDF: DataFrame, yDF: DataFrame) : Double = {

        println("  ")
        println("************************** Test for Significance of Regression **************************")
        println("This test checks the significance of the whole regression model.")
        println("The test for significance of regression in the case of multiple linear regression analysis is carried out using the analysis of variance.")
        println("The test is used to check if a linear statistical relationship exists between the response variable and at least one of the predictor variables.")
        println("The statements for the hypotheses are:")
        println("     H_0 : B_1 = B_2 = ... = B_k = 0")
        println("     H_1 : B_j != 0 for at least one j")
        println("   ")
        println("The test for H_0 is carried out using the following statistic:")
        println("     F_0 = MS_R / MS_E")
        println("where MS_R = regression mean square and MS_E = error mean square")
        println("   ")
        println("If the null hypothesis, H_0, is true then the statistic F_0 follows the F distribution with k degrees of freedom in the numerator and  n - (k + 1) degrees of freedom in the denominator.")
        println("The null hypothesis, H_0, is rejected if the calculated statistic, F_0, is such that:")
        println("F_0 > (the critical value for this test, corresponding to a significance level of 0.1)")
        println("   ")


        // Convert dataframe of predictor columns to a DenseMatrix
        val matX = dfToMatrix(xDF)

        // Convert dataframe of outcome columns to a DenseMatrix
        val Y = dfToMatrix(yDF)

        // Add column of ones to predictors matrix
        val X = matrixAppendDouble(1.0, matX)



        // The Hat Matrix
        val H = HatMatrix(X)

        // Sum of Squares Regression
        val SumofSquaresReg = SS_R(Y, H)

        // Degrees of Freedom of SS_R
        val dofSS_R = xDF.columns.length.toDouble

        // Mean Square Regression = SS_R / dof
        val MS_R : Double = SumofSquaresReg / dofSS_R




        // Sum of Squares Error
        val SumofSquaresError = SS_E(Y, H)

        // Mean Square Error = SS_E / dof_SS_E
        val dofsse = dof_SS_E(xDF)
        val MS_E = SumofSquaresError / dofsse




        // The statistic to test the significance of regression can now be calculated as:
        // F_0 = MS_R / MS_E
        val F_0 : Double = MS_R / MS_E

        F_0
    }


    // ========================================================================================
    def test_Coefficients_Significance(xDF: DataFrame, yDF: DataFrame) : DenseMatrix[Double] = {

        println("  ")
        println("************************** Test on Individual Regression Coefficients (t Test) **************************")
        println("The t test is used to check the significance of individual regression coefficients in the multiple linear regression model.")
        println("Adding a significant variable to a regression model makes the model more effective, while adding an unimportant variable may make the model worse.")
        println("The hypothesis statements to test the significance of a particular regression coefficient, B_j are:")
        println("     H_0 : B_j = 0")
        println("     H_1 : B_j != 0")
        println("   ")
        println("The test statistic for this test is based on the t distribution")
        println("     T_0 = BHat_j / se(BHat_j)")
        println("The analyst would fail to reject the null hypothesis if the test statistic lies in the acceptance region:")
        println("     -t_alpha/2 < T_0 < t_alpha/2")
        println("This test measures the contribution of a variable while the remaining variables are included in the model.")
        println("For the model yHat = BHat_0 + Bhat_1 * x_1 + Bhat_2 * x_2 + Bhat_3 * x_3, if the test is carried out for BHat_1 +" +
          "then the test will check the significance of including the variable x_1 in the model that contains x_2 and x_3")
        println("   ")


        // Convert dataframe of predictor columns to a DenseMatrix
        val matX = dfToMatrix(xDF)

        // Convert dataframe of outcome columns to a DenseMatrix
        val Y = dfToMatrix(yDF)

        // Add column of ones to predictors matrix
        val X = matrixAppendDouble(1.0, matX)


        // Mean Square Error: MS_E

        // The Hat Matrix
        val H = HatMatrix(X)

        // Sum of Squares Error
        val SumofSquaresError = SS_E(Y, H)

        // Mean Square Error = SS_E / dof_SS_E
        val dofsse = dof_SS_E(xDF)
        val MS_E = SumofSquaresError / dofsse


        // Covariance Matrix of the estimated regression coefficients
        // C = MS_E * ( X' * X )^-1

        val C = MS_E * inv(X.t * X)

        //println("Covariance Matrix of X")
        //println(C.toString(Int.MaxValue, Int.MaxValue))

        // Return diagonal values of matrix as a vector
        val diagonalC = diag(C)
        // Square Root to get Est Standard Errors for BHat_1, BHat_2, etc
        val stdError = sqrt(diagonalC)
        // Convert DenseVector to DenseMatrix
        val matrixStdError = new DenseMatrix(3, 1, stdError.toArray)

        val testStatistics = least_Squares_Regression(xDF, yDF) / matrixStdError

        testStatistics
    }


    def main(args: Array[String]) : Unit = {

        // Record clock time of code execution
        val t1 = System.nanoTime()

        // Create Spark Session
        val spark = SparkSession
          .builder()
          .getOrCreate()

        import spark.implicits._    // for .toDf() method




        // Source: http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis




        //// Data ======================================================================================================

        // X
        val predictors = Seq(
            (41.9, 29.1),
            (43.4, 29.3),
            (43.9, 29.5),
            (44.5, 29.7),
            (47.3, 29.9),
            (47.5, 30.3),
            (47.9, 30.5),
            (50.2, 30.7),
            (52.8, 30.8),
            (53.2, 30.9),
            (56.7, 31.5),
            (57.0, 31.7),
            (63.5, 31.9),
            (65.3, 32.0),
            (71.1, 32.1),
            (77.0, 32.5),
            (77.8, 32.9)
        ).toDF("x1", "x2")

//        predictors.show()

        // Yield
        // Y
        val yld = Seq(
            251.3, 251.3, 248.3, 267.5, 273.0, 276.5, 270.3, 274.9, 285.0, 290.0, 297.0, 302.5, 304.5, 309.3, 321.7, 330.7, 349.0
        ).toDF("yield")

//        yld.show()

        //// Data ======================================================================================================

        //// Convert Dataframe to DenseMatrix --------------------------------------------------------------------------

        val matX = dfToMatrix(predictors)
//        println(matX.toString(Int.MaxValue, Int.MaxValue))

        val y = dfToMatrix(yld)
//        println(y.toString(Int.MaxValue, Int.MaxValue))


        // Add column of ones to predictors matrix
        val X = matrixAppendDouble(1.0, matX)
//        println(X.toString(Int.MaxValue, Int.MaxValue))






        //// Estimating Regression Models Using Least Squares ==========================================================

        println("  ")
        println("************************** Estimating Regression Models Using Least Squares **************************")
        println("Returns coefficients that are used to construct fitted regression model")

        val BHat = least_Squares_Regression(predictors, yld)
        println(BHat.toString(Int.MaxValue, Int.MaxValue))

        val BHat_1 = BHat(1,0)
        val BHat_2 = BHat(2,0)

        println("Fitted Regression Model Equation using Least Squares")
        for (i <- 0 until BHat.rows) {
            if (i == 0) {
                print("yHat = " + BHat(0, 0))
            } else {
                print(" + " + BHat(i, 0) + " * x_" + i)
            }
        }
        println("   ")


        println("========================== Hypothesis Tests in Multiple Linear Regression ==========================")
        println("This section discusses hypothesis tests on the regression coefficients in multiple linear regression.")
        println("As in the case of simple linear regression, these tests can only be carried out if it can be assumed that the random error terms +" +
          "are normally and independently distributed with a mean of zero and variance of sigma squared.")
        println("Three types of hypothesis tests can be carried out for multiple linear regression models:")
        println("     1. Test for significance of regression: This test checks the significance of the whole regression model.")
        println("     2. t test: This test checks the significance of individual regression coefficients.")
        println("     3. F test: This test can be used to simultaneously check the significance of a number of regression coefficients. It can also be used to test individual coefficients.")
        println("   ")


        //// 1. Test for Significance of Regression

        val F = test_Regression_Significance(predictors, yld)
        println("Test for Significance of Regression")
        println(F)


        //// 2. Test on Individual Regression Coefficients (t Test)

        // Test the significance of B_2
        // The null hypothesis is H_0 : B_2 = 0
        // To test the test statistics, T_0, we need to calculate the standard error.
        // The error mean square is an estimate of the variance


        val testStatistics = test_Coefficients_Significance(predictors, yld)
        println(testStatistics.toString(Int.MaxValue, Int.MaxValue))

        println(s"Test Statistics for BHat_1 = " + testStatistics(1,0))
        println(s"Test Statistics for BHat_2 = " + testStatistics(2,0))
        println("   ")


        print("Look up critical value in T Table (two-sided) with dof_SSE degrees of freedom " +
          "(http://www.statisticshowto.com/tables/t-distribution-table-two-tails/ ")
        println("With alpha = 0.1 & df = 14 the critical values are 1.761 & -1.761")

        println("For BHat_2:")
        println("The test statistic for BHat_2 of " + testStatistics(2,0) + " does not lie within the range - 1.761 to + 1,761.")
        println("Therefore, the null hypothesis, H0, is rejected and it is concluded that B2 is signficant at alpha = 0.1 with 14 degrees of freedom")

        //// 3. Test on Subsets of Regression Coefficients (Partial F Test)

        // This test can be considered to be the general form of the t test
        // This is because the test simultaneously checks the significance of including many (or even one) regression coefficients in the multiple linear regression model.
        // Adding a variable to a model increases the regression sum of squares, SS_R.
        // The test is based on this increase in the regression sum of squares. The increase in the regression sum of squares is called the extra sum of squares.
        //


        // !!! work in progress !!!



        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        // Print elapsed time of code execution
        val codeDuration = (System.nanoTime() - t1) / 6e10d
        println("   ")
        println("=== Time ===")
        println("Code ran for: " + codeDuration + " mins")


    }

}
