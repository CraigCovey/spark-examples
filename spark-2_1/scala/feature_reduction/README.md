# Spark Feature Reduction / Selection
Using Spark 2.1 framework in the Scala programming language

_By Craig Covey - 2018_

This section outlines my Spark Scala feature (column) reduction workflow:

* Correlation of features (use hypothesis testing to determine if correlation is significant)
* Covariance of features
* Principal Component Analysis (PCA)
* All Possible Combination Regression _(My custom code of taking every combination of features and performing multiple linear regression and recording the rsquared and RMSE. Using the result to identify which combination of features have the highest rsquared and lowest RMSE.)_
* Random Forest Feature Importance
* Using Hypothesis Testing to Identify Feature Significance:

	* Test for significance of regression
	* _t_ Test (test on individual regression coefficients)
	* _F_ Test (test of subsets of regression coefficients)

This code consists of two Scala files. _feature\_reduction\_dev.scala_ contains the correlation, covariance, PCA, All Possible Combination Regression, and the Random Forest Feature Importance code. While _hypothesis\_testing.scala_ has all of the statistics code.


_Code work in progress_