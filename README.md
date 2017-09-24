# Gradient-descent

LINEAR REGRESSION WITH GRADIENT DESCENT ALGORITHM
------------------------------------------------------------------------------------------------------
File name: Readme.txt
Created by: Maj Amit Pathania
Roll No: 163054001
Date:05 Mar 17
------------------------------------------------------------------------------------------------------
1. To run: python3.5 main.py

2.	Feauture engineering:
In order to find relation between features and output (MEDV), first graphs were plotted between each feature and output which gave the rough idea about how output varies with each feature.It also gave idea about features which are non uniformly distributed.The cross correlation between features was also calculated to check inter dependencies between features.After that,graphs between higher order(square and cube) of features, exponential and log of features and output were plotted to check the non linear dependence between output and features. The data was also normalised using mean and standard deviation.
Based on these graphs and trials,following feature engineering was done:
	(a)	Log was taken for attribute(LSTAT) first without normalisation as taking log after normalisation was giving nan values due to 0 values.
	(b)	Covariance matrix was calculated wherein product of each attribute was taken with every other attribute.
	(c)	Square of attributes [0,1,2,3,4,5,6,8,11,12]
	(d)	Cube of attributes [0,1,2,4,6,7,9,10,11,12]
	(e)	Exponential of attributes [12,2,5]
	(f)	Outliners were removed. The rows having any value greater than 7.5 or less than -7.5 after normalisation were removed.

3.	Selection of lambda:
Different values of lambda were tried to find optimal solution.For higher values of lambda (more than .01),there is very less difference between error in k and k+1 iteration step ie very slow convergence rate towards optimal solution.Further, for the higher values of lambda,the error in validation data is too high.Also, high values of lambda increased the error for lp norm solutions.The value of lambda helps us to control overfitting.Increasing lambda results in less overfitting but also greater bias as seen in results.

4.	Selection of step size:
Different values of lambda and step size were tried to find optimal solution.Without normalising the data, the step size was required to be kept low(.000001) since the higher values of step size in case of unnormalised data was giving overflow(nan)values which had very slow convergence rate thus requiring many iterations.However,after normalisation the value of step size was coming in the range of .002-.0045. Hence, faster convergence to optimal solution without jumping the optimal point.

5.	Cross validation: was also tried by dividing data into training and validation data randomly with 80% training and 20% validation data.k- fold cross validation was also tried with k=3 and 7. However, there was not any significant improvement in the result. Hence, same was not included in final submission file.

6.	Implementation:
	(a)	Log was taken for attribute(LSTAT) first based on the feature vs output graphs.All attributes are then normalised using mean and standard deviation(attr-mean/std dev).
	(b)	Covariance matrix was calculated wherein product of each attribute was taken with every other attribute.This matrix is then appended into feature matrix.
	(c)	Sqaures, cubes and expononential of some features were taken based on the feature vs output graphs and appended into the feature matrix.
	(d)	Outliners were removed by checking that all normalised values lie within -7.5 to 7.5 range.
	(e)	Step size and lambda for gradient descent was tuned by taking different values as explained in serial 3 and 4.
	(f)	Gradient solution for Lp norm was also calculated at p=1.25,1.5 and 1.75

7.	Parameters for gradient descent:
	step_size = 0.0045 # step size
	conv_condn=0.000001 #condition for convergence. In case, error difference between k and k+1 step is less than this value,loop will terminate
	max_iter=30000 #maximum number of iterations
	lambda=.0005 #value of lambda

8. Comparison between result of L2 closed form solution and L2 gradient descent:The training data error with closed form solution was 4.610 and test data error was 5.25294. However, training data error using gradient descent was 2.61748431042 and test data error was  2.91193 after 30000 iterations. The closed form solution can be computed faster than gradient descent alogorithm as gradient descent requires time to reach to optimal solution(ie it needs many iterations to arrive at optimal solution and sometimes it may or may not even converge at optimal point giving nan values). However, as the number of iterations increase the gradient descent soltuion becomes becomes more closer to optimal soltuion and error keeps descreasing (keeping step size in control).Closed form solution exists in this case but if in the case where inverse of (phitranspose*phi) doesnt exist, the closed form solution may not exist and then we have to use gradient descent method to calculate optimal solution.

9. The w value was also calculated using gradient descent algorithm for Lp norm at p=1.25, 1.5 and 1.75.The value of lambda,step size and number of iterations was kept same as for gradient descent to compare the results under same parameters.Higher values of p gave lower error value as viewed from the results.
									Training data error			Test Data Error
				L2 Norm 			2.61748431042				2.91193
				Lp norm p=1.25		2.61473034247				2.90681
				Lp norm p=1.5		2.61503134733				2.90675
				Lp norm p=1.75		2.61547050902				2.90670
