# Linear Regression
This MATLAB code to be for predicting the compressive strength of concrete based on various input features. 

Here's a step-by-step breakdown of what the code is doing:

* Loading the dataset from the file "ConcreteDataHCT.mat".
* Labelling the dataset and summarizing it using the "summary()" function.
* Assigning the input features (cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age) to variable "x" and the target output (concrete compressive strength) to variable "y".
* Detecting any missing values in the dataset and printing the count of missing values.
* Visualizing the dataset using correlation, box plots, and scatter plots.
* Detecting and removing any outliers using the quartile method.
* Normalizing the input features using feature scaling.
* Dividing the dataset into training, cross-validation, and test sets.
* Implementing gradient descent without regularization to find the optimal values of the parameters (theta) for each learning rate (alpha) and computing the cost on the training set. The cost is plotted against the number of iterations.
* Predicting the compressive strength of concrete for 10 random training samples and printing the predicted and actual values.
