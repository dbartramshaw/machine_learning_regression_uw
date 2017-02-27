#################################################################
# REGRESSION - WASHINGTON MACHINE LEARNING COURSE
#################################################################

# neogi - just numpy functions BEST ONE
# ramaranjanruj - uses sklearn
# corylstewart - uses graphlab
# justindomingue - very high level only functions


import pandas as pd
import sframe
import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt

# import data
sales = sframe.SFrame('/Users/davidbartram-shaw/Machine Learning Course/Course 2 - Regression/kc_house_data.gl/')
#sales=sframe.SFrame.to_dataframe(sales)

# Test/Train split
train_data,test_data = sales.random_split(.8,seed=0)
#train_data, test_data = train_test_split(sales, test_size=0.33, random_state=42)

#################################################################
# MULTIPLE LINEAR REGRESSION
#################################################################


# optimize matrix operations
def get_numpy_data(data_sframe, features, output):
    # Add constant for w0
    data_sframe['constant'] = 1
    # Chosen features only
    features = ['constant'] + features
    features_df = data_sframe[features]
    feature_matrix = features_df.to_numpy()
    # assign outout as array
    output_array = data_sframe[output].to_numpy()
    return(feature_matrix, output_array)

# Example
features = ['sqft_living']
output='price'
(example_features, example_output) = get_numpy_data(sales, features, output)



# Predicting output given regression weights (co-efficiencts)
def predict_output(feature_matrix, weights):
    """
    The predictions from all the observations is just the RIGHT dot product
    between the features matrix and the weights vector
    """
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

#Example
my_weights = np.array([1., 1.])
test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] # should be 1181.0
print test_predictions[1] # should be 2571.0



# Computing the Derivative
def feature_derivative(errors, feature):
    """
        computing the derivative of the regression cost function
        cost function is the sum over all points of squared difference between an observed output and a predicted output.
        Since the derivative of a sum is the sum of the derivatives we compute the derivative for a single point and then sum over all points.
        We can write the squared difference between the observed output and predicted output for a single point as follows:

        (w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)^2

        Where we have k features and a constant.
        So the derivative with respect to weight w[i] by the chain rule is:

        2*(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)* [feature_i]

        The term inside the paranethesis is just the error (difference between prediction and output). So we can re-write this as:

        2*error*[feature_i]

        That is, the derivative for the weight for feature i is the sum (over all points) of 2 times the product of the error and the feature itself.
        In the case of the constant then this is just twice the sum of the errors!
        The sum of the product of two vectors is just twice the dot product of the two vectors.
        Therefore the derivative for the weight for feature_i is just two times the dot product between the values of feature_i and the current errors.
    """
    derivative = 2*np.dot(errors, feature)
    return(derivative)

# Example
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print derivative
print -np.sum(example_output)*2 # should be the same as derivative




# Gradient Descent¶
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    """
        Given a starting point we update the current weights by moving in the negative gradient direction.
        Recall that the gradient is the direction of increase and therefore negative gradient is the direction of decrease and we're trying to minimize a cost function.
        The amount by which we move in the negative gradient direction is called the 'step size'.
        We stop when we are 'sufficiently close' to the optimum.
        We define this by requiring that the magnitude (length) of the gradient vector to be smaller than a fixed 'tolerance'.
    """
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predictions = predict_output(feature_matrix, weights)
        errors = predictions-output
        # initialize the gradient sum of squares
        gradient_sum_squares = 0
        # while we haven't reached the tolerance yet, update each feature's weight, loop over each weight
        for i in range(len(weights)): #
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += (derivative**2)
            # subtract the step size times the derivative from the current weight
            weights[i] -= derivative * step_size
        # compute the square-root of the gradient sum of squares to get the gradient magnitude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


#Example
simple_features = ['sqft_living']
simple_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, simple_features, simple_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

test_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)













##
