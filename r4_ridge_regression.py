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
#import graphlab

# import data
sales = sframe.SFrame('/Users/davidbartram-shaw/Machine Learning Course/Course 2 - Regression/kc_house_data.gl/')
#sales=sframe.SFrame.to_dataframe(sales)

# Test/Train split
train_data,test_data = sales.random_split(.8,seed=0)
#train_data, test_data = train_test_split(sales, test_size=0.33, random_state=42)

#################################################################
# REGRESSION - RIDGE REGRESSION
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


# Predicting output given regression weights (co-efficiencts)
def predict_output(feature_matrix, weights):
    """
    The predictions from all the observations is just the RIGHT dot product
    between the features matrix and the weights vector
    """
    predictions = np.dot(feature_matrix, weights)
    return(predictions)



# Computing the Derivative¶
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    """
    computing the derivative of the regression cost function.
    Recall that the cost function is the sum over the data points of the squared difference between an observed output and a predicted output, plus the L2 penalty term.

        Cost(w) = SUM[ (prediction - output)^2 ] + l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).

    Since the derivative of a sum is the sum of the derivatives, we can take the derivative of the first part (the RSS) and add the derivative of the regularization part.

    The derivative of the RSS with respect to w[i] can be written as:

        2*SUM[ error*[feature_i] ].

    The derivative of the regularization term with respect to w[i] is:

        2*l2_penalty*w[i].

    Summing both, we get:

        2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].

    That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the feature itself, plus 2*l2_penalty*w[i].
    We will not regularize the constant. Thus, in the case of the constant, the derivative is just twice the sum of the errors (without the 2*l2_penalty*w[0] term).
    Recall that twice the sum of the product of two vectors is just twice the dot product of the two vectors. Therefore the derivative for the weight for feature_i is just two times the dot product between the values of feature_i and the current errors, plus 2*l2_penalty*w[i].
    """
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant==True:
        derivative = 2*np.dot(errors, feature)
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2*np.dot(errors, feature)+(2*l2_penalty*weight)
    return derivative


# Example
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.




# Gradient Descent
def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    """
    Now we will write a function that performs a gradient descent.
    The basic premise is simple:
    Given a starting point we update the current weights by moving in the negative gradient direction.
    Recall that the gradient is the direction of increase and therefore the negative gradient is the direction of decrease and we're trying to minimize a cost function.
    The amount by which we move in the negative gradient direction is called the 'step size'.
    We stop when we are 'sufficiently close' to the optimum.
    We will set a maximum number of iterations and take gradient steps until we reach this maximum number.
    If no maximum number is supplied, the maximum should be set 100 by default. (Use default parameter values in Python.)
    """
    print 'Starting gradient descent with l2_penalty = ' + str(l2_penalty)
    weights = np.array(initial_weights)
    iteration = 0 # iteration counter
    while iteration < max_iterations:
        # compute the predictions based on feature_matrix and weights using the predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        error = predictions - output
        for i in xrange(len(weights)):
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            feature = feature_matrix[:,i]
            # compute the derivative for weight[i].
            derivative = feature_derivative_ridge(error, feature, weights[i], l2_penalty, i==0)
            #(Remember: when i=0, you are computing the derivative of the constant!)
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size*derivative
        iteration += 1
    else:
        print 'reached maximum iterations'
    print 'Done with gradient descent at iteration ', iteration
    print 'Learned weights = ', str(weights)
    print iteration
    return weights




# Visualizing effect of L2 penalty¶
simple_features = ['sqft_living']
my_output = 'price'
train_data,test_data = sales.random_split(.8,seed=0)

# Convert data to matrix
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

# Set initial params
initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000


# First, let's consider no regularization.
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix
                                                           , output
                                                           , initial_weights
                                                           , step_size
                                                           , 0.0
                                                           , max_iterations)


# Next, let's consider high regularization
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix
                                                              , output
                                                              , initial_weights
                                                              , step_size
                                                              , 1e11
                                                              , max_iterations)


import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(simple_feature_matrix,output,'k.',
         simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')



# RSS
def get_residual_sum_of_squares(feature_matrix, outcome, weights):
    predicted = predict_output(feature_matrix, weights)
    residuals = predicted - outcome
    RSS = (residuals * residuals).sum()
    return(RSS)

rss_initial = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, initial_weights)
print rss_initial

rss_initial = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_0_penalty)
print rss_initial

rss_initial = get_residual_sum_of_squares(simple_test_feature_matrix, test_output, simple_weights_high_penalty)
print rss_initial













##
