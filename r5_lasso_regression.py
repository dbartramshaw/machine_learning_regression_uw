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
# REGRESSION - LASSO REGRESSION
#################################################################
"""
 Note on co-ordinate descent
 The lasso cost function is non-differentiable, so gradient decent is not viable
 We could impliment subgradient decent to overcome this
 Instead we use co-ordiate decent
"""

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

# Normalise features
def normalize_features(feature_matrix):
    """
    In the most datasets, features vary wildly in their relative magnitude:
        (House E.G.
         sqft_living is very large overall compared to bedrooms, for instance.
         As a result, weight for sqft_living would be much smaller than weight for bedrooms.
        )
    This is problematic because "small" weights are dropped first as l1_penalty goes up.
    To give equal considerations for all features, we need to normalize features.
    We divide each feature by its 2-norm so that the transformed feature has norm 1.
    """
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalised = feature_matrix / norms
    return (normalised,norms)

# Example
features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print features
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print norms
# should print
# [5.  10.  15.]



# Implementing Coordinate Descent with normalized features
"""
We seek to obtain a sparse set of weights by minimizing the LASSO cost function

    SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|).

(By convention, we do not include w[0] in the L1 penalty term. We never want to push the intercept to zero.)
The absolute value sign makes the cost function non-differentiable,
so simple gradient descent is not viable (you would need to implement a method called subgradient descent).
Instead, we will use coordinate descent:
at each iteration, we will fix all weights but weight i and find the value of weight i that minimizes the objective.
That is, we look for

    argmin_{w[i]} [ SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|) ]

where all weights other than w[i] are held to be constant.
We will optimize one w[i] at a time, circling through the weights multiple times.

    1) Pick a coordinate i
    2) Compute w[i] that minimizes the cost function SUM[ (prediction - output)^2 ] + lambda*( |w[1]| + ... + |w[k]|)
    3) Repeat Steps 1 and 2 for all coordinates, multiple times

For this notebook, we use cyclical coordinate descent with normalized features,
where we cycle through coordinates 0 to (d-1) in order, and assume the features were normalized as discussed above.
The formula for optimizing each coordinate is as follows:

           ┌ (ro[i] + lambda/2)     if ro[i] < -lambda/2
    w[i] = ├ 0                      if -lambda/2 <= ro[i] <= lambda/2
           └ (ro[i] - lambda/2)     if ro[i] > lambda/2
where

    ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ].

Note that we do not regularize the weight of the constant feature (intercept) w[0], so, for this weight, the update is simply:

    w[0] = ro[i]
"""

# Effect of L1 penalty
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
simple_feature_matrix
simple_feature_matrix, norms = normalize_features(simple_feature_matrix)
norms

# Initial weights
weights = np.array([1., 4., 1.])

# Predictions
prediction = predict_output(simple_feature_matrix, weights)
prediction

def get_ro(feature_matrix, output, prediction, weights, i):
    error = output - prediction
    return (feature_matrix[:,i] * (error + (weights[i] * feature_matrix[:,i]))).sum()


"""
QUIZ QUESTION
Recall that, whenever ro[i] falls between -l1_penalty/2 and l1_penalty/2, the corresponding weight w[i] is sent to zero.
Now suppose we were to take one step of coordinate descent on either feature 1 or feature 2.
What range of values of l1_penalty would not set w[1] zero, but would set w[2] to zero, if we were to take a step in that coordinate?
"""
print 'Question 1 and 2'
print 'ro_0', '%.4g' % (get_ro(simple_feature_matrix, output, prediction, weights, 0)*2)
print 'ro_1', '%.4g' % (get_ro(simple_feature_matrix, output, prediction, weights, 1)*2)
print 'ro_2', '%.4g' % (get_ro(simple_feature_matrix, output, prediction, weights, 2)*2)


#compute the value of ro[i] for each features in this simple model
ro = []
for i in range(len(simple_feature_matrix[0])):
    prediction = predict_output(simple_feature_matrix, weights)
    this_ro = np.dot(simple_feature_matrix[:,i] , (output - prediction + weights[i]*simple_feature_matrix[:,i]))
    ro.append(this_ro)

print(ro)


# Single Coordinate Descent - Single Step
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    ro_i = (feature_matrix[:,i] * (output - prediction + weights[i]*feature_matrix[:,i]) ).sum()
    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i+(l1_penalty/2)
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i-(l1_penalty/2)
    else:
        new_weight_i = 0.
    return new_weight_i


# Example - should print 0.425558846691
import math
i = 1
feature_matrix = np.array([[3./math.sqrt(13),1./math.sqrt(10)],[2./math.sqrt(13),3./math.sqrt(10)]])
output = np.array([1., 1.])
weights = np.array([1., 4.])
l1_penalty = 0.1
print lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)



# Cyclical coordinate descent
def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
    Now that we have a function that optimizes the cost function over a single coordinate,
    let us implement cyclical coordinate descent where we optimize coordinates 0, 1, ..., (d-1) in order and repeat.
    When do we know to stop? Each time we scan all the coordinates (features) once,
    we measure the change in weight for each coordinate. If no coordinate changes by more than a specified threshold, we stop.

    For each iteration:
        As we loop over features in order and perform coordinate descent, measure how much each coordinate changes.
        After the loop, if the maximum change across all coordinates is falls below the tolerance, stop.
        Otherwise, go back to step 1.
    Return weights
    """
    converge = True
    weights = list(initial_weights)
    while(converge):
        max_change = 0
        changes = []
        for i in range(len(weights)):
            old_weights_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            #print "new weight = %d" %weights[i]
            this_change = weights[i] - old_weights_i
            changes.append(this_change)

        max_change =  max(np.absolute(changes))
        #print max_change
        if (max_change < tolerance) :
            converge = False
    return weights[:]



# Example
simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

# create a normalized version of the feature matrix,
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

# Learn weights using lasso co-ordinate decent
weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
weights

predicted_value = np.dot(normalized_simple_feature_matrix, weights)
RSS = sum((predicted_value-output)**2)
print RSS






#
