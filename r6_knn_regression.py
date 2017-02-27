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
(train_and_validation, test) = sales.random_split(.8, seed=1) # initial train/test split
(train, validation) = train_and_validation.random_split(.8, seed=1) # split training set into training and validation sets

#################################################################
# REGRESSION - KNN REGRESSION & KERNALS
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

# Normalise features
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalised = feature_matrix / norms
    return (normalised,norms)


feature_list = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition',
                'grade','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

# Normalise
normalize_features(features_train)
normalize_features(features_test)
normalize_features(features_valid)

# Convert all to Numeric
# this solves the above ufunc 'divide' not supported error
def input_types_numeric(input_array):
    df = pd.DataFrame(input_array)
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return df.as_matrix()

# Normalise - Take 2
features_train = input_types_numeric(features_train)
features_test = input_types_numeric(features_test)
features_valid = input_types_numeric(features_valid)



################################
# Compute a single distance
################################
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))


################################
# Compute multiple distances
################################

"""
To do nearest neighbor regression, we need to compute the distance between our query point and all points in the training set.
To visualize this nearest-neighbor search, let's first compute the distance from our query point (features_test[0])
to the first 10 houses of the training set (features_train[0:10]) and then search for the nearest neighbor within
this small set of houses. Through restricting ourselves to a small set of houses to begin with,
we can visually scan the list of 10 distances to verify that our code for finding the nearest neighbor is working.
"""

# E.G. Among the first 10 training houses, which house is the closest to the query house?
for i in range(0,10):
    print str(i) + " : " + str(np.sqrt(np.sum((features_train[i]-features_test[2])**2)))

"""
It is computationally inefficient to loop over computing distances to all points in our training dataset.
Fortunately, many of the Numpy functions can be vectorized,
applying the same operation over multiple values or vectors.
We now walk through this process.
"""

# instead we computes the element-wise difference between the features of the query point and the first 3 training points
for i in xrange(3):
    print features_train.as_matrix()[i]-features_test.as_matrix()[0]


# Perform 1-nearest neighbor regression¶
def compute_distances(features_instances, features_query):
    """
    function that computes the distances from a query point to all training points.
    """
    diff = features_instances[0:len(features_instances)] - features_query
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return distances




# Perform k-nearest neighbor regression
"""
For k-nearest neighbors, we need to find a set of k houses in the training set closest to a given query house.
We then make predictions based on these k nearest neighbors.

Fetch k-nearest neighbors

Inputs
    1) the value of k;
    2) the feature matrix for the training houses; and
    3) the feature vector of the query house

Outputs
The indices of the k closest training houses.

Example
For instance, with 2-nearest neighbor,
a return value of [5, 10] would indicate that the 6th and 11th training houses are closest to the query house.
"""

def k_nearest_neighbors(k, feature_train, features_query):
    distances = compute_distances(features_train, features_query)
    neighbors = np.argsort(distances)[0:k]
    return neighbors


# Example - find closest house to third house in test set
print k_nearest_neighbors(4, features_train, features_test[2])


# Make a single prediction by averaging k nearest neighbor outputs
def predict_output_of_query(k, features_train, output_train, features_query):
    k_nearest = k_nearest_neighbors(k, features_train, features_query)
    prediction = np.mean(output_train[k_nearest])
    return(prediction)


# Make multiple predictions
def predict(k, features_train, output_train, features_query):
    predictions = np.zeros((features_query.shape[0]))
    for idx in range(features_query.shape[0]):
        predictions[idx] = predict_output_of_query(k, features_train, output_train, features_query[idx])
    return predictions









#
