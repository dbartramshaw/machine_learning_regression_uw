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

# import data
sales = sframe.SFrame('/Users/davidbartram-shaw/Machine Learning Course/Course 2 - Regression/kc_house_data.gl/')
sales=sframe.SFrame.to_dataframe(sales)

# Test/Train split
train_data, test_data = train_test_split(sales, test_size=0.33, random_state=42)




#################################################################
# SIMPLE LINEAR REGRESSION
#################################################################

# generic simple linear regression function - single input
def simple_linear_regression(input_feature, output):
    """
    Purpose: Compute intercept and slope
    Input  : input_feature (x), output (y)
    Output : Estimate of intercept (w0) and slope (w1)
    """
    # compute the sum of input_feature and output
    sum_input = input_feature.sum()
    sum_output = output.sum()
    num_data = len(output)

    # compute the product of the output and the input_feature and its sum
    product_of_inp_out = input_feature * output
    sum_product = product_of_inp_out.sum()

    # compute the squared value of the input_feature and its sum
    input_squared = input_feature*input_feature
    squared_sum  = input_squared.sum()

    # use the formula for the slope
    numerator = sum_product - ((float(1)/num_data) * (sum_input*sum_output))
    denominator = squared_sum - ((float(1)/num_data) * (sum_input*sum_input))
    slope = numerator/denominator

    # use the formula for the intercept
    intercept = output.mean()-(slope*input_feature.mean())
    return (intercept, slope)


# Example model
intercept, slope = simple_linear_regression(train_data.sqft_living, train_data.price)
print "Intercept: " + str(intercept)
print "Slope: " + str(slope)




# Prediting values
def get_regression_predictions(input_feature, intercept, slope):
    """
    Purpose: Compute predictions
    Input  : input_feature (x), intercept (w0), slope (w1)
    Output : Predicted output based on estimated intercept, slope and input feature
    """
    # calculate the predicted values:
    predicted_values = input_feature * slope + intercept
    return predicted_values


# Exampel predictions
get_regression_predictions(2650, intercept,slope)             # One house prediction
get_regression_predictions(test_data.sqft_living, intercept,slope)  # Test set prediction


# RSS
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    """
    Purpose: Compute Residual Sum of Squares (RSS)
    Input  : input_feature (x), output (y),
             intercept (w0), slope (w1)
    Output : Residual sum of squares = sum((actual output (y) - predicted output)^2)
    """
    predictions = get_regression_predictions(input_feature, intercept, slope)
    residuals = predictions - output
    RSS = sum(residuals**2)
    #RSS = np.sum((output - (intercept + slope * input_feature)) ** 2) #SINGLE COMPUTION
    return(RSS)

# Example rss
rss = get_residual_sum_of_squares(train_data.sqft_living, train_data.price, intercept, slope)
print 'The RSS is : ' + str(rss)



# Estimate input
def inverse_regression_predictions(output, intercept, slope):
    """
    Purpose: Reverse predicitons
    Input  : output (y), intercept (w0), slope (w1)
    Output : Estimate of input based on intercept, slope and given output
    """
    estimated_input = (output - intercept)/slope
    return(estimated_input)

estimated_input = inverse_regression_predictions(696511.51818612812, intercept, slope)




#################################################################
# PLOTTING
#################################################################

import matplotlib.pyplot as plt
input_feature = train_data.sqft_living
output = train_data.price

# Plots of input vs output
plt.plot(input_feature, output, 'b.', label='train data')
plt.title('Price vs Sqft.')
plt.ylabel('Price')
plt.xlabel('Sq.ft.')

# Regression line
z = get_regression_predictions(input_feature, intercept, slope)
plt.plot(input_feature, z, 'r', linewidth=2.0, label='regression line')
plt.legend(loc='upper left')
plt.show()

# Plot Residuals
plt.plot(input_feature, output - z, '.')
plt.title('Residual - Price vs Sq.ft.')
plt.ylabel('Residual (Price - predicted price)')
plt.xlabel('Sq.ft.')
plt.show()










####
