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
# REGRESSION - MODEL SELECTION & ASSESMENT
#################################################################


# Polynomial_sframe function
def polynomial_sframe(feature, degree):
    # assume that degree >= 1 & initialize the SFrame
    poly_sframe = sframe.SFrame()
    # first degree
    poly_sframe['power_1']=feature
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            name = 'power_' + str(power)
            poly_sframe[name]=feature.apply(lambda x: x**power)
    return poly_sframe

# Example
tmp = sframe.SArray([1., 2., 3.])
print polynomial_sframe(tmp, 6)


'''
# Visualizing polynomial regression
sales = sales.sort(['sqft_living', 'price'])             # For plotting purposes (connecting the dots),we must sort by the values
poly_data = polynomial_sframe(sales['sqft_living'], 1)
poly_data['price'] = sales['price']
poly_data = sframe.SFrame.to_dataframe(poly_data)

output = poly_data['price']
input_features = poly_data.drop('price', axis=1)

from sklearn import linear_model
regr = linear_model.LinearRegression()
model = regr.fit(input_features, output)
print model.intercept_
print model.coef_




# Changing the data and re-learning
set_1, set_2 = sales_split1.random_split(0.5, seed=0)
set_3, set_4 = sales_split2.random_split(0.5, seed=0)
set_1 = set_1.sort(['sqft_living', 'price'])
set_2 = set_2.sort(['sqft_living', 'price'])
set_3 = set_3.sort(['sqft_living', 'price'])
set_4 = set_4.sort(['sqft_living', 'price'])

# 15th order poly
test1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = test1.column_names() # get the name of the features
test1['price'] = set_1['price'] # add price to the data since it's the target
model_test1 = graphlab.linear_regression.create(test1, target = 'price', features = my_features, validation_set = None)

model_test1.get("coefficients").print_rows(num_rows = 16)
plt.plot(test1['power_1'],test1['price'],'.',test1['power_1'], model_test1.predict(test1),'-')




# Selecting a Polynomial Degree

# Initialise the results sframe
rss_results = graphlab.SFrame({'i': [int()], 'RSS': [float()]})
#for i in range(1, 15+1):
for i in range(1, 18+1):
#for i in range(1, 4):
    loop_sframe = polynomial_sframe(training['sqft_living'], i)
    my_features = loop_sframe.column_names() # get the name of the features
    loop_sframe['price'] = training['price']
    model_name = 'loop_model_' + str(i)
    model_name = graphlab.linear_regression.create(loop_sframe, target = 'price', features = my_features, validation_set = None,verbose = False)
    loop_validation_sf = polynomial_sframe(validation['sqft_living'], i)
    loop_validation_sf['price'] = validation['price']

    predictions = model_name.predict(loop_validation_sf)

    RSS = sum((validation['price']-predictions)**2)
    rss_loop = graphlab.SFrame({'i': [i], 'RSS': [RSS]})
    rss_results = rss_results.append(rss_loop)
    print 'loop '+ str(i)+' RSS='+str(RSS)




'''
