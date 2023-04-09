# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 08:18:00 2023

@author: jiten
"""

import numpy as np
import pickle


# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\jiten\trained_model.sav', 'rb'))

input_data=[9.465,21.01,60.11,269.4,0.1044,0.07773,0.02172,0.01504,0.1717,0.06899,0.2351,2.011,1.66,14.2,0.01052,0.01755,0.01714,0.009333,0.02279,0.004237,10.41,31.56,67.03,330.7,0.1548,0.1664,0.09412,0.06517,0.2878,0.09211]

#Change the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#Reshape the numpy array as we are predicting for one data point 
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

#Standarising the input data
#input_std=scaler.transform(input_data_reshape)

prediction = loaded_model.predict(input_data_reshape)   #this will again give two values just we saw in test case.
print("Prediction= ",prediction)

prediction_label=np.argmax(prediction)
print("Prediction Label= ",prediction_label)

if(prediction_label==0):
    print("The tumor is Maligant")
else:
    print("The tumor is Benign")


