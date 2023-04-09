# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 08:27:03 2023

@author: jiten
"""

import numpy as np
import pickle
import streamlit as st




# loading the saved model
loaded_model = pickle.load(open(r'C:\Users\jiten\trained_model.sav', 'rb'))

def Breast_cancer_detection(input_data):
    

    #Change the input data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #Reshape the numpy array as we are predicting for one data point 
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

    #Standarising the input data
    #input_std=scaler.transform(input_data_reshape)

    prediction = loaded_model.predict(input_data_reshape )   #this will again give two values just we saw in test case.
    print("Prediction= ",prediction)

    prediction_label=np.argmax(prediction)
    print("Prediction Label= ",prediction_label)

    if(prediction_label==0):
        return "The tumor is Maligant"
    else:
        return "The tumor is Benign"



def main():
    #giving a title
    st.title("Breast Cancer Detection Web App")
    
    meanradius=st.text_input("Raidus value")
    meantexture=st.text_input("Mean Texture value") 
    meanperimeter=st.text_input("Mean Perimeter value")
    meanarea=st.text_input("Mean Area value")
    meansmoothness=st.text_input("Mean Smoothness value")
    meancompactness=st.text_input("Mean Compactness value")
    meanconcavity=st.text_input("Mean Concavity value")
    meanconcavepoints=st.text_input("Mean Concave Points value")
    meansymmetry=st.text_input("Mean Symmetry value")
    meanfractaldimension=st.text_input("Mean Fractal Dimension value")
    radiuserror=st.text_input("Raidus Error")
    textureerror=st.text_input("texture Error") 
    perimetererror=st.text_input("Perimeter Error") 
    areaerror=st.text_input("Area Error")
    smoothnesserror=st.text_input("Smoothness Error")
    compactnesserror=st.text_input("Compactness Error") 
    concavityerror=st.text_input("Concavity Error")
    concavepointserror=st.text_input("Concave Points Error") 
    symmetryerror=st.text_input("Symmetery Error")
    fractaldimensionerror=st.text_input("Fractal Dimension Error")
    worstradius=st.text_input("Worst Raidus")
    worsttexture=st.text_input("Worst Texture")
    worstperimeter=st.text_input("Worst Perimeter")
    worstarea=st.text_input("Worst Area")
    worstsmoothness=st.text_input("Worst Smoothness")
    worstcompactness=st.text_input("Worst Compactness") 
    worstconcavity=st.text_input("Worst Concavity")
    worstconcavepoints=st.text_input("Worst Concavity Points") 
    worstsymmetry=st.text_input("Worst Symmetry")
    worstfractaldimension=st.text_input("Worst Fractal Dimensions")
    
    
    diagnosis = ''
    
    
    if st.button('Cancer Test Result'):
        diagnosis = Breast_cancer_detection([
            ['meanradius', 'meantexture', 'meanperimeter', 'meanarea','meansmoothness', 'meancompactness', 'meanconcavity','meanconcavepoints', 'meansymmetry', 'meanfractaldimension','radiuserror', 'textureerror', 'perimetererror', 'areaerror','smoothnesserror', 'compactnesserror', 'concavityerror','concavepointserror', 'symmetryerror', 'fractaldimensionerror','worstradius', 'worsttexture', 'worstperimeter', 'worstarea','worstsmoothness', 'worstcompactness', 'worstconcavity','worstconcavepoints', 'worstsymmetry', 'worstfractaldimension'] ])
        
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    