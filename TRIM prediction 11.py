# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 00:33:40 2022

@author: Saba Akhlaq
"""

import pickle
import streamlit as st
import numpy as np

loaded_model = pickle.load(open('C:/Users/Saba Akhlaq/OneDrive/Desktop/ML model deployment/trained_model6.sav', 'rb'))

def TRIM_prediction(x_new):
   
    x_new = x_new.reshape((1,-1))
    y_pred1 = loaded_model.predict(x_new)

    y_pred1
    
    
def main():
    st.title('TRIM Prediction')
    
    
    Mobily = st.text_input('Mobily')
    STC = st.text_input('STC')
    Zain = st.text_input('Zain')
    Communication = st.slider('Communication',0,100,50)
    Proposition = st.slider('Proposition',0,100,50)
    Sales_Experience = st.slider('Sales Experience',0,100,50)
    Opt_inout = st.slider('Opt in out',0,100,50)
    Network = st.slider('Network',0,100,50)
    Customer_Support = st.slider('Customer Support',0,100,50)
    Billing_Payment = st.slider('Billing & Payment',0,100,50)
    
    
    y_pred1 = ''
    
    if st.button('TRIM Index Prediction'):
        y_pred1= TRIM_prediction([Mobily,STC,Zain,Communication,Proposition,Sales_Experience,Opt_inout,Network,Customer_Support,Billing_Payment])
        
    st.success(y_pred1)
        
        
        
if __name__=='__main__':
    main()
    
       