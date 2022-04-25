import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('https://raw.githubusercontent.com/sharibhumayun/predict-trim/main/CSAT%20TRIM%20impact.csv')
df.drop(['Months'], axis =1, inplace = True)
x= df.drop(['TRIM_Index'],axis = 1)
y = df['TRIM_Index']

import sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

Input_values = [0,0,0,0,0,0,0,0,0,0]
Input_values = np.asarray(Input_values)
Input_values = Input_values.reshape(1,-1)
y_pred1 = lr.predict(Input_values)

import pickle
import streamlit as st
import numpy as np


def TRIM_prediction(Input_values):
   
    Input_values = np.asarray(Input_values)
    Input_values = Input_values.reshape(1,-1)
    y_pred1 = lr.predict(Input_values)
    y_pred1
    
    
def main():
    st.title('TRIM Prediction')
    
    
    Mobily = st.text_input('Mobily')
    STC = st.text_input('STC')
    Zain = st.text_input('Zain')
    Communication = st.sidebar.slider('Communication',0.0,100.0,50.0)
    Proposition = st.sidebar.slider('Proposition',0.0,100.0,50.0)
    Sales_Experience = st.sidebar.slider('Sales Experience',0.0,100.0,50.0)
    Opt_inout = st.sidebar.slider('Opt in out',0.0,100.0,50.0)
    Network = st.sidebar.slider('Network',0.0,100.0,50.0)
    Customer_Support = st.sidebar.slider('Customer Support',0.0,100.0,50.0)
    Billing_Payment = st.sidebar.slider('Billing & Payment',0.0,100.0,50.0)
    
    
    y_pred1 = 'TRIM Score'
    
    if st.button('TRIM Index Prediction'):
        y_pred1= TRIM_prediction([Mobily,STC,Zain,Communication,Proposition,Sales_Experience,Opt_inout,Network,Customer_Support,Billing_Payment])
        
    st.success(y_pred1)
        
        
        
if __name__=='__main__':
    main()
