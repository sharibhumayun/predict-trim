import pickle
import streamlit as st
import numpy as np


model = 'C:/Users/Sharib Shamsi/Desktop/ML model/trained_model.pkl'
loaded_model = model

def TRIM_prediction(Input_values):
   
    Input_values = np.asarray(Input_values)
    Input_values = Input_values.reshape(1,-1)
    y_pred1 = model.predict(Input_values)
    y_pred1
    
    
def main():
    st.title('TRIM Prediction')
    
    
    Mobily = st.text_input('Mobily')
    STC = st.text_input('STC')
    Zain = st.text_input('Zain')
    Communication = st.sidebar.slider('Communication',0,100,50)
    Proposition = st.sidebar.slider('Proposition',0,100,50)
    Sales_Experience = st.sidebar.slider('Sales Experience',0,100,50)
    Opt_inout = st.sidebar.slider('Opt in out',0,100,50)
    Network = st.sidebar.slider('Network',0,100,50)
    Customer_Support = st.sidebar.slider('Customer Support',0,100,50)
    Billing_Payment = st.sidebar.slider('Billing & Payment',0,100,50)
    
    
    y_pred1 = ''
    
    if st.button('TRIM Index Prediction'):
        y_pred1= TRIM_prediction(['Mobily','STC','Zain','Communication','Proposition','Sales_Experience','Opt_inout,Network','Customer_Support','Billing_Payment'])
        
    st.success(y_pred1)
        
        
        
if __name__=='__main__':
    main()
    
       
