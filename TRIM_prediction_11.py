import pickle
import streamlit as st
import numpy as np
import joblib


loaded_model = 'https://github.com/sharibhumayun/predict-trim/blob/main/trained_model.sav'


def TRIM_prediction(Input_values):
   
    Input_values = np.asarray(Input_values)
    Input_values = Input_values.reshape(1,-1)
    y_pred1 = loaded_model.predict(Input_values)
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
    
    
    y_pred1 = 'TRIM Score'
    
    if st.button('TRIM Index Prediction'):
        y_pred1= TRIM_prediction([Mobily,STC,Zain,Communication,Proposition,Sales_Experience,Opt_inout,Network,Customer_Support,Billing_Payment])
        
    st.success(y_pred1)
        
        
        
if __name__=='__main__':
    main()
