import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write(""" 
# Employee Churn Prediction App

""")



st.sidebar.title('Please enter these parameters to predict!')

html_temp = """
<div style="background-color:tomato;padding:1.5px">

<h2 style="color:white;text-align:center;">This app predicts "If an employee leaves the company or not" !</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

def user_input_features():
    satisfaction_level = st.sidebar.slider('Satisfaction level of the employee:', 0.09,1.00, 0.61)
    time_spend_company  = st.sidebar.slider('How many years spent the employee in the company?', 2.0, 10.0, 3.0)
    number_project = st.sidebar.slider('How many projects assigned to the employee?', 2.0, 7.0, 4.0)
    average_montly_hours  = st.sidebar.slider('How many hours in avereg the employee worked in a month?', 96.0, 310.0, 200.0)
    last_evaluation = st.sidebar.slider("Last evaluation of Employer on the performance of employee:", 0.36, 1.0, 0.70)

    

    data = {'satisfaction_level' : satisfaction_level,
            'time_spend_company' : time_spend_company,
            'number_project' : number_project,
            'average_montly_hours' : average_montly_hours,
            'last_evaluation' : last_evaluation }

    features = pd.DataFrame(data, index=[0])
    return features


model_name=st.sidebar.selectbox("Select your model:",("XGBOOST","Gradient Boosting"))
df = user_input_features()

st.subheader('Your Inputs:')
if model_name=="XGBOOST":
	model=pickle.load(open("churn_prediction_xgb.pkl","rb"))
	st.success("You selected {} model".format(model_name))
else :
	model=pickle.load(open("churn_prediction_grad.pkl","rb"))
	st.success("You selected {} model".format(model_name))
 
st.write(df)



# Apply model to make predictions
prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write(prediction[0])
if prediction[0]== 1:
    st.write(' = Will leave!')
elif prediction[0]== 0:
    st.write('= Will not leave!')
 

st.subheader('PredictionProbability')
st.write(round(prediction_proba[0][1], 2))

if prediction_proba[0][0] <0.5:
    st.write('According to the data that you provide, the Employee will not leave the company with the probability of ' + str(round(prediction_proba[0][1],2)) + '!')

else:
    st.write('According to the data that you provide, the Employee will leave the company with the probability of ' + str(round(prediction_proba[0][1],2)) + '!')
