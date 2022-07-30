import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
st.image(
    "Clarusway_logo.png",
    width=300,
)
st.title("DS-Group 4️⃣ Capstone Project-2")

#st.title("Employee Churn Prediction App")





html_temp = """
<div style="background-color:DodgerBlue;padding:1.5px">

<h4 style="color:white;text-align:center;">This app predicts "If an employee leaves the company or not" !</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


# USER INPUTS
st.sidebar.title('Please enter these parameters to predict!')

def user_input_features():
    satisfaction_level = st.sidebar.slider('Satisfaction level of the employee:', 0.09,1.00, 0.61)
    time_spend_company  = st.sidebar.slider('How many years spent the employee in the company?', 2.0, 10.0, 3.0)
    number_project = st.sidebar.slider('How many projects assigned to the employee?', 2.0, 7.0, 4.0)
    average_montly_hours  = st.sidebar.slider('How many hours in avereg the employee worked in a month?', 96.0, 310.0, 200.0)
    last_evaluation = st.sidebar.slider("Last evaluation of Employer on the performance of employee:", 0.36, 1.0, 0.70)
  

    data = {'Satisfaction_level' : satisfaction_level,
            'Time_spent_company' : time_spend_company,
            'number_of_project' : number_project,
            'Average_working_hours' : average_montly_hours,
            'Last_evaluation' : last_evaluation }

    features = pd.DataFrame(data, index=[0])
    return features


model_name=st.sidebar.selectbox("Select your model:",("XGBOOST","Gradient Boosting"))
df = user_input_features()

# Showing the Inputs
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

st.subheader('Prediction:')

if prediction[0]== 1:
    st.write('The employee will leave the company!')
elif prediction[0]== 0:
    st.write('The employee will **not** leave the company!')
 

st.subheader('Probability:')
if round(prediction_proba[0][1],2) >0.70:
    st.error('According to your input data, the Employee will leave the company with the probability of ' + str(round(prediction_proba[0][1],2)) + '!')

elif round(prediction_proba[0][1],2) <0.35:
    st.success('According to your input data, the Employee will leave the company with the probability of ' + str(round(prediction_proba[0][1],2)) + '!')

else:
    st.warning('According to your input data, the Employee will leave the company with the probability of ' + str(round(prediction_proba[0][1],2)) + '!')
    
st.text("             ")
st.text("             ")
st.text("             ")
st.text("             ")
st.text("             ")
st.text("             ")



# SECOND PART:
html_temp = """
<div style="background-color:DodgerBlue;padding:1.5px">

<h4 style="color:white;text-align:center;">Getting list of N employees who are more likely to churn!</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.info(
                f"""
                    By this part of the application the N employees who are more likely to churn can be filtered!
    
                    """
            )

N = st.slider('Value of N:', 0,500, 1)

#c29, c30, c31 = st.columns([1, 6, 1])

#with c30:    

# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X = df.loc[:,['satisfaction_level', 'time_spend_company', 'last_evaluation','number_project', 'average_montly_hours']]
    proba = model.predict_proba(X)
    X["pred_proba"] = proba[:,1]
    df2 = X.sort_values(by='pred_proba', ascending=False).head(N)
    st.write(df2)
else:
    st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [Example CSV input file]('https://raw.githubusercontent.com/AriFatih/Churn_analysis_capstone_project/main/HR_Dataset.csv')

                """
        )   

    st.stop()
        
