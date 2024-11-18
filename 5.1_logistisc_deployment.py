#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


model= pickle.load(open('logm_pkl','rb'))


# In[3]:


st.title('Model Deployment using Logistic Regression')


# In[4]:


import streamlit as st
import pandas as pd

st.title("Prediction on Titanic Data")

# Option for the user to choose between CSV upload or manual input
input_option = st.selectbox("How would you like to input the data?", ("Upload CSV", "Manual Input"))

if input_option == "Upload CSV":
    # Upload CSV file containing df_test
    uploaded_file = st.file_uploader("Upload your CSV file containing the test data", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded CSV file
        df_test = pd.read_csv(uploaded_file)
        
        # Define the required features
        required_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Female', 'Male', 'Embarked_C', 'Embarked_S', 'Embarked_Q']
        
        # Check if df_test contains all the required features
        if all(feature in df_test.columns for feature in required_features):
            df = df_test[required_features]  # Keep only the necessary features
            
            st.subheader('Uploaded Test Data')
            st.write(df)

            # Assuming 'model' is already defined and trained
            pred_prob = model.predict_proba(df)
            pred = model.predict(df)

            # Display the predicted class
            st.subheader('Predicted')
            st.write('Yes' if pred_prob[0][1] > 0.5 else 'No')

            # Display the prediction probabilities
            st.subheader('Prediction Probabilities')
            st.write(pred_prob)
        else:
            st.error("The uploaded file does not contain the required features.")
    else:
        st.info("Please upload a CSV file to proceed.")
        
elif input_option == "Manual Input":
    st.subheader("Enter the Features Manually")

    # Manual input fields for each feature
    Pclass = st.selectbox('Pclass (Ticket Class)', [1, 2, 3])
    Age = st.number_input('Age', min_value=0, max_value=100, step=1)
    SibSp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, step=1)
    Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, step=1)
    gender = st.selectbox('Gender', ['Female', 'Male'])
    Female = 1 if gender == 'Female' else 0
    Male = 1 if gender == 'Male' else 0
    Embarked = st.selectbox('Port of Embarkation', ['C', 'S', 'Q'])
    Embarked_C = 1 if Embarked == 'C' else 0
    Embarked_S = 1 if Embarked == 'S' else 0
    Embarked_Q = 1 if Embarked == 'Q' else 0

    # Create a DataFrame from the manual input
    input_data = {
        'Pclass': [Pclass],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Female': [Female],
        'Male': [Male],
        'Embarked_C': [Embarked_C],
        'Embarked_S': [Embarked_S],
        'Embarked_Q': [Embarked_Q]
    }
    
    df_manual = pd.DataFrame(input_data)
    st.subheader('Manual Input Data')
    st.write(df_manual)

    # Assuming 'model' is already defined and trained
    pred_prob = model.predict_proba(df_manual)
    pred = model.predict(df_manual)

    # Display the predicted class
    st.subheader('Predicted')
    st.write('Yes' if pred_prob[0][1] > 0.5 else 'No')

    # Display the prediction probabilities
    st.subheader('Prediction Probabilities')
    st.write(pred_prob)


# In[ ]:




