import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Loading all the needed pickles
with open('RandomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('Selector.pkl', 'rb') as selector_file:
    selector = pickle.load(selector_file)

# Options for categorical features
job_types = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
             'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']

marital_status = ['divorced', 'married', 'single']

education_levels = ['primary', 'secondary', 'tertiary', 'unknown']

# Defining the Streamlit interface
st.title("Forecast using Random Forest")

# Input by the User
age = st.number_input("Age", min_value=0)
job = st.selectbox("Job type", job_types)
marital = st.selectbox("Marital Status", marital_status)
education = st.selectbox("Education Level", education_levels)
default = st.selectbox("Has credit?", ('No', 'Yes'))
balance = st.number_input("Average annual balance", min_value=0)
housing = st.selectbox("Has housing loan?", ('No', 'Yes'))
loan = st.selectbox("Has a personal loan?", ('No', 'Yes'))
day = st.number_input("Day", min_value=1, max_value=31)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month = st.selectbox("Month", months)
campaign = st.number_input("Number of times contacted in this campaign", min_value=0)
pdays = st.number_input("Number of days since the customer was last contacted", min_value=-1)
previous = st.number_input("Number of times contacted before this campaign", min_value=0)

# Converting Yes & No values to numeric
default = 1 if default == 'Yes' else 0
housing = 1 if housing == 'Yes' else 0
loan = 1 if loan == 'Yes' else 0

# Converting data into DataFrame
data = {
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'day': [day],
    'month': [month],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous]
}

bank = pd.DataFrame(data)

# Functions for pre-processing input data
def oneHotEncoder_job(df, job_var):
    for job_type in job_types:
        if job_var == job_type:
            df[f'job_{job_type}'] = 1
        else:
            df[f'job_{job_type}'] = 0
    return df

def oneHotEncoder_marital(df, marital_var):
    for marital_stat in marital_status:
        if marital_var == marital_stat:
            df[f'marital_{marital_stat}'] = 1
        else:
            df[f'marital_{marital_stat}']= 0
    return df

def labelEncoder_education(df):
    education_map = {'primary': 0, 'secondary': 1,'tertiary': 2, 'unknown': 3}
    df['education'] = df['education'].map(education_map)
    return df

def cyclical_encoding(df):
    num_month = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    df['month_num'] = df['month'].map(num_month)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df.drop('month', axis=1, inplace=True)
    df.drop('month_num', axis=1, inplace=True)
    return df

def log_transform(df):
    df['balance_log'] = np.log(df['balance'] + 1)
    df.drop('balance', axis=1, inplace=True)
    return df

def scale_data(df, num_var):
    num_vars = df[num_var]
    scaled_data = scaler.transform(num_vars)
    df[num_var] = scaled_data
    return df

def select_features(df):
    selected_features = selector.transform(df)
    return selected_features

# Make the predictor button
if st.button('Make Predictions'):
    # Apply pre-processing to input data
    bank = cyclical_encoding(bank)
    bank = oneHotEncoder_job(bank, job)
    bank = oneHotEncoder_marital(bank, marital)
    bank = labelEncoder_education(bank)
    bank = log_transform(bank)
    num_var = ['age', 'balance_log', 'education', 'day', 'month_sin', 'month_cos', 'campaign', 'pdays', 'previous']
    bank = scale_data(bank, num_var)

    # Ensure the DataFrame has all the expected columns
    expected_columns = selector.get_feature_names_out()
    for col in expected_columns:
        if col not in bank.columns:
            bank[col] = 0
    
    # Reorder the DataFrame to match the expected columns
    bank = bank[expected_columns]
    
    # Select the best features
    bank_selected = select_features(bank)

    # Making predictions with created model
    prediction = model.predict(bank_selected)

    # Show prediction
    st.write("Prediction:", "Yes" if prediction[0] == 1 else "No")
