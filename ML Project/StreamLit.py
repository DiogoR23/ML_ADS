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

with open('EducationEncoder.pkl', 'rb') as educationEncoder_file:
    educationEncoder = pickle.load(educationEncoder_file)

with open('JobEncoder.pkl', 'rb') as jobEncoder_file:
    jobEncoder = pickle.load(jobEncoder_file)

with open('MaritalEncoder.pkl', 'rb') as maritalEncoder_file:
    maritalEncoder = pickle.load(maritalEncoder_file)

# Options for categorical features
job_types = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
             'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']

marital_status = ['divorced', 'married', 'single']

education_levels = ['primary', 'secondary', 'tertiary', 'unknown']

# Defining the streamLit interface
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

# Converting Yes & No values for numeric
default = 1 if default == 'Yes' else 0
housing = 1 if housing == 'Yes' else 0
loan = 1 if loan == 'Yes' else 0

# Converting data into Dataframe
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
def encode_variables(df):
    df['job'] = jobEncoder.transform(df[['job']])
    df['marital'] = maritalEncoder.transform(df[['marital']])
    df['education'] = educationEncoder.transform(df[['education']])
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
    df['balance'] = np.log(df['balance'] + 1)
    return df

def scale_data(df):
    num_vars = df.select_dtypes(include=[np.number])
    scaled_data = scaler.transform(num_vars)
    df[num_vars.columns] = scaled_data
    return df

def select_features(df):
    selected_features = selector.transform(df)
    return selected_features

# Apply pre-processing to input data
bank = encode_variables(bank)
bank = cyclical_encoding(bank)
bank = log_transform(bank)
bank = scale_data(bank)

# Select the best features
bank_selected = select_features(bank)

# Making predictions with created model
prediction = model.predict(bank_selected)

# Show prediction
st.write("Prediction:", "Yes" if prediction[0] == 1 else "No")
