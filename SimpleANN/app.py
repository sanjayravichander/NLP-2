import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the model and encoders
model = tf.keras.models.load_model('model.h5')

with open('geo_encoders.pkl', 'rb') as f:
    encoder_geo = pickle.load(f)

with open('gender_encoders.pkl', 'rb') as f:
    encoder_gender = pickle.load(f)

with open('scaler_df.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit application
st.title("Churn Prediction App")

# User Input
geography = st.selectbox("Geography", encoder_geo.categories_[0])
gender = st.selectbox("Gender", encoder_gender.classes_)
age = st.number_input("Age", min_value=0, max_value=100, value=50)
tenure = st.number_input("Tenure", min_value=0, value=1)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_credit_card = st.checkbox("Has Credit Card")
is_active_member = st.checkbox("Is Active Member")
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=500)
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [int(has_credit_card)],  # Convert Boolean to int (1/0)
    "IsActiveMember": [int(is_active_member)],  # Convert Boolean to int (1/0)
    "EstimatedSalary": [estimated_salary]
})

# Encode Gender (Label Encoding)
input_data['Gender'] = encoder_gender.transform(input_data['Gender'])

# Encode Geography (One-Hot Encoding) properly
geo_one_hot = encoder_geo.transform(input_data[['Geography']]).toarray()  # Convert sparse matrix to dense
geo_one_hot_df = pd.DataFrame(geo_one_hot, columns=encoder_geo.get_feature_names_out(['Geography']))

# Merge one-hot encoded Geography and drop original column
input_df = pd.concat([input_data.drop(columns=['Geography']), geo_one_hot_df], axis=1)

# Ensure all features match those used during training
missing_cols = set(scaler.feature_names_in_) - set(input_df.columns)
for col in missing_cols:
    input_df[col] = 0  # Add missing columns with default value 0

# Reorder columns to match scaler training data
input_df = input_df[scaler.feature_names_in_]

# Scale the data
input_scaled = scaler.transform(input_df)

# Predict Churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

st.write(f"Prediction Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.error("The customer is likely to churn")
else:
    st.success("The customer is likely to not churn")
