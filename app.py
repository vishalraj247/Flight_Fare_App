import streamlit as st
from src.data.data_preprocessor import DataPreprocessor
from src.data.ml_model_data_preprocessor import PreProcessor
import joblib
import tensorflow as tf
import pandas as pd
from datetime import time, timedelta

# Set Streamlit app title
st.title("Flight Fare Prediction")

# List of airports
airports_list = ['ATL', 'MIA', 'PHL', 'SFO', 'LGA', 'LAX', 'ORD', 'IAD', 'EWR', 'DEN', 'DFW', 'BOS', 'OAK', 'DTW', 'CLT', 'JFK']

# Collect user inputs for the starting airport
starting_airport = st.selectbox('Select origin airport:', options=airports_list)

# Update destination options based on the starting airport choice
destination_airports_list = [airport for airport in airports_list if airport != starting_airport]
destination_airport = st.selectbox('Select destination airport:', options=destination_airports_list)

# User selects time in 15-minute intervals
departure_time = st.slider(
    "Select departure time:", 
    value=time(12, 0),  # default value to current time or any other logic you prefer
    format="HH:mm",  # format of the time displayed
    step=timedelta(minutes=3)  # step size as 3 minutes
)

# Store user inputs in a dictionary
user_input = {
    'startingAirport': starting_airport,
    'destinationAirport': destination_airport,
    'flightDate': st.date_input('Select flight date:'),
    'segmentsDepartureTimeRaw': departure_time,
    'segmentsCabinCode': st.selectbox('Choose cabin type:', options=['coach', 'premium coach', 'first', 'business'])
}

# Create a "Predict" button
if st.button("Predict"):
    # Preprocess user input
    data_preprocessor = DataPreprocessor()
    preprocessed_input = data_preprocessor.preprocess_user_input(
        user_input,
        'preprocessor and mappings/preprocessor_dl.joblib',
        'preprocessor and mappings/category_mappings_dl.joblib',
        'preprocessor and mappings/avg_features_dl.csv'
    )

    ml_preprocessor = PreProcessor()
    prediction_ronik = ml_preprocessor.preprocess_for_user_input(user_input, 'preprocessor and mappings/mapped_average_values_ronik.csv')
    prediction_aibarna = ml_preprocessor.preprocess_for_user_input_filtered(user_input, 'data/processed/mapped_average_values_ronik.csv')

    # Paths to all the students' models
    model_student_mapping = {
        "models/best_model-vishal_raj": "Vishal Raj's Wide And Deep Model",
        "models/best_model_Shivatmak": "Shivatmak's LSTM Model",
    }

    # Loop through each model, predict and display results
    for model_path, student_name in model_student_mapping.items():
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        # Identify and print the input shapes
 #       for input_tensor in model.inputs:
 #           st.write(f"Input tensor shape: {input_tensor.shape}")
 #       st.dataframe(preprocessed_input)

        # 1. Other wide features
        wide_features = preprocessed_input[['flightDate_year', 'flightDate_month', 'flightDate_day', 'flightDate_weekday', 'flightDate_is_weekend', 'segmentsDepartureTimeRaw_hour', 'segmentsDepartureTimeRaw_minute']].values

        # 2. startingAirport input
        startingAirport = preprocessed_input[['startingAirport']].values

        # 3. destinationAirport input
        destinationAirport = preprocessed_input[['destinationAirport']].values

        # 4. segmentsCabinCode input
        segmentsCabinCode = preprocessed_input[['segmentsCabinCode']].values

        # 5. Deep features
        deep_features = preprocessed_input[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values

        # 6. Numerical features
        numerical_features = preprocessed_input[['totalTravelDistance', 'segmentsDurationInSeconds', 'segmentsDistance']].values

        # Predict and display results depending on model path
        if "vishal_raj" in model_path:
            predicted_fare = model.predict([wide_features, startingAirport, destinationAirport, segmentsCabinCode, deep_features])
            st.write(f"Prediction from {student_name}: ${predicted_fare[0][0]:.2f}")
        elif "Shivatmak" in model_path:
            predicted_fare1 = model.predict([startingAirport, destinationAirport, segmentsCabinCode, numerical_features])
            st.write(f"Prediction from {student_name}: ${predicted_fare1[0][0]:.2f}")

    st.write(f"Prediction from Ronik's XGBRegressor Model: ${prediction_ronik[0]:.2f}")
    st.write(f"Prediction from Aibarna's Random Regressor Model: ${prediction_aibarna[0]:.2f}")
    