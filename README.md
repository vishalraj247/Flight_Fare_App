# Flight Fare Prediction App

## Overview

The Flight Fare Prediction App is a Streamlit web application that uses deep learning and machine learning to provide real-time flight fare predictions. This repository contains the application's code, models object, and the necessary preprocessing components.

## Repository Structure
```
FLIGHT_FARE_APP/
│
├── models/ # Directory for trained models.
│ ├── best_model/ # Contains the best performing model artifacts for Ronik's XGBRegressor model.
│ ├── best_model_Shivatmak/ # Shivatmak's LSTM model artifacts.
│ ├── best_model-vishal_raj/ # Vishal Raj's Wide and Deep model artifacts.
│ ├── best_model_aibarna/ # Aibarna's Random Forest Regressor model artifacts.
│
├── preprocessor and mappings/ # Directory for data preprocessing and mappings.
│ ├── avg_features_dl.csv # Average features for deep learning model.
│ ├── category_mappings_dl.joblib # Mappings for categorical data preprocessing for deep learning model.
│ ├── mapped_average_values_ronik.csv # Mapped average values for Ronik's XGBRegressor model.
│ ├── preprocessor_dl.joblib # Preprocessor for deep learning model input.
│
├── src/
│ ├── data/ # Scripts for data preprocessing.
│ ├── data_preprocessor.py # Script for DL model preprocessing input data.
│ ├── ml_model_data_preprocessor.py # Script for ML model data preprocessing.
│
├── .gitattributes
├── app.py # Streamlit application.
├── LICENSE
├── README.md
├── requirements.txt # Required libraries to run the app.
```

## Getting Started

### Prerequisites

Before running the app, you will need to have Python installed on your system. The app has been tested on Python 3.8+.

### Installation

1. Clone the repository to your local machine.
2. Navigate to the cloned directory.
3. Install the required dependencies using the following command:

`pip install -r requirements.txt`

### Running the Application

To run the app, execute the following command in the terminal:

`streamlit run app.py`


This will start the Streamlit server and the app will be available in your web browser at the local address provided by Streamlit.

## Usage

Follow the on-screen instructions on the web application to input the required flight details and get fare predictions.

## Contributing

We welcome contributions to improve the app. If you have suggestions or improvements, please fork the repository and create a pull request.

## License

This project is licensed under the terms of the MIT license - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

A big thank you to our team members who have contributed their models and preprocessing expertise to this app:

- Vishal Raj: Wide and Deep Neural Network model, preprocessing and mappings for dl models
- Shivatmak: LSTM model
- Ronik: XGBRegressor model, preprocessing and mappings for ml models
- Aibarna: Random Forest Regressor model

Their hard work and dedication have made this tool what it is.

## Contact

If you have any questions or want to discuss the app further, please open an issue in this repository.