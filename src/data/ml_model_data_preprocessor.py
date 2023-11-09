import pandas as pd
import numpy as np
import os
import joblib

class PreProcessor:
    def __init__(self):
        self.data = None
        self.preprocessed_data = None
        self.user_df = None

    def split_and_explode(self, columns_to_explode):
        """
        Split and explode columns based on '||' delimiter.
        
        columns_to_explode: List of column names to explode
        """
        # Diagnostic print before explosion
        print("Data before explosion:")
        print(self.data.head())
        # Fill NaN values in segmentsDistance with 'None' before splitting and exploding
        if 'segmentsDistance' in columns_to_explode:
            self.data['segmentsDistance'].fillna('None', inplace=True)

        # Number of splits for each row across the first column
        splits = self.data[columns_to_explode[0]].str.split('\|\|').apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Ensure same number of splits across all columns
        for col in columns_to_explode[1:]:
            col_splits = self.data[col].str.split('\|\|').apply(lambda x: len(x) if isinstance(x, list) else 0)
            if not all(col_splits == splits):
                raise ValueError(f"Columns {columns_to_explode[0]} and {col} do not have the same number of '||' splits.")
        
        # Split and explode
        for col in columns_to_explode:
            self.data[col] = self.data[col].str.split('\|\|')
        
        # Using pandas' explode simultaneously on all columns
        for col in columns_to_explode:
            self.data = self.data.explode(col)

        # Reset the index to ensure unique indices
        self.data.reset_index(drop=True, inplace=True)
        print(f"Number of Data Before dropping duplicates:{len(self.data)}")
        self.data = self.data.drop_duplicates()
        print(f"Number of Data After dropping duplicates:{len(self.data)}")

        # Diagnostic print after explosion
        print("Data after explosion:")
        print(self.data.head())
        # Check if the column is the datetime column 'segmentsDepartureTimeRaw'
        if 'segmentsDepartureTimeRaw' in columns_to_explode:
            # Ensure that the exploded values are valid datetime strings
            self.data['segmentsDepartureTimeRaw'] = self.data['segmentsDepartureTimeRaw'].str.extract(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')[0]

    def get_date_time(self, df):
        
        df['segmentsDepartureTimeRaw'] = pd.to_datetime(df['segmentsDepartureTimeRaw'])
        df['year'] = df['segmentsDepartureTimeRaw'].dt.year
        df['month'] = df['segmentsDepartureTimeRaw'].dt.month
        df['day'] = df['segmentsDepartureTimeRaw'].dt.day
        df['hour'] = df['segmentsDepartureTimeRaw'].dt.hour
        df['minute'] = df['segmentsDepartureTimeRaw'].dt.minute

        df = df.drop(['flightDate', 'segmentsDepartureTimeRaw'], axis=1)
        return df


    def imputation_numerical(self, df):
        df['totalTravelDistance'].fillna(df.groupby(['startingAirport', 'destinationAirport'])['totalTravelDistance'].transform(lambda x: x.mode().max()), inplace=True)
        df['segmentsDistance'].fillna(df.groupby(['startingAirport', 'destinationAirport'])['segmentsDistance'].transform(lambda x: x.mode().max()), inplace=True)
        return df
    
    def downcast(self, df):
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32']).columns
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, downcast='integer', errors='ignore')
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, downcast='float', errors='ignore')
        return df
    
    def map_categorical_features(self, df):
        airport_code_map = {
            'BOS': 1,'CLT': 2,'DEN': 3,'DFW': 4,'DTW': 5,'EWR': 6,'IAD': 7,'JFK': 8,'LAX': 9,'LGA': 10,
            'MIA': 11,'OAK': 12,'ORD': 13,'PHL': 14,'SFO': 15,'ATL': 16,'unknown': 0
                            }
        cabin_code_map = {
            'coach':1,
            'first':2,
            'premium coach':3,
            'business':4,
            'unknown':0
        }
        df['startingAirport'] = df['startingAirport'].map(airport_code_map)
        df['destinationAirport'] = df['destinationAirport'].map(airport_code_map)
        df['segmentsCabinCode'] = df['segmentsCabinCode'].map(cabin_code_map)

        return df
    
    def mode_fare(self, df):
        mode_fare_list = ['startingAirport', 'destinationAirport','segmentsCabinCode','year','month','day',
                          'hour','minute']
        mode_values = df.groupby(mode_fare_list)['totalFare'].agg(lambda x:x.mode()[0]).reset_index(name="Total_fare_mode")
        df = df.merge(mode_values, on=mode_fare_list,how='inner')
        print(f'Before Duplicates: {len(df)}')
        df = df.drop_duplicates()
        df = df.drop(['totalFare'], axis=1)
        print(f'After Duplicates:{len(df)}')
        return df


    def preprocess_data(self):
        base_path = 'data/processed'
        file = 'exploded_merged_data.csv'
        self.preprocessed_data = pd.read_csv(os.path.join(base_path, file))
        print(self.preprocessed_data)
        self.preprocessed_data = self.get_date_time(self.preprocessed_data)

        # Downcasting to save memory
        self.preprocessed_data = self.downcast(self.preprocessed_data)

        # Imputing numerical variables with mode values 
        self.preprocessed_data = self.imputation_numerical(self.preprocessed_data)

        # Mapping Categorical Features:
        self.preprocessed_data = self.map_categorical_features(self.preprocessed_data)

        # Downcasting again on converted featuers to save memory
        self.preprocessed_data = self.downcast(self.preprocessed_data)

        # Converting Fare to Maximum Mode values
        self.preprocessed_data = self.mode_fare(self.preprocessed_data)

        return self.preprocessed_data


    def merge_data_and_explode(self):
        """
        Merge all datasets, preprocess the merged dataset, save the preprocessed data,
        save category mappings, save average features, and the preprocessor.
        """
        # Merge datasets by reading from the airport folders in data/interim
        print("split / explode:100 percent of total data")

        data_frames = []
        base_path = 'data/interim'
        required_column_names = ['flightDate', 'startingAirport',
                         'destinationAirport', 'totalFare', 'totalTravelDistance',
                         'segmentsDepartureTimeRaw','segmentsDurationInSeconds',
                         'segmentsDistance', 'segmentsCabinCode']
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.csv'):
                        df = pd.read_csv(os.path.join(folder_path, file), usecols=required_column_names)
                        print(f'Data Size before dropping columns:{len(df)}')
                        df = df.drop_duplicates()
                        print(f'Data Size after dropping columns:{len(df)}')
                        data_frames.append(df)
        
        # Concatenate all dataframes into one
        self.data = pd.concat(data_frames, ignore_index=True)
        print(f'Total MERGED Data size:{len(self.data)}')

        columns_to_split_and_explode = [
            'segmentsDepartureTimeRaw', 'segmentsDurationInSeconds',
            'segmentsDistance', 'segmentsCabinCode'
        ]
        self.split_and_explode(columns_to_split_and_explode)
        self.data.to_csv(f'data/processed/exploded_merged_data.csv', index=False)


    def preprocess_for_user_input(self, user_input, average_values_path):
        self.user_df = pd.DataFrame([user_input])
        if self.user_df['segmentsDepartureTimeRaw'].dtype == 'object':

            self.user_df['flightDate'] = pd.to_datetime(self.user_df['flightDate'].astype(str), format='%Y-%m-%d', errors='coerce')
            self.user_df['segmentsDepartureTimeRaw'] = self.user_df['segmentsDepartureTimeRaw'].astype(str)
            self.user_df['segmentsDepartureTimeRaw'] = self.user_df['flightDate'].astype(str) + ' ' + self.user_df['segmentsDepartureTimeRaw']
            self.user_df['segmentsDepartureTimeRaw'] = pd.to_datetime(self.user_df['segmentsDepartureTimeRaw'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            self.user_df = self.get_date_time(self.user_df)
            self.user_df = self.map_categorical_features(self.user_df)
            average_values = pd.read_csv(average_values_path)
            self.user_df = self.user_df.merge(average_values, on=['startingAirport', 'destinationAirport'], how='left')
            self.user_df = self.user_df[['startingAirport', 'destinationAirport', 'totalTravelDistance', 'segmentsDurationInSeconds',
                                            'segmentsDistance', 'segmentsCabinCode', 'year', 'month', 'day',
                                            'hour', 'minute']]
            model_xgb = joblib.load("models/best_model/best-model-ronik/best_model_ronik_final.pb")
            prediction = model_xgb.predict(self.user_df)

            return np.array(prediction)
        
    def preprocess_for_user_input_filtered(self, user_input, average_values_path):
        self.user_df = pd.DataFrame([user_input])
        if self.user_df['segmentsDepartureTimeRaw'].dtype == 'object':

            self.user_df['flightDate'] = pd.to_datetime(self.user_df['flightDate'].astype(str), format='%Y-%m-%d', errors='coerce')
            self.user_df['segmentsDepartureTimeRaw'] = self.user_df['segmentsDepartureTimeRaw'].astype(str)
            self.user_df['segmentsDepartureTimeRaw'] = self.user_df['flightDate'].astype(str) + ' ' + self.user_df['segmentsDepartureTimeRaw']
            self.user_df['segmentsDepartureTimeRaw'] = pd.to_datetime(self.user_df['segmentsDepartureTimeRaw'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            self.user_df = self.get_date_time(self.user_df)
            self.user_df = self.map_categorical_features(self.user_df)
            average_values = pd.read_csv(average_values_path)
            self.user_df = self.user_df.merge(average_values, on=['startingAirport', 'destinationAirport'], how='left')
            self.user_df = self.user_df[['startingAirport', 'destinationAirport',
                                            'segmentsCabinCode', 'year', 'month', 'day',
                                            'hour', 'minute',
                                           ]]
            model_xgb = joblib.load("models/best_model_aibarna/best_model_aibarna_final.pb")
            prediction = model_xgb.predict(self.user_df)

            return np.array(prediction)
        
    