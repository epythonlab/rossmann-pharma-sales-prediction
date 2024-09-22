import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

class DataPreprocessor:
    def __init__(self, train_path, test_path):
        """
        Initialize the DataPreprocessor with paths to the training and testing datasets.
        - Loads data from the specified paths.
        - Applies a dtype dictionary to ensure proper data types.
        """
        # Define the data types for specific columns
        dtype_dict = {
            'Store': int,
            'Sales': float,
            'Customers': float,
            'Open': float,
            'StateHoliday': str,
            'SchoolHoliday': float,
            'StoreType': str,
            'Assortment': str,
            'CompetitionDistance': float,
            'Promo': float,
            'Promo2': float
        }
        # Load training and test datasets
        self.train_df = pd.read_csv(train_path, dtype=dtype_dict, low_memory=False)
        self.test_df = pd.read_csv(test_path, dtype=dtype_dict, low_memory=False)
        
        # Initialize the scaler for numerical feature scaling
        self.scaler = StandardScaler()
    
    def clean_data(self):
        """
        Clean the data by resetting the index and handling missing 'Id' columns.
        Calls the method to handle missing values.
        """
        self.train_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)
        
        # Drop 'Id' if it exists in either train or test dataset
        if 'Id' in self.train_df.columns:
            self.train_df = self.train_df.drop(columns=['Id'])
        if 'Id' in self.test_df.columns:
            self.test_df = self.test_df.drop(columns=['Id'])
        
        # Handle missing values in the datasets
        self.handle_missing_values()

    def handle_missing_values(self):
        """
        Handle missing values by imputing with median and mode as necessary.
        This method also removes unnecessary columns like 'Promo2SinceYear' and 'Promo2SinceWeek'.
        """
        # Combine train and test for consistent handling of missing values
        combined_df = pd.concat([self.train_df, self.test_df], axis=0, keys=['train', 'test'])
        
        # Fill missing values for 'CompetitionDistance' with the median
        combined_df['CompetitionDistance'] = combined_df['CompetitionDistance'].fillna(combined_df['CompetitionDistance'].median())
        
        # Drop unnecessary columns if they exist
        if 'Promo2SinceYear' in combined_df.columns:
            combined_df = combined_df.drop(columns=['Promo2SinceYear'])
        if 'Promo2SinceWeek' in combined_df.columns:
            combined_df = combined_df.drop(columns=['Promo2SinceWeek'])
        if 'CompetitionOpenSinceYear' in combined_df.columns:
            combined_df = combined_df.drop(columns=['CompetitionOpenSinceYear'])
        
        if 'CompetitionOpenSinceMonth' in combined_df.columns:
            combined_df = combined_df.drop(columns=['CompetitionOpenSinceMonth'])
        

        # Split the data back into train and test after handling missing values
        self.train_df = combined_df.xs('train')
        self.test_df = combined_df.xs('test')

    def extract_datetime_features(self):
        """
        Extract useful datetime features such as day of the week, month, and year.
        Additionally, calculate features like whether the day is a weekend, and 
        proximity to a holiday.
        """
        for df in [self.train_df, self.test_df]:
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is a datetime object
            df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if weekend
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            
            # Calculate days to and after a holiday (example for Christmas)
            df['DaysToHoliday'] = (pd.to_datetime('2023-12-25') - df['Date']).dt.days
            df['DaysAfterHoliday'] = (df['Date'] - pd.to_datetime('2023-12-25')).dt.days
            
            # Create flags for beginning, mid, and end of the month
            df['IsBeginningOfMonth'] = (df['Date'].dt.day <= 7).astype(int)
            df['IsMidMonth'] = df['Date'].dt.day.between(8, 21).astype(int)
            df['IsEndOfMonth'] = (df['Date'].dt.day > 21).astype(int)
            
            # Drop the original 'Date' column if no longer needed
            df.drop(columns=['Date'], inplace=True)

    def feature_engineering(self):
        """
        Perform feature engineering to create new features:
        - 'CompetitionOpenSince' duration in months.
        - A holiday flag indicating if a store is open during a holiday.
        """
        for df in [self.train_df, self.test_df]:
            # Calculate the competition open duration in months
            # df['CompetitionOpenSince'] = (df['Year'] - df['CompetitionOpenSinceYear']) * 12 + (df['Month'] - df['CompetitionOpenSinceMonth'])
            # df['CompetitionOpenSince'] = df['CompetitionOpenSince'].fillna(0)  # Fill missing values
            
            # Create a new feature to indicate if the store is open during holidays
            df['IsHoliday'] = df.apply(lambda x: 1 if (x['StateHoliday'] != '0' or x['SchoolHoliday'] == 1) else 0, axis=1)

    def encode_categorical_data(self):
        """
        Encode categorical variables such as 'StateHoliday', 'StoreType', and 'Assortment'
        using label encoding.
        """
        label_cols = ['StateHoliday', 'StoreType', 'Assortment']
        label_encoder = LabelEncoder()

        # Apply LabelEncoder to categorical columns in both train and test datasets
        for col in label_cols:
            for df in [self.train_df, self.test_df]:
                df[col] = label_encoder.fit_transform(df[col])

    def scale_numeric_features(self):
        """
        Scale numeric features using StandardScaler.
        Ensure consistent scaling between train and test data by only scaling 
        common numeric columns that exist in both datasets.
        """
        # Drop the 'Dataset' column if it exists (if not necessary)
        self.train_df.drop(columns=['Dataset'], inplace=True, errors='ignore')
        self.test_df.drop(columns=['Dataset'], inplace=True, errors='ignore')

        # Drop 'Sales' and 'Customers' from test if they exist, as they are only in the train set
        for col in ['Sales', 'Customers']:
            if col in self.test_df.columns:
                self.test_df.drop(columns=[col], inplace=True)

        # Get numeric columns in train and test datasets
        num_cols_train = self.train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        num_cols_test = self.test_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Identify common numeric columns between train and test
        common_cols = [col for col in num_cols_train if col in num_cols_test]

        # Scale the common numeric columns in both train and test datasets
        self.train_df[common_cols] = self.scaler.fit_transform(self.train_df[common_cols])
        self.test_df[common_cols] = self.scaler.transform(self.test_df[common_cols])

    def preprocess(self):
        """
        Execute the full preprocessing pipeline:
        1. Clean data
        2. Extract datetime features
        3. Perform feature engineering
        4. Encode categorical data
        5. Scale numeric features
        """
        print("Cleaning data...")
        self.clean_data()

        print("Extracting datetime features...")
        self.extract_datetime_features()

        print("Performing feature engineering...")
        self.feature_engineering()

        print("Encoding categorical data...")
        self.encode_categorical_data()

        print("Scaling numeric features...")
        self.scale_numeric_features()

        print("Preprocessing complete.")
        return self.train_df, self.test_df

    def save_data(self, train_file='train_processed.csv', test_file='test_processed.csv'):
        """
        Save the preprocessed train and test data to CSV files.
        """       
        # Save the datasets to csv
        self.train_df.to_csv('../data/'+ train_file, index=False)
        self.test_df.to_csv('../data/'+ test_file, index=False)
        print(f"Processed data saved to {train_file} and {test_file}.")
