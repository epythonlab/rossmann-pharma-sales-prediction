import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, train_path: str, test_path: str, test_id: str):
        """
        Initialize the DataPreprocessor with paths to the training and testing datasets.
        
        Parameters
        ----------
        train_path : str
            Path to the training dataset.
        test_path : str
            Path to the testing dataset.
        test_id : str
            Path to the CSV file containing test IDs.
        """
        dtype_dict = {
            'Store': int,
            'Sales': float,
            'Open': float,
            'StateHoliday': str,
            'SchoolHoliday': float,
            'Promo': float
        }
        
        self.train_data = pd.read_csv(train_path, dtype=dtype_dict, low_memory=False)
        self.test_data = pd.read_csv(test_path, dtype=dtype_dict, low_memory=False)
        self.test_id = pd.read_csv(test_id, dtype={'Id': int}, low_memory=False)
        self.test_data['Id'] = self.test_id['Id']
        
        self.train_data = self.train_data[self.train_data['Open'] == 1]
        self.train_df = self.train_data.copy()
        self.test_df = self.test_data.copy()
        self.scaler = StandardScaler()

    def clean_data(self):
        """Clean the datasets by resetting indexes and dropping unnecessary columns."""
        self.train_df.reset_index(drop=True, inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)
        self.train_df.drop(columns=['Customers'], errors='ignore', inplace=True)
        self.test_df.drop(columns=['Id'], errors='ignore', inplace=True)
        self.handle_missing_values()

    def handle_missing_values(self):
        """Handle missing values consistently across train and test datasets."""
        combined_df = pd.concat([self.train_df, self.test_df], axis=0, keys=['train', 'test'])
        combined_df.fillna({'Open': combined_df['Open'].mode()[0]}, inplace=True)
        self.train_df = combined_df.xs('train')
        self.test_df = combined_df.xs('test')

    def extract_datetime_features(self):
        """Extract datetime features such as weekday, month, and holiday-related variables."""
        for df in [self.train_df, self.test_df]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['IsWeekday'] = df['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df['Month'] = df['Date'].dt.month
            df['IsBeginningOfMonth'] = (df['Date'].dt.day <= 7).astype(int)
            df['IsMidMonth'] = ((df['Date'].dt.day > 7) & (df['Date'].dt.day <= 21)).astype(int)
            df['IsEndOfMonth'] = (df['Date'].dt.day > 21).astype(int)
            df.drop(columns=['Date', 'Dataset', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)

    def feature_engineering(self):
        """Create new features based on existing data, such as holiday flags and promo duration."""
        for df in [self.train_df, self.test_df]:
            df['IsHoliday'] = df.apply(lambda x: 1 if (x['StateHoliday'] != '0' or x['SchoolHoliday'] == 1) else 0, axis=1)
            df['Promo_duration'] = df.groupby('Store')['Promo'].cumsum()

   
    def encode_categorical_data(self):
        """Encode categorical variables using label encoding."""
        label_cols = ['StateHoliday', 'StoreType', 'Assortment']
        label_encoder = LabelEncoder()

        for col in label_cols:
            self.train_df[col] = label_encoder.fit_transform(self.train_df[col].astype(str))
            self.test_df[col] = label_encoder.transform(self.test_df[col].astype(str))  # Use transform on test

    def preprocess(self):
        """Execute the full preprocessing pipeline."""
        print("Cleaning data...")
        self.clean_data()
        print("Extracting datetime features...")
        self.extract_datetime_features()
        print("Performing feature engineering...")
        self.feature_engineering()
        print("Encoding categorical data...")
        self.encode_categorical_data()

        self.test_df.drop(columns=['Sales'], errors='ignore', inplace=True)
        self.test_df.reset_index(drop=True, inplace=True)
        self.test_df.set_index(self.test_data['Id'], inplace=True)
        self.train_df.reset_index(drop=True, inplace=True)
        self.train_df.set_index(self.train_data['Date'], inplace=True)
        
        print("Preprocessing complete.")
        return self.train_df, self.test_df

    def save_data(self, train_file='../data/train_processed.csv', test_file='../data/test_processed.csv'):
        """Save the preprocessed train and test data to CSV files."""
        self.train_df.to_csv(train_file, index=True)
        self.test_df.to_csv(test_file, index=True)
        print(f"Processed data saved to {train_file} and {test_file}.")
