import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self, df):
        """
        Initialize the DataPreprocessor with the dataframe.
        
        :param df: Pandas DataFrame containing the dataset to preprocess
        """
        self.df = df.copy()
        self.scaler = StandardScaler()

    def handle_missing_values(self):
        """
        Handles missing values in the dataset. 
        Assumptions: 
        - Columns with more than 50% missing data are dropped.
        - Remaining missing data is imputed using median or mode based on column type.
        """
        # Drop columns with more than 50% missing values
        missing_threshold = 0.5
        missing_percent = self.df.isnull().mean()
        self.df.drop(missing_percent[missing_percent > missing_threshold].index, axis=1, inplace=True)

        # Impute remaining missing values with median for numerical and mode for categorical
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = self.df.select_dtypes(include=['object']).columns

        self.df[num_cols] = num_imputer.fit_transform(self.df[num_cols])
        self.df[cat_cols] = cat_imputer.fit_transform(self.df[cat_cols])

    def encode_categorical_data(self):
        """
        Encodes categorical data using a combination of Label Encoding and One-Hot Encoding.
        - Label Encoding is used for ordinal data.
        - One-Hot Encoding is used for nominal data.
        """
        # Apply Label Encoding for ordinal features like StateHoliday, StoreType, Assortment
        label_cols = ['StateHoliday', 'StoreType', 'Assortment']
        label_encoder = LabelEncoder()

        for col in label_cols:
            self.df[col] = label_encoder.fit_transform(self.df[col])

        # Apply One-Hot Encoding for non-ordinal categorical features like 'DayOfWeek'
        onehot_cols = ['DayOfWeek', 'MonthPosition']
        self.df = pd.get_dummies(self.df, columns=onehot_cols, drop_first=True)

    def extract_datetime_features(self):
        """
        Extracts features from the 'Date' column such as:
        - Day of the week (Monday, Tuesday, etc.)
        - Whether it's a weekend
        - Position within the month (start, mid, or end)
        - Quarter of the year
        """
        # Ensure 'Date' is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Day of the week (0 = Monday, 6 = Sunday)
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek

        # Is weekend (1 = weekend, 0 = weekday)
        self.df['IsWeekend'] = self.df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

        # Start, Mid, End of month
        self.df['MonthPosition'] = self.df['Date'].dt.day.apply(
            lambda x: 'Start' if x <= 10 else ('Mid' if x <= 20 else 'End'))

        # Quarter of the year
        self.df['Quarter'] = self.df['Date'].dt.quarter

        # One-Hot encode MonthPosition
        self.df = pd.get_dummies(self.df, columns=['MonthPosition'], drop_first=True)

    def feature_engineering(self):
        """
        Creates additional features:
        - Days to next holiday and days after the last holiday
        - Number of days since the last promotion and until the next promotion
        - Competition feature based on when the nearest competitor opened
        """
        # Assumes a pre-defined list of holiday dates (holidays should be listed beforehand)
        holidays = pd.to_datetime(['YYYY-MM-DD', 'YYYY-MM-DD'])  # Add real holiday dates here

        # Days to next holiday
        self.df['DaysToNextHoliday'] = self.df['Date'].apply(lambda x: min((holidays - x).days if (holidays - x).days >= 0 else np.nan))

        # Days after the last holiday
        self.df['DaysAfterLastHoliday'] = self.df['Date'].apply(lambda x: min((x - holidays).days if (x - holidays).days >= 0 else np.nan))

        # Days since last promo
        self.df['DaysSinceLastPromo'] = self.df.groupby('Store')['Promo'].apply(lambda x: (x != 0).cumsum())

        # Days until next promo
        self.df['DaysUntilNextPromo'] = self.df.groupby('Store')['Promo'].shift(-1)

        # Competitor open duration (years since nearest competitor opened)
        self.df['CompetitionOpenSince'] = 12 * (self.df['Date'].dt.year - self.df['CompetitionOpenSinceYear']) + \
                                          (self.df['Date'].dt.month - self.df['CompetitionOpenSinceMonth'])

    def scale_numeric_features(self):
        """
        Scales the numeric features using Standard Scaler for uniform scaling.
        This ensures that features like 'Sales', 'Customers', 'CompetitionDistance', etc., are normalized.
        """
        num_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])

    def preprocess(self):
        """
        The main function to run all preprocessing steps sequentially.
        """
        print("Handling missing values...")
        self.handle_missing_values()

        print("Extracting datetime features...")
        self.extract_datetime_features()

        print("Performing feature engineering...")
        self.feature_engineering()

        print("Encoding categorical data...")
        self.encode_categorical_data()

        print("Scaling numeric features...")
        self.scale_numeric_features()

        print("Preprocessing complete.")
        return self.df

# Usage
# df = pd.read_csv('path_to_dataset.csv')  # Load the dataset
# preprocessor = DataPreprocessor(df)
# cleaned_df = preprocessor.preprocess()
