import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_input(input_data):
    """Preprocess the input data including feature extraction and encoding."""
    # Convert 'Date' column to datetime
    input_data['Date'] = pd.to_datetime(input_data['Date'])

    # Extract features
    input_data['DayOfWeek'] = input_data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    input_data['IsWeekday'] = input_data['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
    input_data['IsWeekend'] = input_data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    input_data['Month'] = input_data['Date'].dt.month
    
    # Calculate days to/from Christmas
    # input_data['DaysToHoliday'] = (pd.to_datetime('2023-12-25') - input_data['Date']).dt.days
    # input_data['DaysAfterHoliday'] = (input_data['Date'] - pd.to_datetime('2023-12-25')).dt.days
    
    # Identify periods within the month
    input_data['IsBeginningOfMonth'] = (input_data['Date'].dt.day <= 7).astype(int)
    input_data['IsMidMonth'] = ((input_data['Date'].dt.day > 7) & (input_data['Date'].dt.day <= 21)).astype(int)
    input_data['IsEndOfMonth'] = (input_data['Date'].dt.day > 21).astype(int)

    # Drop unnecessary columns
    input_data.drop(columns=['Date'], inplace=True)

    # Feature engineering
    input_data['IsHoliday'] = input_data.apply(lambda x: 1 if (x['StateHoliday'] != '0' or x['SchoolHoliday'] == 1) else 0, axis=1)
    input_data['Promo_duration'] = input_data.groupby('Store')['Promo'].cumsum()

    # Encode categorical variables
    encode_categorical_features(input_data)

    # Ensure all features are aligned with the model's training features
    return align_features(input_data)

def encode_categorical_features(input_data):
    """Encode categorical variables using label encoding."""
    label_encoder = LabelEncoder()
    label_cols = ['StateHoliday', 'StoreType', 'Assortment']
    for col in label_cols:
        input_data[col] = label_encoder.fit_transform(input_data[col].astype(str))

def align_features(input_data):
    """Align the input DataFrame with the model's expected features."""
    expected_features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 
                         'SchoolHoliday', 'StoreType', 'Assortment', 
                         'CompetitionDistance', 'Promo2', 'IsWeekday', 
                         'IsWeekend', 'Month', 'IsBeginningOfMonth', 
                         'IsMidMonth', 'IsEndOfMonth', 'IsHoliday', 
                         'Promo_duration']  # Adjust as needed

    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Add missing features with default value of 0

    return input_data[expected_features]  # Return only expected features
