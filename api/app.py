import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model
import joblib

# Load the model
try:
    model = joblib.load('model/sales_model_2024-09-23-06-17-11.pkl')  # Make sure the filename is correct.pkl')  # Make sure the filename is correct
except EOFError:
    print("Error: The model file is empty or corrupted.")
    
# Initialize the label encoder outside the predict function for reuse
label_encoder = LabelEncoder()

def preprocess_input(input_data):
    """Preprocess the input data including feature extraction and encoding."""
    
    # Extract datetime features
    input_data['Date'] = pd.to_datetime(input_data['Date'])
    
    # Extract features
    input_data['DayOfWeek'] = input_data['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    input_data['Weekday'] = input_data['Date'].dt.weekday  # Weekday (0=Monday, 6=Sunday)
    input_data['IsWeekend'] = input_data['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    input_data['Month'] = input_data['Date'].dt.month
    
    # Calculate days to/from a specific holiday (e.g., Christmas)
    input_data['DaysToHoliday'] = (pd.to_datetime('2023-12-25') - input_data['Date']).dt.days
    input_data['DaysAfterHoliday'] = (input_data['Date'] - pd.to_datetime('2023-12-25')).dt.days
    
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
    label_cols = ['StateHoliday', 'StoreType', 'Assortment']
    for col in label_cols:
        input_data[col] = label_encoder.fit_transform(input_data[col].astype(str))

    # Ensure all features are aligned with the model's training features
    expected_features = ['Store', 'DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 
                         'CompetitionDistance', 'Promo2', 'Weekday', 'IsWeekend', 
                         'Month', 'DaysToHoliday', 'DaysAfterHoliday', 
                         'IsBeginningOfMonth', 'IsMidMonth', 'IsEndOfMonth', 
                         'IsHoliday', 'Promo_duration']  # Adjust this list accordingly

    # Reindex input_data to match the model's expected input
    for feature in expected_features:
        if feature not in input_data.columns:
            input_data[feature] = 0  # Add missing features with 0

    return input_data[expected_features]  # Return only expected features

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form and convert types
    Store = int(request.form['Store'])
    Open = int(request.form['Open'])
    Promo = int(request.form['Promo'])
    StateHoliday = request.form['StateHoliday']
    SchoolHoliday = int(request.form['SchoolHoliday'])
    StoreType = request.form['StoreType']
    Assortment = request.form['Assortment']
    CompetitionDistance = float(request.form['CompetitionDistance'])
    Promo2 = int(request.form['Promo2'])
    Date = request.form['Date']  # Date as string

    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'Store': [Store],
        'Open': [Open],
        'Promo': [Promo],
        'StateHoliday': [StateHoliday],
        'SchoolHoliday': [SchoolHoliday],
        'StoreType': [StoreType],
        'Assortment': [Assortment],
        'CompetitionDistance': [CompetitionDistance],
        'Promo2': [Promo2],
        'Date': [Date]
    })

    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Format the prediction for output
    prediction_value = prediction[0]
    return render_template('index.html', prediction=f"Predicted Sales: {prediction_value}")

if __name__ == '__main__':
    app.run(debug=True)
