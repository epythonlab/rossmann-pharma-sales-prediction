import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from preprocessing import preprocess_input  # Import the preprocessing functions
from load_model import load_model
# Initialize Flask app
app = Flask(__name__)

# Load the model once at the start
model = load_model('model/sales_model_2024-09-23-16-22-02.pkl')

@app.route('/', methods=['GET'])
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get data from the form and convert types
        input_data = {
            'Store': int(request.form['Store']),
            'Open': int(request.form['Open']),
            'Promo': int(request.form['Promo']),
            'StateHoliday': request.form['StateHoliday'],
            'SchoolHoliday': int(request.form['SchoolHoliday']),
            'StoreType': request.form['StoreType'],
            'Assortment': request.form['Assortment'],
            'CompetitionDistance': float(request.form['CompetitionDistance']),
            'Promo2': int(request.form['Promo2']),
            'Date': request.form['Date']  # Date as string
        }

        # Prepare input data as DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        processed_data = preprocess_input(input_df)

        # Make prediction
        prediction = model.predict(processed_data)

        # Format the prediction for output
        prediction_value = round(prediction[0], 2)  # Round to 2 decimal places

        # Return JSON response
        return jsonify({
            'store_id': input_data['Store'],
            'predicted_sales': prediction_value
        })

    except ValueError as ve:
        # Handle value conversion errors (e.g., invalid input)
        return jsonify({'error': 'Invalid input: ' + str(ve)}), 400
    except KeyError as ke:
        # Handle missing keys in the input data
        return jsonify({'error': 'Missing input data: ' + str(ke)}), 400
    except Exception as e:
        # Catch any other unexpected errors
        return jsonify({'error': 'An error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
