# Load the trained machine learning model
import joblib
def load_model(model_path):
    """Load the trained model from the specified path."""
    try:
        model = joblib.load(model_path)
        return model
    except EOFError:
        print("Error: The model file is empty or corrupted.")
        return None