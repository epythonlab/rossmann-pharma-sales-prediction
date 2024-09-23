import joblib

class Model:
    def __init__(self):
        self.model_pipeline = None

    def load_model(self, filename):
        """
        Loads a trained model from a file.
        
        Parameters
        ----------
        filename : str
            The name of the file from which the model will be loaded.
        
        Returns
        -------
        None
        """
        self.model_pipeline = joblib.load(filename)

    def predict(self, input_data):
        """
        Makes predictions using the loaded model.
        
        Parameters
        ----------
        input_data : dict
            Input data for making predictions.
        
        Returns
        -------
        prediction : Any
            The prediction result from the model.
        """
        if self.model_pipeline is None:
            raise ValueError("Model not loaded. Please load a model before prediction.")
        
        # Preprocess input_data if needed, then make prediction
        prediction = self.model_pipeline.predict(input_data)
        return prediction
