import joblib
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class InsurancePremiumPredictor:
    """
    A class to load the trained model and make insurance premium predictions.
    The loaded model is a pipeline that handles preprocessing internally.
    """
    def __init__(self, model_path='models/best_insurance_premium_model.pkl'):
        """
        Initializes the predictor by loading the trained model pipeline.

        Args:
            model_path (str): Path to the saved model pipeline (.pkl file).
        """
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """
        Loads the trained machine learning model pipeline from the specified path.
        """
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}. "
                  "Please ensure you have run model_training.py and the model is saved.")
            self.model = None
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            self.model = None

    def predict_premium(self, customer_data: pd.DataFrame):
        """
        Predicts the insurance premium for new customer data.

        Args:
            customer_data (pd.DataFrame): A Pandas DataFrame containing the
                                         raw features of the customer(s) for whom
                                         to predict the premium.
                                         Column names must match the original
                                         training data features (excluding target, id, etc.)
                                         before preprocessing.

        Returns:
            np.array: An array of predicted insurance premiums.
                      Returns None if the model is not loaded.
        """
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None

        try:
            predictions = self.model.predict(customer_data)
            # Ensure predictions are non-negative, as premiums cannot be negative
            return np.maximum(predictions, 0)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

if __name__ == "__main__":
    predictor = InsurancePremiumPredictor()

    if predictor.model:
        # Creating a sample DataFrame matching the expected input features
        sample_customer_data = pd.DataFrame([{
            'Age': 30,
            'Gender': 'Male',
            'Annual Income': 60000,
            'Marital Status': 'Married',
            'Number of Dependents': 2,
            'Education Level': 'Bachelor\'s',
            'Occupation': 'Employed',
            'Health Score': 75,
            'Location': 'Urban',
            'Policy Type': 'Comprehensive',
            'Previous Claims': 0,
            'Vehicle Age': 5,
            'Credit Score': 720,
            'Insurance Duration': 3,
            'Smoking Status': 'No',
            'Exercise Frequency': 'Weekly',
            'Property Type': 'House'
        },
        {
            'Age': 45,
            'Gender': 'Female',
            'Annual Income': 90000,
            'Marital Status': 'Divorced',
            'Number of Dependents': 1,
            'Education Level': 'Master\'s',
            'Occupation': 'Self-Employed',
            'Health Score': 88,
            'Location': 'Suburban',
            'Policy Type': 'Premium',
            'Previous Claims': 1,
            'Vehicle Age': 2,
            'Credit Score': 680,
            'Insurance Duration': 5,
            'Smoking Status': 'Yes',
            'Exercise Frequency': 'Daily',
            'Property Type': 'Apartment'
        }])

        print("\nSample customer data for prediction:")
        print(sample_customer_data)

        predicted_premiums = predictor.predict_premium(sample_customer_data)

        if predicted_premiums is not None:
            print("\nPredicted Insurance Premiums:")
            for i, premium in enumerate(predicted_premiums):
                print(f"Customer {i+1}: ${premium:.2f}")
        else:
            print("Prediction failed for sample data.")
    else:
        print("Prediction model not available. Cannot run example prediction.")