import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split # Re-introduce for validation split
import joblib
import mlflow
import mlflow.sklearn
from datapreprocess import DataProcessor
import warnings

warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A class to handle model selection, training, evaluation on a validation set,
    and MLflow integration, adapted for train/test datasets.
    """
    def __init__(self, preprocessor, X_train_full, y_train_full):
        """
        Initializes the ModelTrainer with the full training data and preprocessor.

        Args:
            preprocessor (sklearn.compose.ColumnTransformer): The data preprocessor.
            X_train_full (pd.DataFrame): Full training features (from train.csv).
            y_train_full (pd.Series): Full training target (from train.csv).
        """
        self.preprocessor = preprocessor
        self.X_train_full = X_train_full
        self.y_train_full = y_train_full
        self.X_train_split = None # Actual training data for model
        self.X_val_split = None   # Validation data for evaluation
        self.y_train_split = None
        self.y_val_split = None
        self._perform_validation_split() # Perform internal validation split

        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
        }
        self.best_model_pipeline = None # Stores the full pipeline (preprocessor + best model)
        self.best_model_name = None
        mlflow.set_experiment("Insurance Premium Prediction (Train/Validation Split)")

    def _perform_validation_split(self, test_size=0.2, random_state=42):
        """
        Splits the provided full training data into training and validation sets.
        This is for internal model evaluation, not for the final test.csv.
        """
        print(f"\n--- Splitting Training Data into Train ({(1-test_size)*100}%) "
              f"and Validation ({test_size*100}%) ---")
        self.X_train_split, self.X_val_split, self.y_train_split, self.y_val_split = train_test_split(
            self.X_train_full, self.y_train_full, test_size=test_size, random_state=random_state
        )
        print(f"X_train_split shape: {self.X_train_split.shape}")
        print(f"X_val_split shape: {self.X_val_split.shape}")
        print(f"y_train_split shape: {self.y_train_split.shape}")
        print(f"y_val_split shape: {self.y_val_split.shape}")
        print("Validation split complete.")


    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculates and returns evaluation metrics.
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        y_pred_positive = np.maximum(y_pred, 0)
        rmsle = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_positive + 1e-9)))

        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'RMSLE': rmsle
        }

    def train_and_evaluate(self):
        """
        Trains and evaluates each defined model using the train/validation split,
        logging results with MLflow. Selects the best performing model based on RMSE.
        """
        print("\n--- Starting Model Training and Evaluation on Validation Set ---")
        best_rmse = float('inf')

        for name, model in self.models.items():
            with mlflow.start_run(run_name=name):
                print(f"\nTraining and evaluating: {name}")

                # Build the full pipeline: preprocessor + model
                # The preprocessor is fit on X_train_split and then transforms both splits
                full_pipeline = Pipeline(steps=[
                    ('preprocessor', self.preprocessor),
                    ('regressor', model)
                ])

                # Train the model on the X_train_split
                full_pipeline.fit(self.X_train_split, self.y_train_split)

                # Make predictions on the validation set
                y_pred_val = full_pipeline.predict(self.X_val_split)
                val_metrics = self._calculate_metrics(self.y_val_split, y_pred_val)

                # Also get train metrics for comparison 
                y_pred_train = full_pipeline.predict(self.X_train_split)
                train_metrics = self._calculate_metrics(self.y_train_split, y_pred_train)


                print(f"--- {name} Results ---")
                print("Train Metrics:", train_metrics)
                print("Validation Metrics:", val_metrics) 

                # MLflow Logging
                mlflow.log_params({
                    "model_name": name,
                    "target_variable": "Premium Amount",
                    "validation_split_ratio": 0.2
                })
                mlflow.log_metrics({f"train_{k.lower()}": v for k, v in train_metrics.items()})
                mlflow.log_metrics({f"validation_{k.lower()}": v for k, v in val_metrics.items()}) 

                mlflow.sklearn.log_model(full_pipeline, "model")

                # Check for the best model based on validation RMSE
                if val_metrics['RMSE'] < best_rmse:
                    best_rmse = val_metrics['RMSE']
                    self.best_model_pipeline = full_pipeline
                    self.best_model_name = name
                    print(f"New best model found: {name} with Validation RMSE: {val_metrics['RMSE']:.4f}")

        print(f"\n--- Model Training and Evaluation Complete ---")
        if self.best_model_pipeline:
            print(f"Best Model Selected: {self.best_model_name} with Validation RMSE: {best_rmse:.4f}")
        else:
            print("No models were trained.")

    def save_best_model(self, path='models/best_insurance_premium_model.pkl'):
        """
        Saves the best performing model pipeline.
        """
        if self.best_model_pipeline:
            joblib.dump(self.best_model_pipeline, path)
            print(f"Best model pipeline saved to {path}")
        else:
            print("No best model pipeline found to save.")

    def get_best_model_pipeline(self):
        """
        Returns the best trained model pipeline.
        """
        return self.best_model_pipeline


if __name__ == "__main__":
    # Ensure data/train.csv and data/test.csv exist and are accessible
    data_processor = DataProcessor(train_filepath='data/train.csv', test_filepath='data/test.csv')
    train_df, test_df = data_processor.load_data()

    if train_df is not None and test_df is not None:
        data_processor.initial_eda()
        data_processor.preprocess_data() # Sets up preprocessor, X_train_full, y_train_full

        preprocessor, X_train_full, y_train_full = data_processor.get_processed_data_for_training()

        if preprocessor is not None:
            # Initialize and run model training with the full training data
            trainer = ModelTrainer(preprocessor, X_train_full, y_train_full)
            trainer.train_and_evaluate()
            trainer.save_best_model()

            # Making predictions on the actual test.csv data using the best model
            best_model = trainer.get_best_model_pipeline()
            X_test_raw_for_prediction, test_ids = data_processor.get_test_data_for_prediction()

            if best_model and X_test_raw_for_prediction is not None and test_ids is not None:
                print("\n--- Making predictions on test.csv data ---")
                final_predictions = best_model.predict(X_test_raw_for_prediction)
                final_predictions = np.maximum(final_predictions, 0) 

                # Create submission file
                submission_df = pd.DataFrame({
                    'ID': test_ids,
                    'Premium Amount': final_predictions
                })
                submission_filepath = 'data/submission.csv' 
                submission_df.to_csv(submission_filepath, index=False)
                print(f"Final predictions saved to {submission_filepath}")
                print(submission_df.head())
            else:
                print("Could not make final predictions on test.csv. Model or test data missing.")
        else:
            print("Preprocessing failed. Cannot proceed with model training.")
    else:
        print("Data loading failed. Cannot proceed with model training.")