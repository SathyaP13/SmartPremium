import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    A class to handle all data loading, exploration, and preprocessing steps
    for the insurance premium prediction project, adapted for train/test split datasets.
    """
    def __init__(self, train_filepath='data/train.csv', test_filepath='data/test.csv'):
        """
        Initializes the DataProcessor with the dataset filepaths.

        Args:
            train_filepath (str): Path to the training dataset CSV file.
            test_filepath (str): Path to the test dataset CSV file.
        """
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.train_df = None
        self.test_df = None
        self.preprocessor = None # To store the ColumnTransformer pipeline
        self.X_train = None
        self.y_train = None
        self.X_test_processed_for_prediction = None # Processed test features
        self.test_ids = None # To store ids from test.csv for submission
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []
        self.date_features = []
        self.target_variable = 'Premium Amount' 

    def load_data(self):
        """
        Loads the training and test datasets from the specified filepaths.
        """
        print(f"Loading training data from: {self.train_filepath}")
        print(f"Loading test data from: {self.test_filepath}")
        try:
            self.train_df = pd.read_csv(self.train_filepath)
            self.test_df = pd.read_csv(self.test_filepath)
            print("Data loaded successfully.")

            # Store test ids for later submission
            if 'id' in self.test_df.columns: 
                self.test_ids = self.test_df['id']
                # Drop id from test_df so it's not treated as a feature
                self.test_df = self.test_df.drop(columns=['id'])
            else:
                print("Warning: 'id' column not found in test.csv. Cannot store test ids.")

            return self.train_df, self.test_df
        except FileNotFoundError:
            print(f"Error: Dataset not found. Check paths: {self.train_filepath}, {self.test_filepath}")
            return None, None
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None, None

    def initial_eda(self):
        """
        Performs initial Exploratory Data Analysis (EDA) on the loaded training dataset.
        Checks for basic structure, missing values, and data types.
        """
        if self.train_df is None:
            print("No training data loaded. Please call load_data() first.")
            return

        print("\n--- Initial Training Data Exploration ---")
        print(f"Training Dataset Shape: {self.train_df.shape} (rows, columns)")
        print("\nFirst 5 rows (Training):")
        print(self.train_df.head())
        print("\nTraining Dataset Info:")
        self.train_df.info()
        print("\nMissing Values (before handling - Training):")
        print(self.train_df.isnull().sum())
        print("\nDescriptive Statistics for Numerical Features (Training):")
        print(self.train_df.describe().T)

        # identify feature types based on training data
        self.numerical_features = self.train_df.select_dtypes(include=np.number).columns.tolist()
        if self.target_variable in self.numerical_features:
            self.numerical_features.remove(self.target_variable)

        self.categorical_features = self.train_df.select_dtypes(include='object').columns.tolist()

        if 'Customer Feedback' in self.categorical_features:
            self.text_features.append('Customer Feedback')
            self.categorical_features.remove('Customer Feedback')
        if 'Policy Start Date' in self.categorical_features:
            self.date_features.append('Policy Start Date')
            self.categorical_features.remove('Policy Start Date')
        if 'id' in self.numerical_features:
            self.numerical_features.remove('id')


        print(f"\nNumerical Features: {self.numerical_features}")
        print(f"Categorical Features: {self.categorical_features}")
        print(f"Text Features (will be ignored): {self.text_features}")
        print(f"Date Features (will be ignored): {self.date_features}")

        # Basic EDA for test data
        if self.test_df is not None:
            print("\n--- Initial Test Data Exploration ---")
            print(f"Test Dataset Shape: {self.test_df.shape} (rows, columns)")
            print("\nTest Dataset Info:")
            self.test_df.info()
            print("\nMissing Values (before handling - Test):")
            print(self.test_df.isnull().sum())


    def visualize_distributions(self):
        """
        Visualizes distributions of key numerical and categorical features in the training data.
        """
        if self.train_df is None:
            print("No training data loaded. Please call load_data() first.")
            return

        print("\n--- Visualizing Distributions (Training Data) ---")

        # Histograms for numerical features
        self.train_df[self.numerical_features + [self.target_variable]].hist(bins=30, figsize=(15, 10), layout=(3, 4))
        plt.suptitle('Histograms of Numerical Features and Premium Amount (Training Data)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Box plots for target variable vs. categorical features (top N)
        for col in self.categorical_features[:5]: # Plotting for a few to avoid too many plots
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=col, y=self.target_variable, data=self.train_df)
            plt.title(f'{self.target_variable} vs. {col} (Training Data)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        # Correlation Heatmap for numerical features
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.train_df[self.numerical_features + [self.target_variable]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix of Numerical Features (Training Data)')
        plt.show()

    def preprocess_data(self):
        """
        Performs data preprocessing including handling missing values,
        converting categorical variables, and feature scaling.
        This method will fit the preprocessor on training data and
        transform both training and test data.
        """
        if self.train_df is None or self.test_df is None:
            print("Training or test data not loaded. Cannot preprocess.")
            return

        print("\n--- Starting Data Preprocessing ---")

        self.train_df = self.train_df.copy()
        self.test_df = self.test_df.copy()

        # 1. Handle Incorrect Data Types and Derived Features (dropping for now)
        cols_to_drop = []
        if 'Policy Start Date' in self.train_df.columns:
            cols_to_drop.append('Policy Start Date')
        if 'Customer Feedback' in self.train_df.columns:
            cols_to_drop.append('Customer Feedback')
        if 'id' in self.train_df.columns:
            cols_to_drop.append('id')

        if cols_to_drop:
            print(f"Dropping columns: {cols_to_drop} from train and test data.")
            self.train_df = self.train_df.drop(columns=cols_to_drop)
            # Ensure these columns exist in test_df before dropping
            for col in cols_to_drop:
                if col in self.test_df.columns:
                    self.test_df = self.test_df.drop(columns=[col])

        # Update numerical and categorical features after dropping columns
        # These lists are used by ColumnTransformer, so they should reflect actual columns
        self.numerical_features = self.train_df.select_dtypes(include=np.number).columns.tolist()
        if self.target_variable in self.numerical_features: 
            self.numerical_features.remove(self.target_variable)
        self.categorical_features = self.train_df.select_dtypes(include='object').columns.tolist()

        print(f"Numerical Features (after initial cleaning): {self.numerical_features}")
        print(f"Categorical Features (after initial cleaning): {self.categorical_features}")

        # Define preprocessing steps for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )

        print("\nPreprocessing pipeline created:")
        print(self.preprocessor)

        # Separate features (X) and target (y) for training data
        self.X_train = self.train_df.drop(columns=[self.target_variable])
        self.y_train = self.train_df[self.target_variable]

        print("Data preprocessing setup complete. Preprocessor ready to be fit_transformed.")

    def get_processed_data_for_training(self):
        """
        Returns the preprocessor, training features (X_train), and training target (y_train).
        """
        if self.preprocessor is None or self.X_train is None or self.y_train is None:
            print("Training data not fully prepared. Please run preprocess_data() first.")
            return None, None, None
        return self.preprocessor, self.X_train, self.y_train

    def get_test_data_for_prediction(self):
        """
        Returns the raw test features (test_df) and test ids.
        The preprocessor in the model pipeline will handle transformation.
        """
        if self.test_df is None:
            print("Test data not loaded. Cannot provide test data for prediction.")
            return None, None
        return self.test_df, self.test_ids

if __name__ == "__main__":
    processor = DataProcessor(train_filepath='data/train.csv', test_filepath='data/test.csv')

    # Step 1: Load and Explore Data
    train_df, test_df = processor.load_data()
    if train_df is not None and test_df is not None:
        processor.initial_eda()
        processor.visualize_distributions() 

        # Step 2: Data Preprocessing Setup
        processor.preprocess_data()

        preprocessor, X_train, y_train = processor.get_processed_data_for_training()
        X_test_raw, test_ids = processor.get_test_data_for_prediction()

        if preprocessor is not None:
            print(f"\nShape of X_train before preprocessing: {X_train.shape}")
            print(f"Shape of X_test_raw before preprocessing: {X_test_raw.shape}")
        else:
            print("Preprocessor not created.")
    else:
        print("Data loading failed. Cannot proceed with data preparation.")