# Smart Premium
# 💰 Insurance Premium Prediction
- This project aims to build a machine learning model that accurately predicts insurance premiums based on various customer characteristics and policy details.
- It follows a structured MLOps-lite approach, covering data preprocessing, model training with MLflow tracking, and deployment using Streamlit.

## 🌟 Features

* **Data Preprocessing:** Handles missing values, encodes categorical features, and scales numerical data.
* **Multiple Regression Models:** Explores and evaluates Linear Regression, Decision Trees, Random Forest, and XGBoost.
* **MLflow Integration:** Tracks experiments, logs model metrics, parameters, and artifacts for reproducibility and comparison.
* **Automated ML Pipeline:** Combines preprocessing and model training into a streamlined workflow.
* **Interactive Web Application:** A user-friendly Streamlit interface for real-time premium predictions.
* **Modular Codebase:** Organized into `data`, `models` directories for clarity and maintainability.

## 🛠️ Technologies Used

* **Python**
* **Pandas**, **NumPy**
* **Scikit-Learn**
* **XGBoost**
* **MLflow**
* **Streamlit**

## 💻 Getting Started

### Prerequisites

* Python 3.8+

### Installation

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv smartenv
    # On Windows:
    .\smarteny\Scripts\activate
    # On macOS/Linux:
    source smartenv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install pandas numpy scikit-learn xgboost streamlit mlflow joblib matplotlib seaborn
    ```

3.  **Place the Dataset:**
    Download `train.csv`, `test.csv`, and `sample_submission.csv` and place them inside the `data/` directory.

### Running the Project

1.  **Activate your virtual environment** in your terminal (if not already active).

2.  **Train the Models and Generate Predictions:**
    This script will perform data preprocessing, train multiple models, log experiments with MLflow, save the best model, and generate `data/submission.csv` with predictions for the `test.csv` data.
    ```bash
    python datapreprocess.py
    python modeltraining.py
    python dataprediction.py
    ```

3.  **Explore MLflow UI:**
    Open a **new** terminal, activate your virtual environment, navigate to the project root, and run:
    ```bash
    mlflow ui
    ```
    Then, open your web browser and go to `http://localhost:5000`.

4.  **Launch the Streamlit Web Application:**
    Once the model training is complete, run the Streamlit application:
    ```bash
    streamlit run Smartpremium.py
    ```
    This will open the web app in your browser, allowing you to input customer details and get premium predictions.

## 🤝 Support & Contribution

Contributions are welcomed!!! Feel free to fork this repository, open issues, and submit pull requests.
