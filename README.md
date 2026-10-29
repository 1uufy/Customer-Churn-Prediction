# Customer Churn Prediction & Analytics Pipeline

## Overview

An end-to-end machine learning project that analyzes customer data, identifies patterns associated with customer churn, and predicts whether a customer is likely to leave a service provider.

The project uses exploratory data analysis, data preprocessing, classification models, model evaluation, and feature importance analysis to derive actionable customer-retention insights.

## Objectives

- Analyze customer characteristics and churn patterns.
- Identify important predictors of customer churn.
- Train and compare multiple classification models.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Save a trained machine learning pipeline for reuse.

## Dataset

**IBM Telco Customer Churn dataset**

The dataset contains customer demographics, account information, subscribed services, contract details, and churn status.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Joblib

## Machine Learning Models

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Hyperparameter-tuned Random Forest

## Model Evaluation

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8038 | 0.6485 | 0.5722 | 0.6080 | 0.8359 |
| Random Forest | 0.7427 | 0.5106 | 0.7754 | 0.6157 | 0.8326 |
| XGBoost | 0.7832 | 0.6075 | 0.5214 | 0.5612 | 0.8301 |
| Tuned Random Forest | 0.7370 | 0.5034 | 0.7968 | 0.6170 | 0.8348 |

The tuned Random Forest achieved the highest recall (79.68%) and F1-score (0.6170) among the evaluated models. Logistic Regression achieved the highest accuracy (80.38%) and ROC-AUC (0.8359).

## Key Findings

- Month-to-month contracts were an important churn predictor.
- Customer tenure and charges were important features in the Random Forest model.
- Internet service type, online security, and technical support were important predictors in XGBoost.
- Model selection involves balancing churn detection against false alarms.

Feature importance indicates predictive usefulness and does not establish causation.

## Project Structure

```text
Customer-Churn-Prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── visualizations/
├── models/
│   └── customer_churn_model.pkl
├── src/
├── requirements.txt
└── README.md
```

## How to Run

1. Clone or download the repository.
2. Install the dependencies listed in `requirements.txt`.
3. Open `notebooks/customer_churn_analysis.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells in order.
5. Load the saved model from the `models/` directory to make predictions.

## Limitations

- Model performance depends on the dataset and its representativeness.
- Predictions are probabilistic and may be incorrect.
- Feature importance does not prove that a feature causes churn.
- Further validation and monitoring would be needed before real-world deployment.

## Future Improvements

- Perform cross-validated hyperparameter optimization.
- Tune the classification threshold for business needs.
- Add a user interface for entering customer details.
- Monitor model performance and data drift.

## Author

Ammar Hasan
