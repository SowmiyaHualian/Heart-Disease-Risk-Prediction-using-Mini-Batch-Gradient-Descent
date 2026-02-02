# 🫀 Heart Disease Risk Prediction using Logistic Regression + Mini-Batch Gradient Descent

## Project Description
Heart disease is one of the leading causes of death worldwide, making early risk prediction essential. Medical datasets contain multiple clinical and demographic attributes, which increase computational complexity during model training.

Traditional optimization methods such as **Batch Gradient Descent** are slow for large datasets, while **Stochastic Gradient Descent** may suffer from unstable convergence. This project develops a **heart disease risk prediction system using Logistic Regression optimized with Mini-Batch Gradient Descent (MBGD)** to achieve faster training, stable convergence, and accurate probability-based predictions that are meaningful in clinical contexts.

### Key Goals
- Produce probability-based predictions for clinical interpretation  
- Improve training speed and stability using mini-batch gradient descent  
- Keep the model simple and interpretable to support clinical adoption  

---

## Objectives
- Assist in early detection and timely medical consultation  
- Implement Logistic Regression for binary classification of heart disease risk  
- Reduce computational time and memory usage through mini-batch updates  
- Generate probability-based risk predictions  

---

## Dataset
**Recommended Dataset:**  
UCI Heart Disease Dataset (Cleveland or combined variant)

**Expected Format:**
- CSV file with clinical and demographic attributes  
- `target` column:
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

**Typical Features:**
- age  
- sex  
- chest pain type (cp)  
- resting blood pressure (trestbps)  
- cholesterol (chol)  
- fasting blood sugar (fbs)  
- resting ECG results (restecg)  
- maximum heart rate achieved (thalach)  
- exercise-induced angina (exang)  
- ST depression (oldpeak)  
- slope  
- number of major vessels (ca)  
- thalassemia (thal)  

---

## Project Structure
```text
Heart-Disease-Risk-Prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   │   └── heart_disease.csv
│   └── processed/              # Cleaned and preprocessed data
│       └── heart_disease_processed.csv
│
├── notebooks/
│   └── exploration.ipynb       # EDA and feature analysis
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Cleaning, encoding, scaling
│   ├── model.py                # Logistic Regression model
│   ├── train.py                # Training with Mini-Batch GD
│   ├── evaluate.py             # Evaluation metrics
│   └── utils.py                # Helper functions
│
├── models/
│   └── logistic_regression.pkl # Saved trained model
│
├── results/
│   ├── metrics.txt             # Accuracy and evaluation results
│   └── plots/                  # Loss curve, ROC curve
│
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files
