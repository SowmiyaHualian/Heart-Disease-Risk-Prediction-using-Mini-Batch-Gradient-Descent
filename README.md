# Heart-Disease-Risk-Prediction-using-Mini-Batch-Gradient-Descent

Project description
-------------------
Heart disease is one of the leading causes of death worldwide, making early risk prediction essential. Medical datasets contain multiple clinical and demographic factors, which increase computational complexity during model training. Traditional optimization methods such as Batch Gradient Descent are slow for large datasets, while Stochastic Gradient Descent can be unstable. This project develops a heart disease risk prediction system using Logistic Regression optimized with Mini-Batch Gradient Descent to achieve faster, stable, and accurate prediction of disease probability and clinically meaningful risk levels.

**Key goals:**
- Produce probability-based predictions for clinical interpretation.
- Improve training speed and stability by using mini-batch gradient descent.
- Keep the model simple and interpretable (logistic regression) to facilitate adoption in clinical settings.

Objectives
----------
- Assist in early detection and timely medical consultation.
- Implement Logistic Regression for binary classification of heart disease risk.
- Reduce computational time and memory usage during training via mini-batch updates.
- Generate probability-based risk predictions to improve clinical interpretability.

Dataset
-------
Recommended dataset: UCI Heart Disease Dataset (Cleveland or combined variant).
Expected format:
- CSV with clinical/demographic columns and a `target` column with values 0 (no disease) or 1 (disease).
- Typical features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.

**Preprocessing steps:**
1. Load CSV and inspect missing values.
2. Impute or drop missing values consistently.
3. Encode categorical features (one-hot or ordinal where appropriate).
4. Scale continuous features (StandardScaler or MinMaxScaler).
5. Split into train / validation / test sets (or use k-fold CV).  like this 
