# Titanic Survival Prediction

This repository contains a complete machine learning pipeline to predict passenger survival on the **Titanic dataset**. The project includes data loading, cleaning, preprocessing, feature engineering, model training, and evaluation using Logistic Regression and Random Forest Classifier.

---

## 📦 Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

Install all required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📥 Dataset

This project uses the **Kaggle Titanic: Machine Learning from Disaster** dataset.

Download from Kaggle and place the files here:

```
/data/train.csv
/data/test.csv
```

Load data in your notebook or script:

```python
train = pd.read_csv('/content/titanic/train.csv')
test = pd.read_csv('/content/titanic/test.csv')
```

---

## 🧹 Data Preprocessing

The pipeline handles:

* Missing values (Age, Embarked, Fare)
* Scaling numeric features
* Encoding categorical features
* Feature selection for model training

Features used:

* `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

Example preprocessing setup:

```python
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
```

---

## 🧠 Model Training

Two models were trained:

* **Logistic Regression**
* **Random Forest Classifier**

Train-test split:

```python
X = train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Logistic Regression Model

```python
log_model = Pipeline(steps=[('preprocess', preprocess),
                           ('clf', LogisticRegression(max_iter=200))])

log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)
```

**Logistic Regression Accuracy:**

```
0.793296089
```

---

### Random Forest Classifier

```python
rf_model = Pipeline(steps=[('preprocess', preprocess),
                          ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
```

**Random Forest Accuracy:**

```
0.82122905
```

---

## 📊 Evaluation

Both models were evaluated using:

* Accuracy
* Confusion Matrix
* Classification Report

Example:

```python
print(confusion_matrix(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
```

---

## 📁 Project Structure

```
📂 Titanic-Prediction
 ├── data/
 │    ├── train.csv
 │    └── test.csv
 ├── src/
 │    └── model.py (optional script)
 ├── notebook.ipynb
 └── README.md
```

---

## ✔ Results Summary

| Model                    | Accuracy        |
| ------------------------ | --------------- |
| Logistic Regression      | **0.793296089** |
| Random Forest Classifier | **0.82122905**  |

Random Forest performs better due to its ability to capture complex interactions and nonlinear patterns.

---

If you'd like, I can also generate:

* a full Jupyter Notebook
* a `train.py` script
* a model deployment example (Flask/Streamlit)

Let me know!
