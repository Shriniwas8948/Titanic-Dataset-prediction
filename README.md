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
```

Load data in your notebook or script:

```python
train = pd.read_csv('/content/titanic/train.csv')
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

**Logistic Regression Accuracy:**

```
0.793296089
```

---

### Random Forest Classifier

**Random Forest Accuracy:**

```
0.82122905
```

---

## 📊 Evaluation

Both models were evaluated using:

* Accuracy Score

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
 ├── Titanic-Dataset-prediction.ipynb
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
