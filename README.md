# 🤖 Automated Model Selection System

## 🚀 Overview

This project implements an **Automated Machine Learning (AutoML) system** that trains multiple models, tunes them using cross-validation, and automatically selects the best-performing model.

Unlike basic ML workflows that rely on a single model, this system builds a **robust, scalable pipeline** for model comparison and selection.

---

## 🧠 Key Features

* 🔄 **Feature Engineering Pipeline**

  * Handles missing values
  * Encodes categorical variables
  * Scales numerical features
  * Ensures consistent preprocessing

* 🤖 **Multi-Model Training**

  * Linear Regression
  * Decision Tree
  * Random Forest

* 🎛️ **Hyperparameter Tuning**

  * Implemented using GridSearchCV
  * Automatically finds best model configurations

* ⚖️ **Cross-Validation**

  * Ensures stable and reliable model evaluation
  * Reduces overfitting risk

* 🏆 **Automated Model Selection**

  * Compares models based on MAE
  * Selects the best-performing model

* 💾 **Model & Pipeline Persistence**

  * Saves trained model (`best_model.pkl`)
  * Saves preprocessing pipeline (`preprocess_pipeline.pkl`)

---

## 🏗️ Project Structure

```id="str01"
model-selection-system/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── best_model.pkl
│
├── pipelines/
│   └── preprocess_pipeline.pkl
│
├── src/
│   ├── data_loader.py
│   ├── pipeline_builder.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── selector.py
│
├── config/
│   └── config.yaml
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ System Workflow

```id="flow01"
Raw Data
   ↓
Preprocessing Pipeline
   ↓
Train Multiple Models
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Cross-Validation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Save Model + Pipeline
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Pickle

---

## ▶️ How to Run

### 1. Clone the Repository

```id="run01"
git clone <your-repo-link>
cd model-selection-system
```

### 2. Install Dependencies

```id="run02"
pip install -r requirements.txt
```

### 3. Run the System

```id="run03"
python main.py
```

---

## 📊 Example Output

```id="out01"
Linear Regression trained with best params: {}
Decision Tree trained with best params: {...}
Random Forest trained with best params: {...}

Model Performance:
Linear Regression → {'mae': 8750.0}
Decision Tree → {'mae': 7750.0}
Random Forest → {'mae': 8900.0}

🏆 Best Model: Decision Tree
```

---

## 💡 Key Learnings

* Importance of **feature engineering pipelines**
* Preventing **data leakage**
* Using **cross-validation for robust evaluation**
* Automating **model selection and tuning**
* Designing **modular ML systems**

---

## 🔮 Future Improvements

* Add more models (XGBoost, LightGBM)
* Implement RandomizedSearchCV / Bayesian Optimization
* Integrate experiment tracking system
* Build dashboard for model comparison
* Deploy as API using FastAPI

