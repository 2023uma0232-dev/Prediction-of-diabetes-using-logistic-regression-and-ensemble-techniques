# Diabetes Prediction using Logistic Regression and Ensemble Techniques

A clean, modular machine learning project comparing custom algorithms built from scratch with standard implementations to predict diabetes on two distinct datasets: Pima Indians and Vanderbilt African-Americans.

---

## 📌 Project Overview

This repository refactors the analysis of diabetes prediction models as described by **Rajendra & Latifi (2021)** in *Computer Methods and Programs in Biomedicine Update*. 

Key highlights of this project:
- **Models from Scratch:** Logistic Regression, Decision Tree, and Support Vector Machine (SVM using CVXOPT QP solver) are implemented entirely from scratch in Python.
- **Sklearn Models:** K-Nearest Neighbors (KNN) and Gaussian Naive Bayes (GNB) are used as benchmarks.
- **Ensemble Techniques:** Custom Majority Voting and Stacking Classifier configurations evaluate model consensus and meta-learning performance.
- **Dual Datasets:** Directly compares performance on the Pima Indians Diabetes Dataset (DS1) and the Vanderbilt African-Americans Virginia Dataset (DS2).

---

## 📂 Repository Structure

```
diabetes-prediction-logistic-regression-ensemble/
│
├── data/
│   ├── pima_diabetes.csv              # Pima Indians Diabetes Dataset (DS1)
│   └── vanderbilt_diabetes.csv        # Vanderbilt African-Americans Dataset (DS2)
│
├── notebooks/
│   └── diabetes_prediction.ipynb      # Cleaned and refactored Jupyter notebook
│
├── src/
│   ├── preprocessing.py               # Preprocessing pipelines and cleaning
│   ├── feature_selection.py           # Feature engineering and flag generation
│   ├── models.py                      # Scratch model classes and custom ensembles
│   ├── evaluation.py                  # Evaluation helper functions
│   └── utils.py                       # General utilities (e.g., train-test split)
│
├── images/                            # Visualizations and EDA plots
│
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git exclusion rules
├── LICENSE                            # MIT License
└── README.md                          # Repository documentation
```

---

## 📊 Datasets & Preprocessing

### 1. Dataset 1: Pima Indians Diabetes (DS1)
- **Source:** Plotly/GitHub Repository
- **Preprocessing:** Imputes physical impossibility zero-values (in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with column-wise means.
- **Feature Engineering:** Adds 5 binary clinical flags:
  - **NF1:** Age $\le$ 30 & Glucose $\le$ 140
  - **NF2:** Age $>$ 30 & Glucose $\le$ 140
  - **NF3:** Age $\le$ 30 & Glucose $>$ 140
  - **NF4:** Age $>$ 30 & Glucose $>$ 140
  - **NF5:** BMI $>$ 30 & Age $>$ 30

### 2. Dataset 2: Vanderbilt African-Americans (DS2)
- **Source:** Tartu University Repository
- **Preprocessing:** Extracts BMI and Waist-to-Hip ratio. Maps target label based on clinical cutoff `glyhb >= 7.0`. Handles missing values and zeros similarly to DS1.

---

## 🛠️ Models & Ensembles

### Custom Implementations (From Scratch)
1. **Logistic Regression (`LogReg`):** Gradient descent optimization with Standard Scaling.
2. **Decision Tree (`DecisionTree`):** Entropy & Information Gain split metric.
3. **SVM (`SVM`):** Uses the `cvxopt` Quadratic Programming solver to solve the dual optimization problem. Supports Linear, Polynomial, and RBF kernels.

### Ensemble Configurations
- **Majority Voting:** Collects class predictions across base models and selects the majority class. Evaluated with 10-fold cross-validation.
- **Stacking Classifier:** Trains base models (Decision Tree, Naive Bayes, KNN) and generates out-of-fold (OOF) predictions. A Logistic Regression meta-learner is then trained on these predictions.

---

## 🚀 Getting Started

### Prerequisites
Make sure Python 3.8+ is installed on your system.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction-logistic-regression-ensemble.git
   cd diabetes-prediction-logistic-regression-ensemble
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook
To run the full evaluation pipeline and regenerate the results and plots:
```bash
jupyter notebook notebooks/diabetes_prediction.ipynb
```
or convert and run via command line:
```bash
jupyter nbconvert --to notebook --execute notebooks/diabetes_prediction.ipynb --inplace
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
