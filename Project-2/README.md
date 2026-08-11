# Fraud Detection Pipeline

## Project Overview

This project implements a **Supervised Learning Fraud Detection Pipeline** using machine learning to identify fraudulent transactions.

The project focuses on handling transaction data, preprocessing it, applying **SMOTE (Synthetic Minority Over-sampling Technique)**, training classification models, and evaluating their performance using **Precision, Recall, and ROC-AUC** rather than relying primarily on Accuracy.

## Dataset

The project uses the `creditcard_2023.csv` dataset.

* **Total transactions:** 568,630
* **Features after preprocessing:** 29
* **Target variable:** `Class`
* `Class = 0`: Legitimate transaction
* `Class = 1`: Fraudulent transaction

The `id` column was removed because it is an identifier rather than a meaningful predictive feature.

## Project Workflow

The project follows these steps:

1. Import required libraries
2. Load and inspect the dataset
3. Perform exploratory data analysis
4. Check missing values and duplicate records
5. Analyze the class distribution
6. Analyze feature correlations
7. Remove the `id` column
8. Split the data into training and testing sets
9. Apply SMOTE to the training data
10. Train Logistic Regression
11. Train Random Forest
12. Evaluate models using:

* Precision
* Recall
* ROC-AUC

13. Perform Random Forest hyperparameter tuning
14. Evaluate the tuned model
15. Compare model performance
16. Draw conclusions

## Handling Class Imbalance

SMOTE was implemented as required by the project.

However, the dataset was already perfectly balanced:

```text
Class 0: 227,452
Class 1: 227,452
```

Therefore, applying SMOTE did not change the class distribution.

SMOTE was applied **only to the training data**, while the test data remained untouched for unbiased evaluation.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

The features were standardized using `StandardScaler` before training.

### 2. Random Forest

Random Forest was trained as the main tree-based classification model.

The initial model used:

* `n_estimators = 50`
* `max_depth = 15`
* `random_state = 42`
* `n_jobs = -1`

## Model Evaluation

Accuracy was not used as the primary evaluation metric. Instead, the models were evaluated using Precision, Recall, and ROC-AUC.

### Results

| Model               |   Precision |      Recall |     ROC-AUC |
| ------------------- | ----------: | ----------: | ----------: |
| Logistic Regression |     0.97717 |     0.95220 |     0.99350 |
| Random Forest       | **0.99967** | **1.00000** | **0.99999** |
| Tuned Random Forest |     0.99958 | **1.00000** |     0.99997 |

## Hyperparameter Tuning

Randomized hyperparameter search was performed on the Random Forest model.

The best parameters found were:

```text
n_estimators = 30
max_depth = 20
min_samples_split = 5
```

The best cross-validation ROC-AUC was:

```text
0.9999718
```

The tuned Random Forest achieved the following results on the test set:

* **Precision:** 0.99958
* **Recall:** 1.00000
* **ROC-AUC:** 0.99997

## Conclusion

Random Forest performed better than Logistic Regression for this fraud detection task.

The original Random Forest achieved a Precision of **0.99967**, Recall of **1.00000**, and ROC-AUC of **0.99999**, making it the best-performing model among the evaluated models.

Hyperparameter tuning produced a slightly lower test performance than the original Random Forest. This indicates that the initial Random Forest configuration was already highly effective for this dataset.

Overall, the project demonstrates a complete supervised learning pipeline for fraud detection, including data preprocessing, class-imbalance handling, model training, evaluation, and hyperparameter tuning.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

## Project Structure

```text
Fraud-Detection-Pipeline/
│
├── creditcard_2023.csv
├── fraud_detection.ipynb
└── README.md
```

## Requirements

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## How to Run

1. Clone or download the project.
2. Place `creditcard_2023.csv` in the project directory.
3. Open `fraud_detection.ipynb` using Jupyter Notebook or JupyterLab.
4. Run the notebook cells sequentially.
5. Review the model evaluation results, visualizations, and final comparison.

