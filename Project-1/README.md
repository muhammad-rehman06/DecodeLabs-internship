# Titanic Dataset Cleaning & Exploratory Data Analysis

## Project Overview

This project focuses on **cleaning and exploring the Titanic dataset** to prepare it for future machine learning and predictive modeling tasks.

The project includes data inspection, exploratory data analysis (EDA), missing-value handling, outlier removal, feature engineering, and saving the final cleaned dataset.

---

## Dataset

The project uses the **Titanic dataset**, which contains information about passengers aboard the Titanic.

Some of the important features include:

* `Survived` – Whether the passenger survived
* `Pclass` – Passenger class
* `Sex` – Passenger gender
* `Age` – Passenger age
* `SibSp` – Number of siblings/spouses aboard
* `Parch` – Number of parents/children aboard
* `Fare` – Passenger fare
* `Cabin` – Cabin information
* `Embarked` – Port of embarkation

The original dataset contains **891 rows**.

---

## Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Loading

The Titanic dataset was loaded using Pandas:

```python
df = pd.read_csv("train.csv")
```

### 2. Data Inspection

The dataset was examined using:

* `head()`
* `tail()`
* `sample()`
* `shape`
* `columns`
* `info()`
* `describe()`
* `isnull().sum()`
* `duplicated().sum()`

This helped understand the dataset structure, data types, missing values, and duplicate records.

### 3. Exploratory Data Analysis

Several visualizations were created to understand the data, including:

* Survival count
* Gender distribution
* Passenger class distribution
* Age distribution
* Fare distribution
* Age boxplot
* Fare boxplot
* Correlation heatmap

### 4. Handling Missing Values

Missing values were handled as follows:

* Missing `Age` values were replaced with the **median age**.
* Missing `Embarked` values were replaced with the **mode**.
* The `Cabin` column was removed because of missing values.

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df.drop(columns="Cabin", inplace=True)
```

### 5. Outlier Removal

The **IQR (Interquartile Range) method** was used to identify and remove outliers from:

* `Age`
* `Fare`

The following formula was used:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

### 6. Feature Engineering

Three new features were created:

#### FamilySize

Represents the total family size of a passenger.

```python
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
```

#### IsAlone

Indicates whether a passenger was traveling alone.

```python
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

#### FarePerPerson

Calculates the fare per family member.

```python
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
```

---

## Output

After cleaning and feature engineering, the processed dataset was saved as:

```text
cleaned_titanic.csv
```

This cleaned dataset can be used as a starting point for future **machine learning and predictive modeling** tasks.

---

## Project Structure

```text
Titanic-Dataset-Cleaning/
│
├── train.csv
├── cleaned_titanic.csv
├── Titanic_EDA.ipynb
└── README.md
```

> Rename `Titanic_EDA.ipynb` to the actual name of your notebook if it is different.

---

## Conclusion

In this project, the Titanic dataset was cleaned and prepared for machine learning.

The following tasks were completed:

* Performed Exploratory Data Analysis (EDA)
* Inspected the dataset and its structure
* Identified and handled missing values
* Removed outliers using the IQR method
* Created new features:

  * `FamilySize`
  * `IsAlone`
  * `FarePerPerson`
* Saved the cleaned dataset for future machine learning tasks

The resulting dataset is cleaner and more suitable for **predictive modeling and further machine learning analysis**.
