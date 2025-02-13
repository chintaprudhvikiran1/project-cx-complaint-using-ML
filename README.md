# Customer Complaint Prediction

## Overview

This repository contains an end-to-end machine learning pipeline to predict customer complaints using various machine learning models. The dataset undergoes extensive exploratory data analysis (EDA), feature engineering, and model evaluation to determine the best-suited algorithm for predictive analysis.

## Project Workflow

## 1. Exploratory Data Analysis (EDA)

•	Dataset Information: Checked data types, missing values, and summary statistics using .info() and .describe().

•	Handling Missing Values: Identified and imputed missing values.

•	Target Variable Distribution: Analyzed the distribution of the target variable.

•	Correlation Matrix: Visualized feature relationships using a heatmap.

•	Outlier Detection: Used box plots to identify and handle outliers.

•	Feature-wise Analysis: Studied individual features to assess their importance and impact.

## 2. Data Preprocessing & Feature Engineering
•	Feature Engineering: Created new features and transformed existing ones.

•	Feature Scaling: Normalized continuous features using StandardScaler from sklearn.preprocessing.

•	Feature Selection: Applied SelectKBest to select the most relevant features.

•	Train-Test Split: Split the dataset into training and testing sets.

•	Handling Class Imbalance: Used SMOTE to address imbalance in the target variable.

## 3. Model Training & Evaluation
### Imported Libraries

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.feature_selection import SelectKBest, chi2, f_classif


## Trained Machine Learning Models
•	Logistic Regression

•	Support Vector Machine (SVM)

•	Random Forest

•	Gradient Boosting

•	XGBoost

## Cross-Validation & Model Evaluation
•	Used Cross-Validation to assess model performance.

•	Evaluated models based on:

 o	Accuracy
 
 o	Precision
 
 o	Recall
 
 o	F1-score
 
 o	Confusion Matrix
 
## 4. Best Model Selection
After training and evaluating all models, the best model was selected based on performance metrics and cross-validation scores.

### Results
•	The model with the highest accuracy and balanced precision-recall was selected for final deployment.
