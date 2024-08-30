---
date: 2024-08-09 10:45:00 +0300  
title: Titanic Survivors Prediction
subtitle: Machine Learning Project  
image: '/images/titanic_project.jpg'  
---
This project focuses on predicting Titanic survivors by applying various feature selection techniques and evaluating the performance of different machine learning models. We employ methods such as Random Forest Feature Importance, Permutation Importance, Recursive Feature Elimination (RFE), and Statistical Tests to select the most significant features. After feature selection, we build and evaluate models using K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) to predict the survival of passengers.

<!-- ![Titanic](/images/titanic_overview.jpg){: width="1200" height="900"} -->

### Dataset Overview

We begin by loading the Titanic dataset, which includes the training set, test set, and the actual survival outcomes for the test set.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_target = pd.read_csv('gender_submission.csv')
```

### Data Exploration

Before diving into feature selection, we explore the dataset's structure and clean it to prepare for further analysis.

```python
train.info()
test.info()
test_target.info()
```

### Feature Selection: Identifying Important Features

#### Feature Importance with Random Forest

We use a Random Forest model to determine the importance of various features in predicting the survival of passengers.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Preprocessing: Fill missing values and encode categorical features
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
train.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
train['Sex'] = LabelEncoder().fit_transform(train['Sex'])
train['Embarked'] = LabelEncoder().fit_transform(train['Embarked'])

# Separating features and target
X = train.drop(['PassengerId', 'Survived'], axis=1)
y = train['Survived']

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
```

#### Permutation Importance

We calculate permutation importance to see how shuffling each feature impacts the model's performance.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title('Permutation Importance of Features')
plt.show()
```

#### Recursive Feature Elimination (RFE)

RFE is applied to select the top 5 most important features by iteratively eliminating the least significant features.

```python
from sklearn.feature_selection import RFE

rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

# Get the ranking of the features
ranking = rfe.ranking_

# Plot the ranking of the features
feature_ranking = pd.Series(ranking, index=X.columns).sort_values()
feature_ranking.plot(kind='barh')
plt.title('Feature Ranking')
plt.show()
```

#### Statistical Tests (Chi-Squared Test and ANOVA)

For categorical features, we perform a Chi-Squared test, and for continuous features, we perform an ANOVA test to determine their significance.

```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# Chi-Squared Test for categorical features
X_cat = train[['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']]
chi2_selector = SelectKBest(chi2, k='all')
chi2_selector.fit(X_cat, y)

chi2_scores = pd.Series(chi2_selector.scores_, index=X_cat.columns)
chi2_scores.plot(kind='barh')
plt.title('Chi-Squared Scores')
plt.show()

# ANOVA Test for continuous features
X_cont = train[['Age', 'Fare']]
anova_selector = SelectKBest(f_classif, k='all')
anova_selector.fit(X_cont, y)

anova_scores = pd.Series(anova_selector.scores_, index=X_cont.columns)
anova_scores.plot(kind='barh')
plt.title('ANOVA F-Value Scores')
plt.show()
```

#### Correlation Analysis

We calculate and visualize the correlation matrix for numeric features to see their relationship with the target variable, `Survived`.

```python
corr_matrix = train.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Model Training and Evaluation

After selecting the most relevant features, we proceed to train and evaluate machine learning models.

#### Preprocessing

We fill missing values, encode categorical features, and standardize the features for modeling.

```python
from sklearn.preprocessing import StandardScaler

# Fill missing values and encode categorical features for test set
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
test['Sex'] = LabelEncoder().fit_transform(test['Sex'])
test['Embarked'] = LabelEncoder().fit_transform(test['Embarked'])

# Separating features and target
X_train = train[['Sex', 'Pclass', 'Embarked', 'Fare', 'Age']]
y_train = train['Survived']
X_test = test[['Sex', 'Pclass', 'Embarked', 'Fare', 'Age']]
y_test = test_target['Survived']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
```

#### Train & Test KNN Model

We use GridSearchCV to find the best hyperparameters for the KNN model and evaluate its performance.

```python
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Initialize the KNN model
knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Use the best estimator to make predictions
best_knn = grid_search.best_estimator_
y_pred_best_knn = best_knn.predict(X_test_scaled)

# Evaluate the best model
print("Best KNN Accuracy:", accuracy_score(y_test, y_pred_best_knn))
print("Best KNN Classification Report:\n", classification_report(y_test, y_pred_best_knn))

# Plot the confusion matrix
classes = ['Died', 'Survived']
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best_knn, display_labels=classes, cmap=plt.cm.Blues)
plt.show()
```

#### Train & Test SVM Model

We also train and evaluate a Support Vector Machine (SVM) model to compare its performance against KNN.

```python
from sklearn.svm import SVC

# Train SVM model
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test_scaled)

# Evaluate the model
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))

classes = ['Died', 'Survived']
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, display_labels=classes, cmap=plt.cm.Blues)
plt.show()
```

### Conclusion

This project demonstrates the importance of feature selection in building effective machine learning models. By applying different techniques, we identified key features that significantly impact survival predictions on the Titanic dataset. Both KNN and SVM models were trained, and their performance was evaluated, providing insights into model selection and hyperparameter tuning.

For the complete code and further details, you can visit the [GitHub repository](https://github.com/Youssef-KhaledMo/Titanic-Survivors-Prediction).
