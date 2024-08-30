---
title: Enhancing Predictive Models with Feature Selection Techniques
date:   2024-08-06
image:  '/images/feature-selection.jpg'
tags:   [machine learning, feature selection, data science]
---

In machine learning, selecting the right features can significantly impact the performance of your model. This blog post walks through various feature selection techniques, using the Titanic dataset as an example. We'll explore methods like Feature Importance with Random Forest, Permutation Importance, and Recursive Feature Elimination (RFE), followed by training models like Decision Tree, Random Forest, and AdaBoost.

<!-- ![Feature Selection](/images/feature-selection.jpg){: width="1200" height="900"} -->

### Feature Selection Techniques

#### **1. Feature Importance with Random Forest**

One of the simplest ways to determine the significance of features is by using a Random Forest model. This model allows us to evaluate the importance of each feature based on how much they improve the model's accuracy.

```python
# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.show()
```

#### **2. Permutation Importance**

Permutation Importance measures the importance of a feature by observing how random shuffling of the feature impacts model accuracy. This method is model-agnostic and offers more nuanced insights.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.barh(X.columns[sorted_idx], result.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title('Permutation Importance of Features')
plt.show()
```

#### **3. Recursive Feature Elimination (RFE)**

RFE is a technique that recursively removes the least important features and builds a model using the remaining attributes. The goal is to select a subset of features that contribute most to the target variable.

```python
from sklearn.feature_selection import RFE

rfe = RFE(model, n_features_to_select=5)
rfe.fit(X, y)

feature_ranking = pd.Series(rfe.ranking_, index=X.columns)
feature_ranking.plot(kind='barh')
plt.title('Feature Ranking by RFE')
plt.show()
```

### Correlation Analysis

Understanding the correlation between features and the target variable is crucial in feature selection. We'll visualize the correlation matrix to identify the most impactful features.

```python
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Modeling

Once the key features are selected, it's time to train our models. Here’s how we approach it:

#### **1. Decision Tree**

```python
from sklearn.tree import DecisionTreeClassifier

# Train a Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Evaluate the model
y_pred = tree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
```

#### **2. Random Forest**

```python
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

# Evaluate the model
y_pred = forest.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
```

#### **3. AdaBoost**

AdaBoost combines weak learners to create a strong classifier. We’ll use GridSearchCV to find the best parameters for our AdaBoost model.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)
```

### Conclusion

Feature selection is a critical step in building efficient and accurate machine learning models. By using techniques like Feature Importance, Permutation Importance, and RFE, we can refine our models and improve their predictive power. The Titanic dataset offers a great starting point to explore these methods, and the same concepts can be applied to more complex datasets.

For the complete code and detailed explanations, visit the [GitHub repository](https://github.com/Youssef-KhaledMo/Feature-Selection).