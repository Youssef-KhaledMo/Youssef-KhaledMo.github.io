---

title: Regression Techniques for Car Price Prediction  
date: 2024-08-11  
image: '/images/car_price_analysis.jpg'  
tags: [machine learning, data science, regression]  
---
In this post, we'll dive into the process of analyzing and predicting car prices using machine learning techniques. We'll explore how to clean and preprocess data, select the most relevant features, and apply both linear and polynomial regression models. 

### Data Loading and Initial Exploration

We begin by loading the car price dataset and taking a quick look at the first few rows to understand the structure of the data.

```python
import pandas as pd # type: ignore

data = pd.read_csv('CarPrice_Assignment.csv')
data.head()
```

### Data Cleaning

To prepare the data for analysis, we drop irrelevant columns (such as `car_ID`) and check the dataset's structure to ensure everything is in order.

```python
data.drop(['car_ID'], axis=1, inplace=True)
data.info()
```

We also check for duplicate rows, which could distort our model's accuracy.

```python
data.duplicated().sum()
```

### Correlation Analysis

Next, we analyze the correlation between numeric features and the target variable, `price`. This helps in identifying the most influential features for car price prediction.

```python
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Select numeric features
numeric_features = data.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
correlation_matrix = numeric_features.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Save the 15 most correlated features with price
top_features_correlation = correlation_matrix['price'].abs().sort_values(ascending=False)[:15]
```

### Feature Selection with Random Forest

To further refine our feature selection, we use a Random Forest model to evaluate the importance of each feature.

```python
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

# Preprocessing categorical features
data_encoded = data.copy()
for col in data.select_dtypes(include=['object']).columns:
    data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

# Separate features and target
X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest')
plt.show()

# Save the 15 most important features
top_features_importance = feature_importance_df['Feature'].head(15)
```

### Regression Models: Linear and Polynomial

With our selected features, we move on to building and evaluating regression models. First, we implement a linear regression model:

```python
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

# Plot the predicted vs actual values
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red')  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Linear Regression)')
plt.show()
```

To capture potential non-linear relationships, we also fit a polynomial regression model:

```python
from sklearn.preprocessing import PolynomialFeatures # type: ignore

# Create polynomial features
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions using the polynomial model
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate the mean squared error for polynomial regression
mse_poly = mean_squared_error(y_test, y_pred_poly)

# Calculate the R-squared value for polynomial regression
r2_poly = r2_score(y_test, y_pred_poly)

# Plot the predicted vs actual values for polynomial regression
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_poly, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_pred_poly), max(y_pred_poly)], color='red')  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted (Polynomial Regression)')
plt.show()
```

### Conclusion

Through this analysis, we demonstrated how to effectively select features and apply regression models to predict car prices. The use of both linear and polynomial regression models allows for a comprehensive evaluation of the data, ensuring that the final model captures the underlying patterns accurately.

For the complete code and further details, you can check out the [GitHub repository](https://github.com/Youssef-KhaledMo/Car-Price-Prediction).
