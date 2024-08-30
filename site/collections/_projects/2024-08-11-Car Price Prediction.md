---

date: 2024-08-11 05:20:35 +0300  
title: Car Price Prediction  
subtitle: Machine Learning Project  
image: '/images/car_price_analysis.jpg'  
---
In this project, we explore the process of analyzing and predicting car prices using machine learning techniques. The project covers data cleaning, feature selection, and the implementation of regression models, with the goal of accurately predicting car prices.

<!-- ![Correlation Matrix](/images/correlation_matrix.png){: width="1200" height="900"} -->

### Data Cleaning and Preprocessing

The dataset is first cleaned by removing irrelevant columns, checking for duplicates, and preparing it for further analysis. This step ensures that the data is in the best possible shape for model training.

```python
import pandas as pd

data = pd.read_csv('CarPrice_Assignment.csv')
data.drop(['car_ID'], axis=1, inplace=True)
data.info()
```

### Feature Selection

Using correlation analysis and a Random Forest model, we identify the most influential features that affect car prices. This step helps in reducing the dimensionality of the dataset and focuses on the most relevant variables.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Preprocessing categorical features
data_encoded = data.copy()
for col in data.select_dtypes(include=['object']).columns:
    data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])

X = data_encoded.drop('price', axis=1)
y = data_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
features = X.columns
```

![Feature Importance](/images/feature_importance.png){: width="1200" height="900"}

### Regression Models: Linear and Polynomial

We implement both linear and polynomial regression models to capture the relationship between the selected features and car prices. The models are evaluated based on Mean Squared Error (MSE) and R-squared values.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Visualizing Results

The results from the regression models are visualized to compare predicted vs actual car prices, providing insights into the model's performance.

![Actual vs Predicted](/images/actual_vs_predicted_lr.png){: width="1200" height="900"}
![Actual vs Predicted](/images/actual_vs_predicted_pr.png){: width="1200" height="900"}

### Conclusion

This project demonstrates the application of feature selection and regression techniques in predicting car prices. The methods used ensure that the model is both accurate and interpretable, making it a valuable tool for real-world applications.

For the complete code and further details, you can check out the [GitHub repository](https://github.com/Youssef-KhaledMo/Car-Price-Prediction).