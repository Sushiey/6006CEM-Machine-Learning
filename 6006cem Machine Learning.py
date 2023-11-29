# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = '/Users/sushiey/Desktop/traffic.csv'
df = pd.read_csv(file_path, parse_dates=['DateTime'])

# Display basic information about the dataset
df.info()

# Display the first few rows of the dataset
df.head()

# Plotting the traffic flow over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.lineplot(x='DateTime', y='Vehicles', data=df, label='Traffic Flow')
plt.title('Traffic Flow Over Time')
plt.xlabel('Date and Time')
plt.ylabel('Number of Vehicles')
plt.legend()

# Visualizing the distribution of Vehicles at each Junction
plt.subplot(2, 2, 2)
sns.boxplot(x='Junction', y='Vehicles', data=df)
plt.title('Distribution of Vehicles at Each Junction')
plt.xlabel('Junction')
plt.ylabel('Number of Vehicles')

# Data Pre-processing
# Performed data pre-processing steps (e.g., DateTime parsing, handling missing data, feature engineering)

# Feature selection
X = df[['DateTime', 'Junction']]  # Features
y = df['Vehicles']  # Target variable

# Convert DateTime to timestamp
X['Timestamp'] = df['DateTime'].astype(int) // 10**9  # Convert nanoseconds to seconds

# Drop the original DateTime column
X = X.drop('DateTime', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# ARIMA Model (example, actual parameters may vary)
arima_model = ARIMA(y_train, order=(1, 1, 1))
arima_fit = arima_model.fit()

# Model Evaluation
# Linear Regression
linear_predictions = linear_model.predict(X_test)
linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

print("Linear Regression Metrics:")
print(f"MAE: {linear_mae}")
print(f"MSE: {linear_mse}")
print(f"R2 Score: {linear_r2}")
print()

# ARIMA
arima_predictions = arima_fit.forecast(steps=len(X_test))
arima_mae = mean_absolute_error(y_test, arima_predictions)
arima_mse = mean_squared_error(y_test, arima_predictions)
arima_r2 = r2_score(y_test, arima_predictions)

print("ARIMA Metrics:")
print(f"MAE: {arima_mae}")
print(f"MSE: {arima_mse}")
print(f"R2 Score: {arima_r2}")
print()

# Create a DataFrame with actual and predicted values
results_df = pd.DataFrame({
    'Actual Traffic Flow': y_test,
    'Linear Regression Predictions': linear_predictions,
    'ARIMA Predictions': arima_predictions
}, index=y_test.index)  # Use the index of y_test

# Display the table
print(results_df)

# Visualizations
# Scatter plot for Linear Regression
plt.subplot(2, 2, 3)
plt.scatter(y_test, linear_predictions)
plt.xlabel("Actual Traffic Flow")
plt.ylabel("Predicted Traffic Flow")
plt.title("Actual vs. Predicted Traffic Flow (Linear Regression)")

# Scatter plot for ARIMA
plt.subplot(2, 2, 4)
plt.scatter(y_test, arima_predictions)
plt.xlabel("Actual Traffic Flow")
plt.ylabel("Predicted Traffic Flow")
plt.title("Actual vs. Predicted Traffic Flow (ARIMA)")

plt.tight_layout()
plt.show()
