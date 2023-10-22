import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the TESLA stock price data
data = pd.read_csv('TESLA.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Sort the data by date
data = data.sort_index()

# Split the data into training and testing sets
train_data = data['Close'][:int(0.8*len(data))]
test_data = data['Close'][int(0.8*len(data)):]

# Fit the ARIMA model
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit(disp=0)

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))[0]

# Calculate the Mean Squared Error to evaluate the model
mse = mean_squared_error(test_data, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs. predicted prices
plt.plot(data['Close'], label='Actual')
plt.plot(test_data.index, predictions, color='red', label='Predicted')
plt.legend()
plt.show()
