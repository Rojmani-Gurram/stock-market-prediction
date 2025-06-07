import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset (replace with your own stock CSV if available)
url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents-financials.csv"
data = pd.read_csv(url)

# Predict 'Price' from 'Market Cap'
data = data[['Market Cap', 'Price']].dropna()

X = data[['Market Cap']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, predictions, color='red', label='Predicted')
plt.xlabel('Market Cap')
plt.ylabel('Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()
