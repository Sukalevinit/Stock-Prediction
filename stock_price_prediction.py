import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Download stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        print(f"No data found for {ticker}. Please check the symbol and date range.")
        return None
    stock_data['Date'] = stock_data.index
    return stock_data

# Step 2: Prepare the data for modeling
def prepare_data(stock_data):
    stock_data['Days'] = np.arange(len(stock_data))
    X = stock_data[['Days']].values
    y = stock_data['Close'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 4: Make predictions
def predict_and_plot(model, X_test, y_test, stock_data):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Prices', color='blue')
    plt.scatter(stock_data['Date'][X_test.flatten()], predictions, color='red', label='Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    ticker = "BTC-USD"  # Example: Bitcoin in USD
    start_date = "2020-01-01"
    end_date = "2025-01-17"

    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data is None:
        exit()  # Exit the program if no data is found

    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_model(X_train, y_train)
    predict_and_plot(model, X_test, y_test, stock_data)
