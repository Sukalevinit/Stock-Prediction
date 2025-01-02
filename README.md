Description
This Python script uses the yfinance library to fetch historical stock data and applies linear regression to predict future stock prices. It performs the following steps:

Stock Data Retrieval: Downloads historical stock data for a given ticker symbol (e.g., Google - "GOOG") over a specified date range using the yfinance library.
Data Preparation: Converts the stock data into a format suitable for training a machine learning model by creating a new feature Days, representing the number of days from the start date.
Model Training: Trains a Linear Regression model on the prepared data.
Prediction & Plotting: Predicts stock prices for the test dataset and plots both the actual and predicted stock prices, displaying the results in a line chart.
The script uses sklearn for model training and evaluation, matplotlib for visualization, and numpy and pandas for data manipulation.

Features:
Downloads stock data using yfinance.
Trains a simple Linear Regression model to predict future stock prices.
Visualizes actual vs predicted stock prices.
Evaluates the model using Mean Squared Error (MSE).

Stock Price Prediction using Linear Regression:

This project predicts future stock prices using historical stock data. It applies Linear Regression to forecast the stock prices and evaluates the model's performance with Mean Squared Error (MSE). A visualization of actual vs predicted stock prices is also generated.

Requirements:

To run this project, make sure to install the required Python libraries. You can install them via pip:
pip install yfinance pandas numpy scikit-learn matplotlib

Files:
stock_prediction.py: Python script for downloading stock data, preparing the data, training the model, and making predictions.

Usage
Download the Stock Data:

The script retrieves historical stock data for a given stock ticker (e.g., "GOOG" for Google) from Yahoo Finance.
You can modify the ticker, start_date, and end_date in the script.

Run the Script:

Simply execute the script
python stock_prediction.py

Results:

The script will display a plot with actual stock prices in blue and predicted prices in red.
The Mean Squared Error (MSE) of the model will be printed in the terminal.

Notes
The script uses a simple Linear Regression model, which is not ideal for stock price predictions, as stock prices are typically influenced by many complex factors.
The accuracy of the model may not be high, but this script demonstrates a basic use case of machine learning for stock price prediction.

License
This project is open-source and available under the MIT License.
