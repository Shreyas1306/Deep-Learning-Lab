# Assignment 04 – LSTM-Based Time-Series Forecasting

## Aim

To develop an LSTM-based model for time-series forecasting using historical stock price data.

## Problem Statement

Develop an LSTM-based model for time-series forecasting using stock price, weather, or sales datasets.

This assignment uses historical Apple Inc. (AAPL) stock price data to predict future closing prices.

## Dataset

The assignment uses historical **Apple Inc. (AAPL)** stock market data.

The data is downloaded directly using the `yfinance` Python library for the period:

- Start Date: 2015-01-01
- End Date: 2025-01-01
- Stock: AAPL
- Target variable: Closing Price

The dataset does not need to be stored separately because it is downloaded automatically when the notebook is executed.

### Data Source

Yahoo Finance data accessed through the `yfinance` Python library.

## Methodology

The notebook follows these steps:

1. Install and import the required libraries
2. Download historical AAPL stock data
3. Inspect the dataset and missing values
4. Extract the closing price
5. Visualize the historical closing price
6. Normalize the closing prices using Min-Max scaling
7. Split the data into training and testing sets
8. Create time-series sequences using a 60-day lookback window
9. Reshape the data for LSTM input
10. Build the LSTM model
11. Compile the model using the Adam optimizer and Mean Squared Error loss
12. Train the model
13. Plot training and validation loss
14. Generate predictions on the test data
15. Convert predictions back to the original price scale
16. Evaluate the model using MSE, RMSE, and MAE
17. Compare actual and predicted stock prices
18. Predict the next day's closing price

## Model Architecture

The LSTM model consists of:

- LSTM layer with 64 units and `return_sequences=True`
- Dropout layer with dropout rate 0.2
- LSTM layer with 64 units
- Dropout layer with dropout rate 0.2
- Dense output layer with 1 neuron

### Training Configuration

- Optimizer: Adam
- Loss function: Mean Squared Error
- Epochs: 20
- Batch size: 32
- Validation split: 10%
- Time-step: 60 days

## Evaluation Metrics

The model is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

The notebook also visualizes:

- Training vs validation loss
- Actual vs predicted AAPL closing prices

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- yfinance
- Google Colab / Jupyter Notebook

## Files

```text
Assignment-04/
├── Assignment-04.docx
├── Assignment-04.ipynb
└── README.md
```
