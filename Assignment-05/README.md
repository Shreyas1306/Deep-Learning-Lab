# Assignment 05 — RNN vs LSTM vs GRU for Sequence Classification

## Aim

To implement and compare Simple RNN, LSTM, and GRU models for sequence classification and analyze their performance using appropriate evaluation metrics.

## Problem Statement

Implement and compare RNN, LSTM, and GRU models for sequence classification, and analyze their performance using appropriate evaluation metrics.

## Dataset

The experiment uses historical **Apple Inc. (AAPL)** stock-price data obtained from Yahoo Finance.

- **Period:** 2015-01-01 to 2025-01-01
- **Feature used:** Daily Closing Price
- **Task:** Predict whether the next day's closing price will increase or decrease
- **Class 0:** DOWN — next day's closing price is lower than or equal to today's
- **Class 1:** UP — next day's closing price is higher than today's

The dataset contains **2,515 observations** after removing the final record for which a next-day target is unavailable.

Class distribution:

| Class | Samples |
|---|---:|
| DOWN (0) | 1,179 |
| UP (1) | 1,336 |

## Methodology

1. Import the required libraries for data handling, preprocessing, visualization, model development, and evaluation.
2. Download AAPL historical stock-price data using `yfinance`.
3. Extract the daily closing price.
4. Create a binary target based on the following day's price movement.
5. Split the data chronologically into 80% training and 20% testing data.
6. Fit a Min-Max scaler only on the training data and use it to transform both training and testing data.
7. Use the previous **30 trading days** as the input sequence.
8. Reshape the data into `(samples, time steps, features)` format.
9. Build Simple RNN, LSTM, and GRU models with 64 recurrent units.
10. Add Dropout regularization and a sigmoid output layer.
11. Train all three models using Adam optimizer and binary cross-entropy loss.
12. Use class weights to account for the difference between the two classes.
13. Generate probability predictions and convert them to binary classes using a threshold of 0.5.
14. Evaluate the models using Accuracy, Precision, Recall, and F1-score.
15. Generate confusion matrices and compare model performance.
16. Select the best-performing model based on F1-score.
17. Use the latest 30 observations to demonstrate next-day direction prediction.

## Data Preprocessing

### Train-Test Split

The time-series data is divided chronologically:

- **Training observations:** 2,012
- **Testing observations:** 503
- **Training ratio:** 80%
- **Testing ratio:** 20%

A chronological split is used to avoid using future observations for training.

### Normalization

`MinMaxScaler` is fitted only on the training data to prevent data leakage. The fitted scaler is then used to transform both training and testing data.

### Sequence Creation

A sliding window of **30 previous trading days** is used as input to predict the corresponding stock-price direction.

Final input shapes:

- `X_train`: `(1982, 30, 1)`
- `X_test`: `(503, 30, 1)`

## Model Architecture

All three models use the same overall structure so that their performance can be compared fairly.

### Simple RNN

- Input: 30 time steps × 1 feature
- SimpleRNN: 64 units
- Dropout: 0.2
- Dense: 1 neuron
- Activation: Sigmoid

### LSTM

- Input: 30 time steps × 1 feature
- LSTM: 64 units
- Dropout: 0.2
- Dense: 1 neuron
- Activation: Sigmoid

### GRU

- Input: 30 time steps × 1 feature
- GRU: 64 units
- Dropout: 0.2
- Dense: 1 neuron
- Activation: Sigmoid

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Epochs | 30 |
| Batch Size | 32 |
| Validation Split | 10% |
| Dropout | 0.2 |
| Recurrent Units | 64 |
| Classification Threshold | 0.5 |
| Early Stopping | Yes |
| Early Stopping Patience | 5 |

Class weights were calculated from the training data:

- Class 0: approximately **1.0432**
- Class 1: approximately **0.9603**

## Evaluation Metrics

The models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted UP observations that were actually UP.
- **Recall:** Proportion of actual UP observations correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for DOWN and UP classes.

## Results

The obtained test-set results are:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| SimpleRNN | 0.4374 | 0.4000 | 0.0071 | 0.0139 |
| LSTM | 0.4573 | 0.6667 | 0.0638 | 0.1165 |
| GRU | **0.4871** | **0.5645** | **0.3723** | **0.4487** |

### Best Model

The **GRU** achieved the highest F1-score:

**F1 Score = 0.4487**

Therefore, GRU is the best-performing model among the three models for this experiment based on the selected F1-score criterion.

## Confusion Matrix Results

### SimpleRNN

| | Predicted DOWN | Predicted UP |
|---|---:|---:|
| Actual DOWN | 218 | 3 |
| Actual UP | 280 | 2 |

The SimpleRNN identified most DOWN cases but detected very few UP cases.

### LSTM

| | Predicted DOWN | Predicted UP |
|---|---:|---:|
| Actual DOWN | 212 | 9 |
| Actual UP | 264 | 18 |

The LSTM achieved higher precision for UP predictions than SimpleRNN, but its recall remained low.

### GRU

| | Predicted DOWN | Predicted UP |
|---|---:|---:|
| Actual DOWN | 140 | 81 |
| Actual UP | 177 | 105 |

The GRU produced a more balanced distribution of predictions and achieved substantially better recall and F1-score for the UP class.

## Next-Day Direction Prediction

Using the most recent 30 observations, the notebook produced the following predictions:

| Model | Direction | Probability of UP |
|---|---|---:|
| SimpleRNN | DOWN | 0.4778 |
| LSTM | DOWN | 0.4889 |
| GRU | DOWN | 0.4943 |

All three models predicted **DOWN** for the demonstrated next-day direction, with probabilities of UP remaining below the 0.5 classification threshold.

## Observations

- SimpleRNN obtained an accuracy of approximately 43.74%, but its UP-class recall and F1-score were very low.
- LSTM achieved a higher precision for UP predictions but still had low UP-class recall.
- GRU provided the best balance between precision and recall and consequently achieved the highest F1-score.
- The overall predictive performance is below 50% accuracy, showing that next-day stock-price direction is difficult to classify using only historical closing prices.
- Stock prices are influenced by many external factors that are not represented by the single closing-price feature used in this experiment.
- The chronological train-test split and training-only scaler fitting are appropriate for time-series data.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- yfinance
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook / Google Colab

## Files

Recommended assignment folder:

```text
Assignment-05/
├── Assignment-05.docx
├── Assignment-05.ipynb
└── README.md
```

The dataset does not need to be stored in the repository because it is downloaded automatically using `yfinance`.

## How to Run

1. Open `Assignment-05.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required packages if necessary.
3. Run the notebook cells sequentially.
4. The notebook downloads the AAPL dataset automatically.
5. The notebook trains the SimpleRNN, LSTM, and GRU models.
6. View the generated metrics, classification reports, confusion matrices, comparison chart, and next-day direction predictions.

## Conclusion

The experiment successfully implemented and compared SimpleRNN, LSTM, and GRU models for sequence classification using historical AAPL stock-price data. The previous 30 trading days were used as sequential input, and the target represented the direction of the following trading day.

Among the three architectures, **GRU performed best according to F1-score, achieving 0.4487**, followed by LSTM with 0.1165 and SimpleRNN with 0.0139. The results demonstrate the differences between recurrent architectures and show that GRU was better able to balance precision and recall for this particular dataset and experimental setup.

However, the relatively low test accuracy indicates that predicting stock-price direction from historical closing prices alone is difficult. More features, longer historical context, alternative architectures, and additional market information could be explored in future work.
