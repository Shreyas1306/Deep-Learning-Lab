# Assignment 03 – Bank Marketing Classification using Neural Network

## Aim

To implement a neural network for binary classification using the Bank Marketing dataset and analyze the effect of learning rate and number of training epochs on model performance.

## Problem Statement

Build a neural network model to predict whether a customer will subscribe to a term deposit based on customer and campaign-related features.

The assignment also studies the effect of different learning rates and numbers of training epochs on model performance.

## Dataset

This assignment uses the **Bank Marketing dataset** from the UCI Machine Learning Repository.

The dataset contains information about customers contacted during a bank's direct marketing campaigns. The target variable `deposit` indicates whether the customer subscribed to a term deposit.

The `bank.csv` dataset used by the notebook is stored locally in the `dataset` folder.

### Dataset Source

Official UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset is included in this repository as:

```text
dataset/bank.csv
```

## Dataset Information

The dataset used in the notebook contains:

- 11,162 records
- 17 columns
- Numerical and categorical features
- Binary target variable: `deposit`

Target classes:

- `no` – Customer did not subscribe to a term deposit
- `yes` – Customer subscribed to a term deposit

## Workflow

The notebook follows these steps:

1. Load the dataset
2. Inspect the dataset structure and missing values
3. Analyze the target variable
4. Visualize the target distribution
5. Analyze term-deposit subscriptions by job category
6. Separate input features and target variable
7. Encode categorical features using one-hot encoding
8. Split the dataset into training and testing sets
9. Standardize the features
10. Build the neural network
11. Demonstrate forward propagation and backpropagation
12. Train the model
13. Analyze training and validation performance
14. Study the effect of different learning rates
15. Study the effect of different numbers of epochs
16. Train the final model
17. Evaluate the model on the test set
18. Generate predictions
19. Display the confusion matrix

## Model Architecture

The neural network consists of:

- Input layer
- Dense layer with 32 neurons and ReLU activation
- Dense layer with 16 neurons and ReLU activation
- Dense layer with 8 neurons and ReLU activation
- Output layer with 1 neuron and sigmoid activation

### Training Configuration

- Optimizer: Adam
- Loss function: Binary Crossentropy
- Evaluation metric: Accuracy
- Batch size: 32
- Learning rates tested: 0.0001, 0.001, 0.01, 0.1
- Epochs tested: 5, 10, 25, 50, 100

## Experiments

### Learning Rate Experiment

The following learning rates are compared:

- 0.0001
- 0.001
- 0.01
- 0.1

The effect of each learning rate is evaluated using validation performance and test performance.

### Epoch Experiment

The following numbers of epochs are compared:

- 5
- 10
- 25
- 50
- 100

The effect of training duration on model performance is evaluated using test accuracy and test loss.

## Evaluation

The final model is evaluated using:

- Test accuracy
- Test loss
- Predicted class labels
- Confusion matrix

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Google Colab / Jupyter Notebook

## Files

```text
Assignment-03/
├── Assignment-03.docx
├── Assignment-03.ipynb
├── README.md
└── dataset/
    └── bank.csv
```

| File | Description |
|------|-------------|
| `Assignment-03.docx` | Assignment document |
| `Assignment-03.ipynb` | Neural network implementation and experiments |
| `README.md` | Assignment documentation |
| `dataset/bank.csv` | Bank Marketing dataset used by the notebook |

## How to Run

### Google Colab

1. Open `Assignment-03.ipynb` in Google Colab.
2. Make sure the `bank.csv` dataset is available at `dataset/bank.csv`.
3. Run the notebook cells sequentially.
4. Review the data analysis, training results, learning-rate experiment, epoch experiment, final evaluation, and confusion matrix.

### Jupyter Notebook

1. Clone or download this repository.
2. Open the `Assignment-03` folder as the working directory.
3. Ensure the dataset is located at:

```text
dataset/bank.csv
```

4. Open `Assignment-03.ipynb`.
5. Run the notebook cells sequentially.

## Conclusion

The assignment demonstrates the use of a feed-forward neural network for binary classification and analyzes how preprocessing, learning rate, and the number of training epochs affect model performance.
