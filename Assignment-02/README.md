# Assignment 02 – Single-Layer Perceptron

## Aim

To implement a Single-Layer Perceptron for binary classification and
analyze the effect of weights, bias, and activation functions.

## Problem Statement

Implement a Single-Layer Perceptron for binary classification and analyze
the effect of weights, bias, and activation functions.

## Description

This assignment implements a simple neural network using TensorFlow/Keras
to perform binary classification on an AND gate.

The model consists of a single Dense neuron with a sigmoid activation
function. The model is trained using Stochastic Gradient Descent (SGD)
and Binary Crossentropy loss.

After training, the model is used to:

- Predict the output for all possible AND gate input combinations.
- Convert the predicted probabilities into binary outputs.
- Display the learned weights and bias.
- Analyze the role of weights, bias, and the activation function.

## Dataset

No external dataset is required.

The input data consists of the four possible combinations of a two-input
AND gate:

| Input 1 | Input 2 | Expected Output |
| ------- | ------- | --------------- |
| 0       | 0       | 0               |
| 0       | 1       | 0               |
| 1       | 0       | 0               |
| 1       | 1       | 1               |

## Model Architecture

The perceptron uses a single Dense layer:

- **Input features:** 2
- **Neurons:** 1
- **Activation:** Sigmoid
- **Optimizer:** SGD
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy
- **Training Epochs:** 500

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook / Google Colab

## Files

| File                  | Description                                            |
| --------------------- | ------------------------------------------------------ |
| `Assignment-02.docx`  | Assignment problem statement and submission document   |
| `Assignment-02.ipynb` | Complete implementation of the Single-Layer Perceptron |
| `README.md`           | Description and documentation of the assignment        |

## How to Run

1. Open `Assignment-02.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install TensorFlow and NumPy if they are not already available.
3. Run the cells sequentially.
4. Observe the predicted values, binary outputs, learned weights, and bias.

## Results

The trained perceptron produces predictions for all four possible AND gate
inputs. A threshold of 0.5 is applied to the predicted probabilities to
obtain the final binary outputs.

The learned weights and bias are also displayed in the notebook.

## Conclusion

A Single-Layer Perceptron with a sigmoid activation function can be used
to learn the binary AND gate. The learned weights determine the influence
of the input features, while the bias shifts the decision boundary.
The sigmoid activation converts the weighted sum into a probability-like
output between 0 and 1.
