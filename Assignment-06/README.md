# Assignment 06 – CNN-Based Tomato Leaf Disease Classification

## Aim

To design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato Leaf Disease dataset.

## Problem Statement

Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.

## Dataset

This assignment uses the Tomato Leaf Disease dataset containing separate training and validation images.

Expected structure:

```text
tomato/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

The disease classes are detected automatically from the folder structure.

## Data Preprocessing

- Images are resized to 128 × 128 pixels.
- Pixel values are normalized to the range 0–1 by dividing by 255.
- Labels are assigned according to the disease-class folders.
- Categorical labels are used for multi-class classification.
- Batch size is 32.

## Methodology

1. Import the required TensorFlow, Keras, and Matplotlib libraries.
2. Specify the paths to the training and validation datasets.
3. Create `ImageDataGenerator` objects for training and validation.
4. Normalize image pixel values using rescaling.
5. Load training and validation images using `flow_from_directory()`.
6. Resize images to 128 × 128 pixels.
7. Automatically assign class labels from the directory structure.
8. Initialize the CNN model.
9. Add a convolutional layer with 32 filters, 3 × 3 kernel, and ReLU activation.
10. Add a 2 × 2 MaxPooling layer.
11. Add a convolutional layer with 64 filters, 3 × 3 kernel, and ReLU activation.
12. Add another 2 × 2 MaxPooling layer.
13. Add a convolutional layer with 128 filters, 3 × 3 kernel, and ReLU activation.
14. Add another 2 × 2 MaxPooling layer.
15. Flatten the extracted feature maps.
16. Add a Dense layer with 128 neurons and ReLU activation.
17. Apply Dropout with a rate of 0.5.
18. Add a Softmax output layer with one neuron for each disease class.
19. Compile the model using Adam and categorical cross-entropy.
20. Train the CNN for 10 epochs while monitoring validation performance.
21. Evaluate the model using the validation dataset.
22. Plot training and validation accuracy against epochs.
23. Save the trained model as `tomato_disease_cnn.keras`.

## CNN Architecture

```text
Input Image (128 × 128 × 3)
        ↓
Conv2D – 32 filters, 3 × 3, ReLU
        ↓
MaxPooling2D – 2 × 2
        ↓
Conv2D – 64 filters, 3 × 3, ReLU
        ↓
MaxPooling2D – 2 × 2
        ↓
Conv2D – 128 filters, 3 × 3, ReLU
        ↓
MaxPooling2D – 2 × 2
        ↓
Flatten
        ↓
Dense – 128 neurons, ReLU
        ↓
Dropout – 0.5
        ↓
Dense – Number of disease classes, Softmax
```

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 128 × 128 |
| Input Channels | 3 (RGB) |
| Batch Size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Dense Layer | 128 neurons |
| Dropout Rate | 0.5 |
| Output Activation | Softmax |

## Evaluation

The model is evaluated using the validation dataset.

The following are reported:

- Validation Loss
- Validation Accuracy

A graph comparing training accuracy and validation accuracy across the 10 epochs is also generated.

## Output

The script produces:

- Dataset loading information
- CNN model summary
- Training progress
- Validation loss
- Validation accuracy
- Training vs validation accuracy graph
- Saved CNN model

The trained model is saved as:

```text
tomato_disease_cnn.keras
```

## Technologies Used

- Python
- TensorFlow
- Keras
- Matplotlib
- NumPy
- Google Colab / Jupyter Notebook

## Files

Recommended repository structure:

```text
Assignment-06/
├── Assignment 6.docx
├── train.py
└── README.md
```

The `tomato/` dataset and virtual-environment folders are kept locally and should not be committed to GitHub.

## How to Run

### 1. Prepare the Dataset

Place the Tomato Leaf Disease dataset inside the `Assignment-06` folder:

```text
Assignment-06/
└── tomato/
    ├── train/
    └── val/
```

Each directory should contain subdirectories corresponding to the disease classes.

### 2. Install Required Libraries

```bash
pip install tensorflow matplotlib
```

### 3. Run the Script

From the `Assignment-06` directory:

```bash
python train.py
```

The script loads the images, preprocesses them, trains the CNN for 10 epochs, evaluates it on the validation set, displays the accuracy graph, and saves the trained model.

## Conclusion

The CNN successfully implements image classification for Tomato Leaf Disease images. The model uses convolutional layers to extract visual features, max-pooling layers to reduce spatial dimensions, a fully connected layer for classification, and dropout to reduce overfitting. The Softmax output layer provides class probabilities for the different disease categories.

The model is trained for 10 epochs using the Adam optimizer and categorical cross-entropy loss and is evaluated using validation accuracy and loss. The training and validation accuracy graph helps analyze the model's learning behavior across epochs. Overall, the experiment demonstrates the practical application of CNNs for automated plant disease image classification.
