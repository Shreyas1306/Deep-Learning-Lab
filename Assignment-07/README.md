# Assignment 07 — Transfer Learning Model Comparison

## Aim

To implement transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 for image classification and compare their performance.

## Problem Statement

Implement transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification on the CIFAR-10 dataset, then compare their classification accuracy and training time.

## Dataset

This practical uses the **CIFAR-10** image-classification dataset, downloaded automatically through `torchvision.datasets.CIFAR10`.

CIFAR-10 contains images from these 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this notebook's CPU-based experiment, subsets are used to keep computation more manageable:

| Split | Images used |
|---|---:|
| Training subset | 2,000 |
| Testing subset | 1,000 |

The images are resized to **224 × 224** pixels, converted to tensors, and normalized using the channel means and standard deviations used for ImageNet-pretrained models.

## Methodology

1. Import PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn utilities.
2. Select CUDA if available; otherwise use CPU.
3. Download the CIFAR-10 training and testing datasets.
4. Resize and normalize images to match the input requirements of the pre-trained models.
5. Create training and testing subsets and load them with PyTorch `DataLoader`.
6. Load the pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 architectures with their default ImageNet weights.
7. Freeze the pre-trained parameters.
8. Replace each model's final classification layer with a new layer that outputs 10 classes.
9. Train the newly added classification layer using the training subset.
10. Evaluate each model on the test subset.
11. Record test accuracy and training time.
12. Compare the models using a results table and performance visualizations.

## Models Used

| Model | Classifier modification |
|---|---|
| AlexNet | Replace the final layer of `classifier` with 10 outputs |
| VGG16 | Replace the final layer of `classifier` with 10 outputs |
| ResNet50 | Replace `fc` with a 10-output linear layer |
| EfficientNetB0 | Replace the final classifier layer with 10 outputs |

The pre-trained feature-extraction layers are frozen. Only the newly added classification layer is trainable.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch / Torchvision |
| Pre-trained weights | ImageNet default weights |
| Input image size | 224 × 224 |
| Batch size | 32 |
| Epochs per model | 2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Cross-Entropy Loss |
| Training subset | 2,000 images |
| Testing subset | 1,000 images |
| Device recorded in notebook | CPU |

**Note:** The experiment uses reduced dataset subsets and two epochs, as configured in the notebook. The reported metrics are results from this setup, not a full-dataset benchmark.

## Results

The notebook recorded the following results:

| Model | Test Accuracy | Training Time (seconds) |
|---|---:|---:|
| AlexNet | 73.60% | 149.82 |
| VGG16 | **74.40%** | 2790.55 |
| ResNet50 | 69.90% | 1150.35 |
| EfficientNetB0 | 70.90% | 427.66 |

### Key observations

- **VGG16 achieved the highest test accuracy** at 74.40%.
- AlexNet achieved 73.60% accuracy and had the shortest recorded training time, at 149.82 seconds.
- EfficientNetB0 achieved 70.90% accuracy with a recorded training time of 427.66 seconds.
- ResNet50 achieved 69.90% accuracy and took 1150.35 seconds in this run.
- Training time depends on hardware, runtime conditions, and the chosen data subset, so the times should be interpreted as measurements from this particular run.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Files

Recommended folder structure:

```text
Assignment-07/
├── Assignment-07.docx
├── Assignment-07.ipynb
└── README.md
```

The CIFAR-10 dataset and downloaded pre-trained model weights are fetched by Torchvision and cached locally. They do not need to be committed to the repository.

## How to Run

1. Open `Assignment-07.ipynb` in Google Colab or Jupyter Notebook.
2. Ensure PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn are installed.
3. Run the notebook cells in order.
4. The notebook downloads CIFAR-10 and the required pre-trained weights automatically.
5. Review the model training logs, test accuracies, training times, and comparison outputs.

Example package installation command:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

## Conclusion

Transfer learning was implemented with AlexNet, VGG16, ResNet50, and EfficientNetB0 on the CIFAR-10 dataset. In the recorded experiment, VGG16 achieved the highest test accuracy at **74.40%**, followed by AlexNet at 73.60%, EfficientNetB0 at 70.90%, and ResNet50 at 69.90%. AlexNet had the shortest recorded training time.

The experiment illustrates how different pre-trained architectures can vary in classification performance and computational cost. Since this run used 2,000 training images, 1,000 testing images, and two epochs on CPU, the results should be treated as a comparison of this specific configuration rather than a definitive ranking of the architectures.
