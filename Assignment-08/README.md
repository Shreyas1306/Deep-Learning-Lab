# Assignment 08 — Pre-trained BERT for Sentiment Analysis

## Aim

To implement a pre-trained BERT model for sentiment analysis using a sample text dataset and evaluate its classification performance.

## Problem Statement

Implement a pre-trained BERT model for sentiment analysis or text classification on a sample dataset.

## Dataset

This practical uses the **SST-2 (Stanford Sentiment Treebank 2)** dataset through the Hugging Face `datasets` library.

- **Dataset source:** `nyu-mll/glue`, configuration `sst2`
- **Split used:** Validation
- **Samples evaluated:** 500 randomly shuffled samples
- **Random seed:** 42
- **Input:** English movie-review sentences
- **Class 0:** Negative sentiment
- **Class 1:** Positive sentiment

The dataset is shuffled with a fixed seed before selecting 500 samples, making the sample selection reproducible.

## Pre-trained Model

The notebook loads the already fine-tuned model:

`textattack/bert-base-uncased-SST-2`

The matching tokenizer is loaded using Hugging Face Transformers. The model is used in evaluation mode to classify text; this notebook **does not train or fine-tune BERT**.

## Methodology

1. Install the required libraries.
2. Import PyTorch, Hugging Face Datasets and Transformers, and Scikit-learn evaluation utilities.
3. Set random seeds for reproducibility.
4. Select GPU when CUDA is available; otherwise, use CPU.
5. Load the pre-trained BERT tokenizer and sequence-classification model.
6. Load the SST-2 validation split and select 500 samples.
7. Tokenize sentences in batches, using padding and truncation with a maximum sequence length of 128.
8. Run inference without gradient calculation.
9. Apply softmax to model logits to obtain class probabilities.
10. Predict Negative or Positive sentiment.
11. Calculate accuracy, precision, recall, and F1-score.
12. Generate a classification report and confusion matrix.
13. Inspect misclassified sentences and the confidence of incorrect predictions.
14. Test the model on four custom sentences.
15. Save prediction-level results and summary metrics as CSV files.

## Implementation Details

| Parameter | Value |
|---|---|
| Model | `textattack/bert-base-uncased-SST-2` |
| Framework | PyTorch |
| Model interface | Hugging Face Transformers |
| Dataset | SST-2 validation split |
| Sample size | 500 |
| Batch size | 32 |
| Maximum sequence length | 128 tokens |
| Random seed | 42 |
| Gradient calculation | Disabled during inference |
| Fine-tuning performed | No |
| Device | CUDA when available; otherwise CPU |

## Evaluation Metrics

- **Accuracy:** Fraction of all sentences classified correctly.
- **Precision:** Fraction of predicted positive sentences that are actually positive.
- **Recall:** Fraction of actual positive sentences correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts correct and incorrect predictions for negative and positive sentiment.

The notebook also prints a class-wise classification report and lists the ten most confident incorrect predictions, where available.

## Results

The practical sheet reports the following results on the 500 selected validation samples:

| Metric | Score |
|---|---:|
| Accuracy | **91.8%** |
| Precision | 92.3% |
| Recall | 92.0% |
| F1-score | 92.1% |

### Confusion Matrix

| Actual / Predicted | Negative | Positive |
|---|---:|---:|
| Negative | 219 | 20 |
| Positive | 21 | 240 |

- Correct predictions: **459 out of 500**
- Misclassified sentences: **41 out of 500**

The notebook also evaluates four custom sentences. In the recorded output, the model classifies positive-sounding examples as Positive and negative-sounding examples as Negative.

## Output Files

The notebook saves two CSV files in the Google Colab `/content` directory:

- `bert_sentiment_predictions.csv` — sentences, actual labels, predicted labels, and positive-class probabilities.
- `bert_evaluation_metrics.csv` — summary evaluation metrics.

These are generated output files; they do not need to be committed unless the assignment specifically requires them.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Files

Recommended assignment folder structure:

```text
Assignment-08/
├── Assignment-08.docx
├── Assignment_08.ipynb
└── README.md
```

The dataset and model files are downloaded by the Hugging Face libraries and cached locally. They should not normally be added to the repository.

## How to Run

1. Open `Assignment_08.ipynb` in Google Colab or a Jupyter environment that supports notebook installation commands.
2. Run the notebook cells in order.
3. The first section installs the required packages.
4. The notebook downloads the SST-2 validation data, tokenizer, and pre-trained model.
5. Review the metrics, classification report, confusion matrix, misclassified examples, and custom-sentence predictions.
6. In Google Colab, the CSV output files are saved under `/content`.

Required packages can be installed with:

```bash
pip install transformers datasets accelerate scikit-learn seaborn torch numpy pandas matplotlib
```

## Conclusion

The practical successfully applies an existing pre-trained BERT model to binary sentiment classification on 500 SST-2 validation samples. The reported accuracy is **91.8%**, with an F1-score of **92.1%**. The confusion matrix shows 459 correct predictions and 41 errors.

The experiment demonstrates how a pre-trained transformer can be reused for text classification without training the model from scratch. Some sentences remain difficult to classify because sentiment can depend on context, phrasing, and ambiguity.
