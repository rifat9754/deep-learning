# Movie Review Sentiment Classification with a Transformer

A binary sentiment classification project built using **TensorFlow and Keras**, implementing a custom Transformer Encoder architecture to classify IMDB movie reviews as Positive or Negative.

## Overview

This project explores how Transformer-based self-attention can be applied to Natural Language Processing (NLP) and sentiment classification.

The model takes a movie review as input and predicts its sentiment:

- `0` — Negative
- `1` — Positive

Instead of using a pretrained Transformer model, this project implements a custom encoder architecture using token embeddings, trainable positional embeddings, multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

The model is trained and evaluated using the IMDB Movie Reviews dataset.

## Key Features

- IMDB dataset loading and exploratory data analysis
- Text preprocessing and sequence padding
- Custom token and positional embeddings
- Multi-head self-attention
- Padding-aware attention masking
- Transformer Encoder block
- Feed-forward neural network
- Residual connections and layer normalization
- Masked global average pooling
- Binary sentiment classification
- Training and validation visualization
- Classification report and confusion matrix
- Custom review prediction
- Model saving and loading

## Model Architecture

The model follows this architecture:

```text
Input Review
     |
     v
Token IDs
     |
     v
Positional Embedding
(Token Embedding + Position Embedding)
     |
     v
Transformer Encoder Block
     |
     +-- Multi-Head Self-Attention
     |
     +-- Residual Connection
     |
     +-- Layer Normalization
     |
     +-- Feed-Forward Network
     |
     +-- Residual Connection
     |
     +-- Layer Normalization
     |
     v
Masked Global Average Pooling
     |
     v
Dropout
     |
     v
Dense Layer (64, ReLU)
     |
     v
Dropout
     |
     v
Dense Layer (1, Sigmoid)
     |
     v
Sentiment Prediction
```

The attention mechanism is implemented using TensorFlow/Keras `MultiHeadAttention`, while the surrounding Transformer Encoder block is custom-built.

## Dataset

**Dataset:** IMDB Movie Reviews

The dataset contains 50,000 labeled movie reviews:

| Dataset Split | Samples |
|---|---:|
| Original Training Set | 25,000 |
| Training Set (after split) | 20,000 |
| Validation Set | 5,000 |
| Test Set | 25,000 |
| Total | 50,000 |

The original training set is divided into training and validation subsets using an 80:20 stratified split.

The official test set is kept separate from training and validation.

The dataset is loaded through:

```python
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)
```

## Data Preprocessing

The following preprocessing steps are applied:

1. Limit the vocabulary to 10,000 words.
2. Convert reviews into integer token sequences.
3. Set the maximum sequence length to 200.
4. Apply post-padding to shorter sequences.
5. Apply post-truncation to longer sequences.
6. Split the training data into training and validation sets.
7. Preserve class distribution using stratified sampling.

### Experiment Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 10,000 |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 64 |
| Attention Heads | 4 |
| Feed-Forward Dimension | 128 |
| Dropout Rate | 0.2 |
| Batch Size | 64 |
| Maximum Epochs | 8 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |

Early stopping, model checkpointing, and learning-rate reduction are used to help control training.

## Experimental Results

The model achieved the following results on the held-out IMDB test set:

### Overall Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **85.50%** |
| Test Loss | 0.3306 |
| ROC-AUC | **0.9328** |

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.8615 | 0.8460 | 0.8537 |
| Positive | 0.8487 | 0.8640 | 0.8563 |
| **Overall Accuracy** | | | **0.8550** |

These are the recorded results from the current notebook execution. Results may vary with different software versions, hardware, and training configurations.

## Training Visualization

The notebook includes:

- Training vs. Validation Accuracy
- Training vs. Validation Loss
- Classification Report
- Confusion Matrix

These visualizations help evaluate model performance and identify potential overfitting.

## Custom Review Prediction

The trained model can classify user-provided movie reviews.

Example:

```python
predict_sentiment(
    "The movie was wonderful, emotional, and beautifully acted."
)

predict_sentiment(
    "I found this film boring and a complete waste of time."
)
```

Example output from the notebook:

```text
Review: The movie was wonderful, emotional, and beautifully acted.
Predicted sentiment: Positive
Positive probability: 0.9994

Review: I found this film boring and a complete waste of time.
Predicted sentiment: Negative
Positive probability: 0.0006
```

The probability represents the model's estimated positive sentiment score, not a guarantee of correctness.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab / Jupyter Notebook

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook
```

Open:

```text
Transformer_Sentiment_Analysis.ipynb
```

Alternatively, upload the notebook to Google Colab and execute the cells sequentially.

### 4. Train the model

Run the notebook from the beginning to:

- Load and preprocess the dataset.
- Build the Transformer model.
- Train and validate the model.
- Evaluate test performance.
- Generate custom predictions.
- Save the trained model.

## Model Artifacts

The notebook saves the trained model in Keras format:

```text
models/
└── transformer_sentiment_classifier.keras
```

A best-validation-loss checkpoint is also saved:

```text
models/
└── best_transformer_sentiment.keras
```

The custom layer definitions should be available when loading the saved model.

## Project Structure

A suggested repository structure:

```text
transformer-sentiment-analysis/
│
├── Transformer_Sentiment_Analysis.ipynb
│
├── models/
│   ├── transformer_sentiment_classifier.keras
│   └── best_transformer_sentiment.keras
│
└── requirements.txt README.md

```

The artifacts directory is generated during notebook execution.

## Limitations

- The model uses a single Transformer Encoder block.
- The vocabulary is limited to 10,000 words.
- Reviews are truncated or padded to 200 tokens.
- The model is trained on the IMDB movie review domain.
- The custom review tokenizer may differ slightly from the original dataset tokenization.
- The model may not generalize equally well to other domains, languages, or informal text.

## Future Improvements

Potential directions for further development:

- Experiment with multiple Transformer Encoder blocks.
- Tune embedding dimensions and attention heads.
- Experiment with different sequence lengths.
- Compare against LSTM and GRU architectures.
- Visualize attention weights.
- Perform more detailed error analysis.
- Explore different optimizers and learning-rate schedules.
- Build an interactive sentiment prediction interface.
- Evaluate generalization on reviews from other domains.

## Learning Outcomes

Through this project, I explored and implemented:

- NLP preprocessing and tokenization
- Word embeddings and positional embeddings
- Self-attention and multi-head attention
- Transformer Encoder architecture
- Residual connections and layer normalization
- Padding masks and masked pooling
- Binary classification with neural networks
- Model evaluation and error analysis
- Model serialization and inference

## Author

**Md. Ahanaf Tahmid Rifat**

Computer Science and Engineering Graduate  
Interested in Artificial Intelligence, Machine Learning and Deep Learning.

---

*This project is part of my hands-on Deep Learning journey, focusing on understanding and implementing Transformer architectures for real-world NLP tasks.*