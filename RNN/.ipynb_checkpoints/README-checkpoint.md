
# Next Word Prediction & Text Generation using LSTM

A Deep Learning project that predicts the next word in a sentence and generates text using an LSTM (Long Short-Term Memory) neural network built with TensorFlow and Keras.

## Project Overview

Next Word Prediction is a Natural Language Processing (NLP) task where a model receives a sequence of words and predicts the most likely next word.

For example:

- **Input:** `the meaning of life`
- **Output:** A predicted next word

This project uses an LSTM model to learn word patterns from a collection of quotes and generate text based on an input phrase.

**Note:** This project focuses on LSTM. The RNN architecture is not trained or evaluated.

## Dataset

The project uses a CSV dataset containing quotes and their authors.

- Dataset: `qoute_dataset.csv`
- Number of records: 3,038
- Main column: `quote`
- Additional column: `Author`

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Workflow

1. Load the dataset.
2. Perform text preprocessing.
3. Convert text into lowercase.
4. Remove punctuation.
5. Tokenize the text using Keras Tokenizer.
6. Generate input sequences and next-word targets.
7. Apply sequence padding.
8. Convert target labels into one-hot encoded vectors.
9. Build the LSTM model.
10. Train the LSTM model.
11. Visualize training and validation accuracy and loss.
12. Save and load the trained model.
13. Predict the next word.
14. Generate text using the trained model.

## LSTM Model Architecture

```text
Input Sequence
      |
  Embedding Layer
      |
   LSTM Layer
      |
 Dense Layer
 (Softmax Activation)
      |
 Predicted Next Word
```

### Model Configuration

| Parameter | Value |
|---|---:|
| Vocabulary Size | 8,978 |
| Maximum Sequence Length | 100 |
| Embedding Dimension | 32 |
| LSTM Units | 64 |
| Epochs | 50 |
| Batch Size | 16 |
| Validation Split | 10% |
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Output Activation | Softmax |

## Training Results

The model's training and validation performance is visualized using Matplotlib.

### Observations

- Training accuracy increases during training.
- Validation accuracy initially improves but later decreases.
- Training loss decreases consistently.
- Validation loss increases over time.

These results indicate overfitting. The model learns the training data but does not generalize well to unseen validation examples.

### Training Performance

The current model is an initial experiment. Further improvements are required to achieve better generalization.

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Dataset Setup

Place the dataset inside the `data` directory:

```text
data/qoute_dataset.csv
```

### 5. Run Jupyter Notebook

```bash
jupyter notebook
```

Open `Next_word_prediction.ipynb` and execute the cells sequentially.

## Next Word Prediction

Example:

```python
seed_text = "the meaning of life"

predicted_word = predict_next_word(seed_text)

print("Input:", seed_text)
print("Predicted next word:", predicted_word)
```

## Text Generation

The trained LSTM model can generate text by repeatedly predicting the next word.

```python
generated_text = generate_text(
    "the meaning of life",
    next_words=15
)

print(generated_text)
```

## Save and Load Model

Save the trained model:

```python
lstm_model.save("lstm_model.h5")
```

Load the model:

```python
from tensorflow.keras.models import load_model

lstm_model = load_model("lstm_model.h5")
```

## Project Structure

```text
Next-Word-Prediction/
│
├── data/
│   └── qoute_dataset.csv
│
├── Next_word_prediction.ipynb
├── lstm_model.h5
├── tokenizer.pkl
└── README.md
```

## Limitations

- The dataset contains a limited number of quotes.
- The current training process can lead to overfitting.
- Validation sequences may contain prefixes from quotes also represented in training data.
- The model may generate repetitive or grammatically incorrect text.
- A separate test dataset is not included in the current implementation.


## Author

**Md.Ahanaf Tahmid**

GitHub: [github.com/rifat9754](https://github.com/rifat9754)
