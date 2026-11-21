# Handwritten Digit Recognition using Perceptron, ANN and CNN

A practical deep learning project that recognizes handwritten digits (0–9) from 28×28 grayscale images. This project compares three neural-network approaches — a Perceptron baseline, a fully connected Artificial Neural Network (ANN), and a Convolutional Neural Network (CNN) — using the Kaggle Digit Recognizer dataset.

## Project Overview

Handwritten digit recognition is a classic image-classification problem. The goal is to train a model that can identify which digit appears in an image.

In this project, three architectures are trained and evaluated on the same held-out validation set:

1. **Perceptron** — a simple baseline with a Softmax output layer.
2. **ANN** — a fully connected neural network with two hidden layers.
3. **CNN** — a convolutional neural network that learns spatial features from images.

The project also visualizes training behavior, compares predictions, and evaluates the CNN using a classification report and confusion matrix.

## Dataset

**Dataset:** [Kaggle - Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data)

The dataset contains 28×28 grayscale handwritten digit images.

| File | Contents | Number of samples |
|---|---|---:|
| `train.csv` | 784 pixel columns and a `label` column | 42,000 |
| `test.csv` | 784 pixel columns; labels are not provided | 28,000 |

- **Image dimensions:** 28 × 28 pixels
- **Channels:** 1 (grayscale)
- **Classes:** 10 (digits 0–9)
- **Pixel values:** 0–255

The Kaggle test file does not include ground-truth labels. Therefore, this notebook creates a separate validation set from the labeled training data for model evaluation. The unlabeled Kaggle test set is reserved for generating predictions and a competition submission.

## Repository Structure

```text
CNN/
├── data/
│   ├── train.csv
│   └── test.csv
├── CNN_project.ipynb
└── README.md
```

> Keep the CSV files inside the `data/` directory. The notebook loads them using `data/train.csv` and `data/test.csv`.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## Project Workflow

### 1. Data Loading and Exploration

- Load `train.csv` and `test.csv` with Pandas.
- Inspect dataset dimensions, columns, data types, missing values, and duplicates.
- Separate pixel features from the target labels.

### 2. Train-Validation Split

The labeled training dataset is split using `train_test_split`:

- **80% training:** 33,600 images
- **20% validation:** 8,400 images
- `random_state=42`
- `stratify=y` to preserve class proportions.

The Kaggle `test.csv` contains 28,000 unlabeled images and is not used to calculate validation accuracy.

### 3. Preprocessing

- Convert pixel values to `float32`.
- Normalize pixel intensities from `[0, 255]` to `[0, 1]`.
- Reshape images to `(28, 28, 1)` for CNN input.
- One-hot encode the training and validation labels into 10 classes.

### 4. Model Architectures

#### Perceptron Baseline

```text
Input (28 × 28)
      ↓
Flatten
      ↓
Dense (10, Softmax)
```

The Perceptron provides a simple baseline for comparison.

#### Artificial Neural Network (ANN)

```text
Input (28 × 28)
      ↓
Flatten
      ↓
Dense (128, ReLU)
      ↓
Dense (64, ReLU)
      ↓
Dense (10, Softmax)
```

The hidden Dense layers learn non-linear combinations of pixel features.

#### Convolutional Neural Network (CNN)

```text
Input (28 × 28 × 1)
      ↓
Conv2D (32 filters, 3 × 3, ReLU)
      ↓
MaxPooling2D (2 × 2)
      ↓
Conv2D (64 filters, 3 × 3, ReLU)
      ↓
MaxPooling2D (2 × 2)
      ↓
Flatten
      ↓
Dense (128, ReLU)
      ↓
Dropout (0.5)
      ↓
Dense (10, Softmax)
```

The convolution layers learn local image patterns, pooling reduces feature-map dimensions, and Dropout helps reduce overfitting.

## Training Configuration

| Model | Optimizer | Loss | Epochs | Batch size |
|---|---|---|---:|---:|
| Perceptron | SGD | Categorical Crossentropy | 5 | 32 |
| ANN | Adam | Categorical Crossentropy | 5 | 32 |
| CNN | Adam | Categorical Crossentropy | 10 | 64 |

The Perceptron and ANN use a 10% validation split from the training subset during fitting. The CNN uses the separately held-out validation set through `validation_data`.

## Results

The notebook produced the following validation accuracies:

| Model | Validation Accuracy |
|---|---:|
| Perceptron | 88.9643% |
| ANN | 96.6190% |
| CNN | **98.9762%** |

**Best result in this run:** CNN achieved the highest validation accuracy at **98.9762%**.

These values are from the notebook run and represent validation performance, not Kaggle test accuracy. Results may vary when the notebook is trained again.

## Visualizations and Evaluation

The notebook includes:

- Dataset inspection and sample exploration
- Training and validation accuracy curves
- Training and validation loss curves
- Validation accuracy comparison across models
- Side-by-side predictions from Perceptron, ANN, and CNN
- CNN classification report (precision, recall, F1-score, and support)
- CNN confusion matrix

## Installation

Create and activate an environment, then install the required libraries.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## How to Run

1. Clone or download this repository.
2. Put `train.csv` and `test.csv` inside the `data/` folder.
3. Open a terminal in the `CNN/` project directory.
4. Start Jupyter Notebook:

```bash
jupyter notebook
```

5. Open `CNN_project.ipynb`.
6. Run the notebook cells from top to bottom.

## Kaggle Test Predictions

Since `test.csv` has no labels, it cannot be used to calculate accuracy, precision, recall, or a confusion matrix.

After training and evaluating the CNN on the validation set, generate predictions for the Kaggle test images:

```python
test_probabilities = cnn.predict(X_test_cnn, verbose=0)
test_predictions = np.argmax(test_probabilities, axis=1)
```

For the Kaggle Digit Recognizer competition, create a submission file:

```python
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(test_predictions) + 1),
    "Label": test_predictions
})

submission.to_csv("submission.csv", index=False)
```

## Key Learning Outcomes

- Understand the difference between a simple Perceptron, ANN, and CNN.
- Prepare image data for neural-network training.
- Normalize and reshape image tensors.
- Build multi-class classifiers with TensorFlow/Keras.
- Monitor training and validation curves.
- Evaluate classification models using precision, recall, F1-score, and a confusion matrix.
- Generate predictions for an unlabeled test dataset.

## Future Improvements

- Add EarlyStopping and ModelCheckpoint callbacks.
- Experiment with CNN filter sizes, kernel sizes, and network depth.
- Apply data augmentation and compare its effect.
- Visualize convolutional feature maps.
- Tune hyperparameters and compare additional architectures.
- Save the trained model and build a small interface for digit prediction.

---

This project is part of my **Deep Learning Journey**, where I document practical implementations and experiments while learning neural networks and modern AI techniques.

