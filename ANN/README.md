import pypandoc, os, textwrap

readme = r"""# Iris Flower Classification Using Artificial Neural Networks (ANN)

A beginner-friendly deep learning project that classifies Iris flowers using **Python, TensorFlow, and Keras**. The notebook explores the Iris dataset, prepares the data, compares a Perceptron baseline with an Artificial Neural Network (ANN), and evaluates the ANN on a held-out test set.

## Project Overview

The goal is to predict the species of an Iris flower from four numerical measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

The project uses the Iris dataset provided by **Scikit-learn**. The target species names are added to the feature DataFrame using the dataset's target labels.

## Technologies and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras (TensorFlow's high-level deep learning API)

## Project Workflow

The notebook follows these steps:

1. **Import libraries** – load the Python packages used for data handling, visualization, preprocessing, evaluation, and deep learning.
2. **Load the Iris dataset** – retrieve the dataset with `sklearn.datasets.load_iris()` and add readable species names.
3. **Explore the dataset** – inspect species counts and dataset information, check missing values, and visualize feature relationships with a Seaborn pair plot.
4. **Separate features and target** – use the four flower measurements as `X` and `Species` as `y`.
5. **Encode species labels** – convert text species labels into integer class labels with `LabelEncoder`.
6. **Split the data** – create training and test sets using an 80/20 split, `random_state=42`, and stratification.
7. **Standardize features** – fit `StandardScaler` on the training features and apply the same transformation to the test features.
8. **Train a baseline model** – fit a Scikit-learn Perceptron and evaluate its predictions using accuracy and a classification report.
9. **Prepare ANN labels** – convert integer labels into one-hot encoded vectors with Keras `to_categorical`.
10. **Build the ANN** – create a Sequential neural network with two hidden layers and a Softmax output layer.
11. **Compile the model** – use the Adam optimizer, categorical crossentropy loss, and accuracy metric.
12. **Train the ANN** – train for up to 150 epochs with a batch size of 8, reserve 20% of the training data for validation, and use Early Stopping.
13. **Evaluate the ANN** – evaluate loss and accuracy on the held-out test set.
14. **Visualize learning curves** – plot training/validation accuracy and training/validation loss.

## ANN Architecture

The notebook defines the following network:

| Layer | Units | Activation |
|---|---:|---|
| Input | 4 features | — |
| Hidden layer 1 | 16 | ReLU |
| Hidden layer 2 | 8 | ReLU |
| Output layer | 3 classes | Softmax |

The number of output units is determined from the number of classes in the training labels.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Categorical crossentropy |
| Metric | Accuracy |
| Maximum epochs | 150 |
| Batch size | 8 |
| Validation split | 20% of the training data |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 15 epochs |
| Restore best weights | `True` |
| Train-test split | 80% / 20% |
| Random state | 42 |

Early Stopping monitors validation loss and stops training when it does not improve for the configured patience period. With `restore_best_weights=True`, the model restores the weights from the epoch with the best monitored validation loss.

## Dataset

The notebook loads the Iris dataset directly from Scikit-learn:

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.data.copy()
df["Species"] = iris.target.map(
    dict(enumerate(iris.target_names))
)
```

**No separate `Iris.csv` file is required.** The dataset is fetched from Scikit-learn's built-in dataset collection.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Create and activate an environment (optional)

If you use Conda:

```bash
conda create -n deep-learning python=3.11
conda activate deep-learning
```

If you already have a suitable Python environment, you can use it instead.

### 3. Install dependencies

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the ANN project notebook and run the cells from top to bottom.

## Evaluation

The notebook evaluates the Perceptron baseline with accuracy and a classification report. It evaluates the ANN on the test set using `model.evaluate()` and visualizes its training and validation accuracy/loss over epochs.

The actual metric values depend on running the notebook in your environment. No fixed accuracy result is claimed here.

## Repository Structure

```text
deep-learning/
└── ANN/
    ├── ANN_project.ipynb
    └── README.md
```

## Learning Outcomes

Through this project, you can practice:

- Exploring and visualizing a tabular dataset
- Separating input features and target labels
- Label encoding and one-hot encoding
- Train-test splitting and stratification
- Feature standardization without fitting on the test set
- Building a neural network with Keras `Sequential` and `Dense` layers
- Understanding ReLU and Softmax activations
- Compiling and training a classification model
- Using validation data and Early Stopping
- Evaluating a model and interpreting learning curves
- Comparing a simple Perceptron baseline with an ANN

## Notes

- Run the notebook cells in order because later steps depend on variables created earlier.
- The test set is kept separate from model training and validation.
- The notebook currently reports baseline classification metrics and ANN test loss/accuracy. It does not include a saved model artifact or an interactive prediction interface.

---

**Project:** Iris Flower Classification Using ANN  
**Framework:** TensorFlow / Keras  
**Dataset source:** Scikit-learn Iris dataset
"""

out = "/mnt/data/README.md"
pypandoc.convert_text(readme, "md", format="md", outputfile=out, extra_args=["--standalone"])
print(f"Created {out} ({os.path.getsize(out)} bytes)")
