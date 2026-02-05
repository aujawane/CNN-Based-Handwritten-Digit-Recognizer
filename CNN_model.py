from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Load MNIST dataset
mnist = fetch_openml("mnist_784", as_frame=False, cache=False)
print("dataset shape:", mnist.data.shape)  # 70000 x 784

# Preprocess data
X = mnist.data.astype("float32")          # ensure float32 for PyTorch/skorch
y = mnist.target.astype("int64")          # ensure integer labels
X /= 255.0                                # scale pixel values to [0, 1]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape, "y_test.shape:", y_test.shape)

# Network dimensions
mnist_dim = X.shape[1]                    # 784
hidden_dim = mnist_dim // 8               # 98
output_dim = len(np.unique(mnist.target)) # 10
print("dims (input, hidden, output):", mnist_dim, hidden_dim, output_dim)

# Neural network module (classifier)
class ClassifierModule(nn.Module):
    def __init__(self, input_dim=mnist_dim, hidden_dim=hidden_dim,
                 output_dim=output_dim, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X


# Reproducibility
torch.manual_seed(0)

# Skorch NeuralNetClassifier
net = NeuralNetClassifier(
    module=ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
)

# Train the model
net.fit(X_train, y_train)

# Make predictions on the test set
y_pred = net.predict(X_test)

# Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Identify misclassified examples (boolean mask)
error_mask = y_pred != y_test
print("Number of misclassified examples:", error_mask.sum())
