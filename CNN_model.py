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

# -------------------------
# Load and preprocess data
# -------------------------
mnist = fetch_openml("mnist_784", as_frame=False, cache=False)
print("dataset shape:", mnist.data.shape)  # 70000 x 784

X = mnist.data.astype("float32")   # float32 for PyTorch/skorch
y = mnist.target.astype("int64")   # integer labels
X /= 255.0                         # scale pixel values to [0, 1]

# Single train / test split (use these for both MLP and CNN)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape, "y_test.shape:", y_test.shape)

# -------------------------
# MLP (skorch) definitions
# -------------------------
mnist_dim = X.shape[1]                    # 784
hidden_dim = mnist_dim // 8               # 98
output_dim = len(np.unique(mnist.target)) # 10
print("dims (input, hidden, output):", mnist_dim, hidden_dim, output_dim)

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

# Reproducibility for PyTorch
torch.manual_seed(0)

# Skorch MLP classifier
net = NeuralNetClassifier(
    module=ClassifierModule,
    max_epochs=20,
    lr=0.1,
    device=device,
    verbose=0
)

# Train MLP
net.fit(X_train, y_train)

# MLP predictions and evaluation
y_pred = net.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"MLP test accuracy: {acc:.4f}")

# Keep this boolean mask for "previously misclassified examples"
error_mask = y_pred != y_test
print("Number of MLP misclassified examples:", error_mask.sum())

# -------------------------
# Prepare data for CNN
# -------------------------
# Reshape only the train/test splits so indices remain aligned
XCnn_train = X_train.reshape(-1, 1, 28, 28)
XCnn_test  = X_test.reshape(-1, 1, 28, 28)
print("XCnn_train.shape:", XCnn_train.shape, "XCnn_test.shape:", XCnn_test.shape)

# -------------------------
# CNN (skorch) definitions
# -------------------------
class Cnn(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        # after two 3x3 convs + 2x2 maxpools on 28x28 input, the feature map size is 64 x 5 x 5 = 1600
        self.fc1 = nn.Linear(64 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))                    # -> 32 x 13 x 13
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # -> 64 x 5 x 5
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))                # flatten to (batch, 64*5*5)
        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

# Reproducibility
torch.manual_seed(0)

cnn = NeuralNetClassifier(
    module=Cnn,
    max_epochs=10,
    lr=0.002,
    optimizer=torch.optim.Adam,
    device=device,
    verbose=0
)

# Train CNN
cnn.fit(XCnn_train, y_train)

# CNN predictions and evaluation
y_pred_cnn = cnn.predict(XCnn_test)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
print(f"CNN test accuracy: {acc_cnn:.4f}")

# Evaluate CNN only on the examples MLP misclassified (same test indices)
if error_mask.sum() > 0:
    acc_cnn_on_mlp_errors = accuracy_score(y_test[error_mask], y_pred_cnn[error_mask])
    print(f"CNN accuracy on MLP-misclassified examples: {acc_cnn_on_mlp_errors:.4f}")
else:
    print("No MLP misclassified examples to evaluate.")

# -------------------------
# Plotting helper for misclassified examples
# -------------------------
def plot_examples(images, true_labels, pred_labels, max_plots=9, figsize=(7,7)):
    """
    images: (N, 784) or (N, 1, 28, 28)
    true_labels, pred_labels: arrays of length N
    """
    # normalize image shape to (N, 28, 28)
    if images.ndim == 4:               # (N, 1, 28, 28)
        imgs = images.squeeze(1)
    elif images.ndim == 2:             # (N, 784)
        imgs = images.reshape(-1, 28, 28)
    else:
        raise ValueError("Unsupported image shape for plotting")

    n = min(max_plots, len(imgs))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")
        plt.title(f"T:{true_labels[i]}\nP:{pred_labels[i]}")
    plt.tight_layout()
    plt.show()

# Plot a few CNN misclassified examples among those MLP misclassified examples
if error_mask.sum() > 0:
    # select indices of test-set examples that MLP misclassified
    mis_idx = np.nonzero(error_mask)[0]
    # take up to 9 of them
    sel = mis_idx[:9]
    # images from XCnn_test (shape: N,1,28,28), corresponding true/pred labels from y_test / y_pred_cnn
    plot_examples(XCnn_test[sel], y_test[sel], y_pred_cnn[sel], max_plots=len(sel))
else:
    print("No MLP misclassifications to plot.")
