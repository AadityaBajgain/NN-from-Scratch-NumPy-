# NN From Scratch (NumPy)

Simple, from-scratch MNIST digit classifier implemented only with NumPy. The notebook
walks through data loading, forward propagation, backpropagation, and gradient descent
for a small fully connected neural network.

## Contents

- `digitrecognizer-from-scratch-numpy.ipynb` — NumPy-only neural network for MNIST.
- `train.csv` — MNIST training data used by the notebook (Kaggle "Digit Recognizer").

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib jupyter
```

3. Launch Jupyter and open the notebook:

```bash
jupyter notebook digitrecognizer-from-scratch-numpy.ipynb
```

## Approach

This project uses a 2-layer fully connected network:

- Input: 784 features (28x28 pixels flattened).
- Hidden layer: 10 neurons, ReLU activation.
- Output layer: 10 neurons, softmax activation (digits 0-9).
- Weight initialization: He initialization (`sqrt(2 / fan_in)`).
- Optimization: full-batch gradient descent.
- Loss: softmax + cross-entropy.

## Math Behind It

Let:

- `X` in R^(784 x m) be the input matrix (m examples).
- `Y` in R^(m) be the labels.
- `W1` in R^(10 x 784), `b1` in R^(10 x 1)
- `W2` in R^(10 x 10), `b2` in R^(10 x 1)

### Forward Propagation

```
Z1 = W1 X + b1
A1 = ReLU(Z1)
Z2 = W2 A1 + b2
A2 = softmax(Z2)
```

Where:

```
ReLU(z) = max(0, z)
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

### Loss (Softmax + Cross-Entropy)

```
L = -(1/m) * sum_{i=1..m} sum_{k=1..10} y_k^(i) * log(a2_k^(i))
```

`y^(i)` is the one-hot label for example `i`.

### Backward Propagation

```
dZ2 = A2 - Y_one_hot
dW2 = (1/m) * dZ2 A1^T
db2 = (1/m) * sum(dZ2)

dZ1 = (W2^T dZ2) * ReLU'(Z1)
dW1 = (1/m) * dZ1 X^T
db1 = (1/m) * sum(dZ1)
```

```
ReLU'(z) = 1 if z > 0 else 0
```

Note: the notebook uses `W2` directly in `dZ1` since it is 10x10 and the shapes
still align; conceptually this is `W2^T`.

### Gradient Descent Update

```
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
```

`alpha` is the learning rate.

## Notes

- The notebook expects `train.csv` to be in this folder.
- Dataset source (Kaggle Digit Recognizer):

```text
https://www.kaggle.com/competitions/digit-recognizer/data
```
- To try a different dataset, update the file path and adjust input/output sizes.
