import numpy as np

# When to use: Classification tasks
# Advantages: Penalizes confident wrong predictions, smooth and continous making it suitable for gradient based algos like gradient descent
# Disadvantages: Sensitive to outliers especially with unbiased data sets, it priortises to minimise large errors which may not always be desirable
# especially when dealing with imbalanced datasets.


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small value to prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Example usage:
y_true = np.array([1, 0, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.2])
print("Cross-Entropy Loss:", cross_entropy_loss(y_true, y_pred))
