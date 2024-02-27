import numpy as np

def gradient_descent(X, y, learning_rate, iterations):
    # Initialize weights and bias
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(iterations):
        # Compute predictions
        predictions = np.dot(X, weights) + bias

        # Error
        error = predictions - y

        # Compute gradients
        d_weights = (1/num_samples) * np.dot(X.T, error)
        d_bias = (1/num_samples) * np.sum(error)

        # Update weights and bias
        weights -= learning_rate * d_weights
        bias -= learning_rate * d_bias

    return weights, bias

# Linear regression - simple example
x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([2, 3, 4, 5])
learning_rate = 0.01
iterations = 1000
weights, bias = gradient_descent(x, y, learning_rate, iterations)
print("Weights:", weights)
print("Bias:", bias)
