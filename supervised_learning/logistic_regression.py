import numpy as np
import random

def sigmoid(z):
    """Sigmoid function: range [0,1]
    For large positive values of z; e^-z approaches 0, so sigmoid(z) approaches 1.
    For large negative values of z; e^-z approaches infinity, so sigmoid(z) approaches 0
    For z=0, sigmoid(z) = 1/2
    """
    return 1 / (1 + np.exp(-z))

def predict_prob(x, weights):
    """Predict probabilities."""
    return sigmoid(np.dot(weights, x))

def predict(x, weights, threshold=0.5):
    """Predict classes."""
    probabilities = predict_prob(x, weights)
    return (probabilities >= threshold).astype(int)

def gradient_descent(x, y, learning_rate, epochs):
    """Gradient descent optimization."""
    num_features = x.shape[0]
    num_samples = x.shape[1]
    weights = np.zeros(num_features)

    for epoch in range(epochs):
        # Calculate predictions
        predictions = predict_prob(x, weights)

        # Calculate errors
        errors = predictions - y

        # Calculate gradients
        gradients = np.dot(x, errors) / num_samples

        # Update weights
        weights -= learning_rate * gradients

    return weights

# Independent and Dependent Variable
x = np.array([[round(random.uniform(1,10), 2) for _ in range(20)],
              [round(random.uniform(1,10), 2) for _ in range(20)]])
x = np.vstack((np.ones(x.shape[1]), x))  # Adding bias term
print(x.T)
y = np.array([random.randint(0, 1) for _ in range(20)])

learning_rate = 0.01
epochs = 1000

# Fitting the logistic regression model
weights = gradient_descent(x, y, learning_rate, epochs)

print("Weights:", weights)

# Example prediction
example_x = np.array([1, 5.7, 8.3])  # Example features including the bias term
predicted_prob = predict_prob(example_x, weights)
predicted_class = predict(example_x, weights)
print("Example predicted probability:", predicted_prob)
print("Example predicted class:", predicted_class)
