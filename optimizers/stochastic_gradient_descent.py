import numpy as np

# It is similar to gradient descent but rather than giving the data all at once we either give
# in batches or as single data points - if batches then it is called batch gradient descent.

# When to use: Large datasets, in cases of online learning where data is received in streams and in the case of
# non-convex optimisation - where there are a lot of local minimas
def stochastic_gradient_descent(X, y, learning_rate, iterations):
    # Initialize weights and bias
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for _ in range(iterations):
        for i in range(num_samples):
            # Select a random sample
            sample_index = np.random.randint(num_samples)
            X_i = X[sample_index]
            y_i = y[sample_index]

            # Compute prediction
            prediction = np.dot(X_i, weights) + bias

            # Compute gradients
            d_weights = X_i * (prediction - y_i)
            d_bias = prediction - y_i

            # Update weights and bias
            weights -= learning_rate * d_weights
            bias -= learning_rate * d_bias

    return weights, bias

# Example usage: Linear regression
# X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
# y = np.array([2, 3, 4, 5])
# learning_rate = 0.01
# iterations = 1000
# weights, bias = stochastic_gradient_descent(X, y, learning_rate, iterations)
# print("Weights:", weights)
# print("Bias:", bias)
