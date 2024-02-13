import numpy as np
import random
import matplotlib.pyplot as plt

def predict(x, weights):
    return np.dot(x, weights)

def gradient_descent(x, y, learning_rate, epochs):
    # Adding a column of ones for the intercept term
    x = np.c_[np.ones(x.shape[0]), x]
    num_features = x.shape[1]
    loss = []

    # Initializing weights with random values
    weights = np.random.rand(num_features)
    # print(weights)

    for _ in range(epochs):
        # Calculate predictions
        predictions = predict(x, weights)

        # Calculate errors
        errors = predictions - y
        l = np.sqrt(np.mean(np.square(predictions - y)))
        loss.append(l)

        # Calculating gradients
        # dot product of features and errors.
        # Dividing by len(y) essentially computes the average gradient across all samples in the dataset.
        # This normalization step helps to ensure that the step size taken during each update of the
        # weights is consistent and independent of the size of the dataset.

        # The dot product is used when calculating the gradients because it efficiently computes the sum of the
        # element-wise multiplication between the input features and the errors. This operation effectively
        # calculates the partial derivative of the loss function with respect to each weight.

        gradients = np.dot(x.T, errors) / len(y)

        # Updating weights
        weights -= learning_rate * gradients

    return weights, loss

# Independent and Dependent Variables
x = np.array([[random.randrange(2,50) for i in range(20)],
              [random.randrange(3,30) for i in range(20)]])
x = x.T
y = np.array([round(random.uniform(1,5), 2) for i in range(20)])

learning_rate = 0.001
epochs = 1000

# Fitting the line
weights, loss = gradient_descent(x, y, learning_rate, epochs)
print("Weights:", weights)
# Output: Weights: [0.65459371 0.00933853 0.11411113] - [y_intercept, weight of feature 1, weight of feature 2]
# y = 0.65459371 + x1*0.00933853 + x2*0.11411113

plt.plot(loss)
plt.show()
