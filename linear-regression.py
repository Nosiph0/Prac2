##!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def main():
    # Generate synthetic data
    np.random.seed(0)
    house_size = 2 * np.random.rand(100, 1)
    house_price = 4 + 3 * house_size + np.random.randn(100, 1)

    data = pd.DataFrame({'House Size': house_size.flatten(), 'House Price': house_price.flatten()})

    # Prepare the data
    X = data[['House Size']]
    y = data['House Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot the results
    plt.scatter(X_test, y_test, color='black', label='Actual data')
    plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Predicted line')
    plt.xlabel('House Size (in 1000 square feet)')
    plt.ylabel('House Price (in $1000)')
    plt.legend()
    plt.show()

    # Output the model parameters
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficient: {model.coef_[0]}")

if __name__ == "__main__":
    main()
