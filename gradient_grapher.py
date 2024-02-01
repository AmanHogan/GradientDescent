""" Graphs the gradient, data, and loss over epochs"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

def plot_gradient(g_descent):

    # if theres only one feature, plot in 2d
    if g_descent.n == 1:
        x = np.linspace(g_descent.x[:, 0].min(), g_descent.x[:, 0].max(), 100)
        plt.scatter(g_descent.x, g_descent.y, c='blue')
        plt.plot(x, g_descent.w[0] * x + g_descent.w[1], linestyle='solid') 
        plt.show()

    # if theres two features, plot in 3d
    if g_descent.n == 2:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot the data points
        ax.scatter(g_descent.x[:, 0], g_descent.x[:, 1], g_descent.y, c='r', marker='o')

        # Create a meshgrid for the plane
        x = np.linspace(g_descent.x[:, 0].min(), g_descent.x[:, 0].max(), 100)
        y = np.linspace(g_descent.x[:, 1].min(), g_descent.x[:, 1].max(), 100)
        x, y = np.meshgrid(x, y)

        # Calculate the corresponding z values using the linear regression model
        z =  g_descent.w[0]*x + g_descent.w[1]*y + g_descent.w[2]

        # Plot the plane
        ax.plot_surface(x, y, z, cmap='viridis')

        # Set labels
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Y')
        plt.show()


def plot_loss(g_descent):

    # Show the plot
    plt.show()
    plt.plot(range(g_descent.epochs), g_descent.errors, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Iterations')
    plt.legend()
    plt.show()