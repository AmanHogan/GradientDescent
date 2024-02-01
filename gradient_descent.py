""" Responsible for calcu;ating gradient descent and parsing cmd line args"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def cmd_parser():
    """
    Parses the command line for arguments to be used in the gradient descent\n
    Returns:parse.args: command line arguments
    """
    parser = argparse.ArgumentParser(description = "Performs Stochastic or Batch Gradient Descent")
    parser.add_argument('filename',type=str, help='Filename')
    parser.add_argument('samples',type=int, help='# of samples')
    parser.add_argument('features', type=int, help='# of features')
    parser.add_argument('columns', type=int, help='# of columns')
    parser.add_argument('--epoch', type=int, help='Number of epochs', default=100)
    parser.add_argument('--alpha', type=float, help='Learning rate', default=.01)
    parser.add_argument('--gtype', type=str, help='Stochastic or Batch gradient. Type <s> or <b>', default='s')
    return parser.parse_args()


def gradient_stochastic(w, x, y, j, i) -> float:
    """
    Calculates the derivitive of the Cost function (Stochastic Gradient)\n
    Args:
        w (ndarray): parameter column vector
        x (ndarray): feature matrix
        y (ndarray): label vector
        j (int): feature index
        i (int): sample index
    Returns: float: Stochastic Gradient of the cost function 
    """
    xi = np.insert(x[i], 0, 1)
    h = np.dot(w.T, xi)
    return np.dot((h - y[i]), x[i][j])

def gradient_batch(w, x, y, m, j) -> float:
    """
    Calculates the derrivative of the cost function. 
    Loops through each example and calculates the gradient and returns the average.\n
    Args:
        w (ndarray): parameter column vector
        x (ndarray): feature matrix
        y (ndarray): label vector
        m (int): sample count
        j (int): feature index
    Returns: float: batch gradient
    """
    grad = 0
    for i in range(m):
        grad += (lrg(w, x[i]) - y[i]) * x[i][j]
    return grad * (1/m)

def cost_batch(w, x, y, m) -> float:
    """
    Cost/Loss function for the batch gradient\n
    Args:
        w (ndarray): parameter column vector
        x (ndarray): feature matrix
        y (ndarray): label vector
        m (int): sample count
    Returns: float: cost of the batch gradient
    """
    cost = 0
    for i in range(m):
        cost += (lrg(w, x[i]) - y[i])**2
    return cost / (2 * m)

def cost_stochastic(w, x, y, i) -> float:
    """
    Cost/Loss function for the stochastic gradient.\n
    Args:
        w (ndarray): parameter column vector
        x (ndarray): feature matrix
        y (ndarray): label vector
        m (int): sample count
        j (int): feature index
    Returns: float: stochastic gradient
    """
    return ((lrg(w, x[i]) - y[i])**2)/2

def lrg(w, xi) -> float:
    """
    Calculates the linear regression value / prediction\n
    Args:
        w (ndarray): paramater vector
        xi (ndarray): feature vector
    Returns: float: prediction value
    """
    xi = np.insert(xi, 0, 1)
    return np.dot(w.T, xi)

class GradientDescent():
    """Calculates the gradient descent using stochastic and batch"""

    def __init__(self, filename, samples:int, features:int, columns:int, epochs:int, alpha, gtype):
        self.filename: str = filename # filename
        self.data = pd.read_csv(self.filename) # dataframe of data
        self.m: int = samples # num of saamples
        self.n: int = features # num of features
        self.columns: int = columns # num of column vectors 
        self.epochs: int = epochs # num of iterations
        self.alpha: int = alpha # learning rate
        self.l: int = 1 # Number of labels
        self.x = np.array(self.data.iloc[:,0:self.n]).reshape(self.m,self.n) # x vectors
        self.y = np.array(self.data.iloc[:,self.columns-1]).reshape(self.m, self.l) # y vectors
        self.w = np.zeros(self.n+1).reshape(self.n+1, 1) # parameter
        self.errors = [] # list of errors
        self.gtype = gtype # 's' for stochastic, 'b' for batch
        self.parameters = []
        self.best_params =  None

class StochasticGradient(GradientDescent):
    """Calculates Stocahstic Gradient"""

    def __init__(self, filename, samples:int, features:int, columns:int, epochs:int, alpha, gtype):
        super().__init__(filename, samples, features, columns, epochs, alpha, gtype)

    def sgd(self):
        """Calculates Stocahstic Gradient"""

        for epoch in range(self.epochs):

            # choose a random example, calculate its gradient and update parameters
            # Do this for the number of samples, for each feature in that sample
            for i in range(self.m):

                index = np.random.randint(self.m)

                for j in range(self.n):

                    gradient = gradient_stochastic(self.w, self.x, self.y, j, index)
                    self.w[j] = self.w[j] - self.alpha * gradient

            # calcuate error
            self.errors.append(cost_stochastic(self.w, self.x, self.y, index))
            self.parameters.append(self.w)

        self.best_params = self.parameters[self.errors.index(min(self.errors))]

class BatchGradient(GradientDescent):
    """Calculates Batch Gradient"""

    def __init__(self, filename, samples:int, features:int, columns:int, epochs:int, alpha, gtype):
        super().__init__(filename, samples, features, columns, epochs, alpha, gtype)

    def bgd(self):
        """Calculates Batch Gradient"""

        for epoch in range(self.epochs):
            
            # for each feature, calculate the gradient by looping through 
            # the entire sample size on each itteration
            for j in range(self.n):

                gradient = gradient_batch(self.w, self.x, self.y, self.m, j)
                self.w[j] = self.w[j] - self.alpha * (1/self.m) * gradient

            # calcualte the error
            self.errors.append(cost_batch(self.w, self.x, self.y, self.m))
            self.parameters.append(self.w)

        self.best_params = self.parameters[self.errors.index(min(self.errors))]