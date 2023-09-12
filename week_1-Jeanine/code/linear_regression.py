import numpy as np

def lsq(X, y):
    """
    Least squares linear regression
    (fits a linear model with coefficient beta to minimize the residual sum of squares between the targets in the dataset)
    
    :param X: Input data matrix (=>input of the model)
    :param y: Target vector (=>output of the model)
    :return: Estimated regressor coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

def predict(beta,X):
    """
    Predict the output of the testing set
    
    """ 
    # add column of ones to the X testing set
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)
    
    #calculate the predicted values using the linear regression line
    y_predicted = np.dot(X,beta)

    return y_predicted


def MSE(y, y_predicted):
    """
    Calculation of the mean squared error(MSE)
    MSE = 1/n_samples * np.sum(y_actual - y_predicted) 
        = mean((y_actual - y_predicted)^2)
    """
    MSE = np.mean((y_predicted - y) ** 2)
    
    return MSE

