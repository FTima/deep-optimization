import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io
import math
import sklearn
import sklearn.datasets
from opt_utils import *
from test_cases import *
from opt_methods import *

def main():
    train_X, train_Y = load_dataset()
    
    
    # # train 3-layer model
    # layers_dims = [train_X.shape[0], 5, 2, 1]
    # parameters = model(trainparameters_X, train_Y, layers_dims, optimizer = "gd")
    # # Predict
    # predictions = predict(train_X, train_Y, parameters)
    # # Plot decision boundary
    # plt.title("Model with Gradient Descent optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5,2.5])
    # axes.set_ylim([-1,1.5])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    # # train 3-layer model
    # layers_dims = [train_X.shape[0], 5, 2, 1]
    # parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")
    # # Predict
    # predictions = predict(train_X, train_Y, parameters)
    # # Plot decision boundary
    # plt.title("Model with Momentum optimization")
    # axes = plt.gca()
    # axes.set_xlim([-1.5,2.5])
    # axes.set_ylim([-1,1.5])
    # plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


    # train 3-layer model
    layers_dims = [train_X.shape[0], 5, 2, 1]
    parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")
    # Predict
    predictions = predict(train_X, train_Y, parameters)
    # Plot decision boundary
    plt.title("Model with Adam optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


if __name__ == "__main__":
    main()