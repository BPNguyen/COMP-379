"""
Homework 2
COMP 379
Brian Nguyen
10/1/2021

----------

This program uses the following datasets:

    LinSep.csv 
    NonLinSep.csv
    train.csv

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from random import *
import warnings
warnings.filterwarnings("ignore")

########## Perceptron Implementation ##########

class Perceptron(object):

    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

########## Decision Boundary Plotting ##########

def plot_decision_regions(X, y, classifier, resolution = 0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1],
            alpha = 0.8, c = colors[idx],
            edgecolor = 'black',
            marker = markers[idx],
            label = cl)

########## Adaline Implementation ##########

class AdalineGD(object):
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, 0)

########## Baseline Model ##########

class Baseline(object):

    def __init__(self, weights, random_state = 1):

        rgen = np.random.RandomState(random_state)
        rand_factor = rgen.normal(loc = 0.0, scale = 0.01, size = len(weights))

        index = 0
        while (index != len(weights)):
            weights[index] = weights[index] * rand_factor[index]
            index = index + 1

        self.weights = weights

    def net_input(self, X):
        return np.dot(X, self.weights[:])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

########## Accuracy Calculation ##########

def calculateAccuracy(actualLabels, predictedLabels):

    predictionsCorrect = 0
    for value in range(actualLabels.size):
        if (actualLabels[value] == predictedLabels[value]):
            predictionsCorrect = predictionsCorrect + 1

    return ((predictionsCorrect / actualLabels.size) * 100)

########## Question 1 ##########

print('\n\t----------  Question 1  ----------\n')

ls_df = pd.read_csv('LinSep.csv')

ls_y = ls_df.iloc[0:ls_df.size, 2].values
ls_y = np.where(ls_y == 'Backcourt', 0, 1)

ls_X = ls_df.iloc[0:ls_df.size, [0, 1]].values

ls_X_train, ls_X_test, ls_y_train, ls_y_test = train_test_split(ls_X, ls_y, test_size = 0.30, random_state = 42)

plt.scatter(ls_X[:11, 0], ls_X[:11, 1],
    color = 'red', marker = 'o', label = 'Backcourt'
)
plt.scatter(ls_X[10:ls_df.size, 0], ls_X[10:ls_df.size, 1],
    color = 'blue', marker = 'x', label = 'Frontcourt'
)

plt.xlabel('Rebounds Per Game')
plt.ylabel('Height (in m)')
plt.title('LinSep Training Dataset')
plt.legend(title = 'Player plays...', loc = 'upper left')
plt.show()

ls_ppn = Perceptron(eta = 0.1, n_iter = 10)
ls_ppn.fit(ls_X_train, ls_y_train)

plt.plot(range(1, len(ls_ppn.errors_) + 1), ls_ppn.errors_, marker = 'o')
plt.title('Perception Convergence on LinSep Training Dataset')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.show()

plot_decision_regions(ls_X_test, ls_y_test, classifier = ls_ppn)
plt.xlabel('Rebounds Per Game')
plt.ylabel('Height (in m)')
plt.title('Perceptron Predictions w/ Decision Regions on LinSep Testing Dataset')
plt.legend(title = 'Player plays...', loc = 'upper left')
plt.show()

ls_y_pred = ls_ppn.predict(ls_X_test)
print('Actual labels of LinSep testing dataset:', ls_y_test)
print('Predicted labels by LinSep Perceptron:', ls_y_pred)
ls_accuracy = calculateAccuracy(ls_y_test, ls_y_pred)
print('LinSep Perceptron accuracy %age:', ls_accuracy)

########## Question 2 ##########

print('\n\t----------  Question 2  ----------\n')

nls_df = pd.read_csv('NonLinSep.csv')

nls_y = nls_df.iloc[0:nls_df.size, 2].values
nls_y = np.where(nls_y == 'Backcourt', 0, 1)

nls_X = nls_df.iloc[0:nls_df.size, [0, 1]].values

nls_X_train, nls_X_test, nls_y_train, nls_y_test = train_test_split(nls_X, nls_y, test_size = 0.30, random_state = 42)

plt.scatter(nls_X[:11, 0], nls_X[:11, 1],
    color = 'red', marker = 'o', label = 'Backcourt'
)
plt.scatter(nls_X[10:nls_df.size, 0], nls_X[10:nls_df.size, 1],
    color = 'blue', marker = 'x', label = 'Frontcourt'
)

plt.xlabel('Rebounds Per Game')
plt.ylabel('Height (in m)')
plt.title('NonLinSep Training Dataset')
plt.legend(title = 'Player plays...', loc = 'upper left')
plt.show()

nls_ppn = Perceptron(eta = 0.1, n_iter = 10)
nls_ppn.fit(nls_X_train, nls_y_train)

plt.plot(range(1, len(nls_ppn.errors_) + 1), nls_ppn.errors_, marker = 'o')
plt.title('Perception Convergence on NonLinSep Training Dataset')
plt.xlabel('Epochs')
plt.ylabel('Number of Updates')
plt.show()

plot_decision_regions(nls_X_test, nls_y_test, classifier = nls_ppn)
plt.xlabel('Rebounds Per Game')
plt.ylabel('Height (in m)')
plt.title('Perceptron Predictions w/ Decision Regions on NonLinSep Testing Dataset')
plt.legend(title = 'Player plays...', loc = 'upper left')
plt.show()

nls_y_pred = nls_ppn.predict(nls_X_test)
print('Actual labels of NonLinSep testing dataset:', nls_y_test)
print('Predicted labels by NonLinSep Perceptron:', nls_y_pred)
nls_accuracy = calculateAccuracy(nls_y_test, nls_y_pred)
print('NonLinSep Perceptron accuracy %age:', nls_accuracy)

########## Question 3 ##########

print('\n\t----------  Question 3  ----------\n')

titanic_df = pd.read_csv('train.csv')

titanic_df = titanic_df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])
titanic_df = titanic_df.dropna(axis = 0, how = 'any')
titanic_df['Sex'] = titanic_df['Sex'].replace(to_replace = 'male', value = 1)
titanic_df['Sex'] = titanic_df['Sex'].replace(to_replace = 'female', value = 0)

titanic_y = titanic_df.iloc[0:titanic_df.size, 0].values

titanic_X = titanic_df.iloc[0:titanic_df.size, [1, 2, 3, 4, 5, 6]].values

titanic_X_train, titanic_X_test, titanic_y_train, titanic_y_test = train_test_split(titanic_X, titanic_y, test_size = 0.30, random_state = 42)

titanic_ada = AdalineGD(eta = 0.0000001, n_iter = 15, random_state = 8)
titanic_ada.fit(titanic_X_train, titanic_y_train)

plt.plot(range(1, len(titanic_ada.cost_) + 1), titanic_ada.cost_, marker = 'o')
plt.title('Adaline Cost Minimization on Titanic Training Dataset\nLearning rate = 0.0000001')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()

titanic_y_pred = titanic_ada.predict(titanic_X_test)
print('Actual labels of Titanic testing dataset:\n', titanic_y_test)
print('\nPredicted labels by Titanic Adaline random_state seed 8:\n', titanic_y_pred)
titanic_accuracy = calculateAccuracy(titanic_y_test, titanic_y_pred)
print('\nTitanic Adaline random_state seed 8 accuracy %age:', titanic_accuracy)

print()

print('Random_state seed accuracy %ages:')
key = True
lr = 0.0000001
seed = 0
while (key):
    ada = AdalineGD(eta = lr, n_iter = 15, random_state = seed)
    ada.fit(titanic_X_train, titanic_y_train)
    y_pred = ada.predict(titanic_X_test)
    accuracy = calculateAccuracy(titanic_y_test, y_pred)
    print('\t' + str(seed) + ':\t', accuracy)

    if seed != 42:
        seed = seed + 1
    else: 
        key = False

########## Question 4 ##########

print('\n\t----------  Question 4  ----------\n')

print('Min of J(w) in Titanic Adaline:', min(titanic_ada.cost_))
print('Heaviest weight in Titanic Adaline =', max(titanic_ada.w_))
print('Lightest weight in Titanic Adaline =', min(titanic_ada.w_))
print('Titanic Adaline Weights:\n\tZero-weight,', titanic_df.columns[1:].values, '\n\t\b', titanic_ada.w_)

########## Question 5 ##########

print('\n\t----------  Question 5  ----------\n')

ls_bl = Baseline(ls_ppn.w_[1:])
ls_bl_y_pred = ls_bl.predict(ls_X_test)
print('Actual labels of LinSep testing dataset:', ls_y_test)
print('Predicted labels by LinSep Perceptron:', ls_y_pred)
print('\tLinSep Perceptron accuracy %age:', ls_accuracy)
print('Predicted labels of LinSep Baseline:', ls_bl_y_pred)
ls_bl_accuracy = calculateAccuracy(ls_y_test, ls_bl_y_pred)
print('\tLinSep Baseline accuracy %age:', ls_bl_accuracy)

print()

nls_bl = Baseline(nls_ppn.w_[1:])
nls_bl_y_pred = nls_bl.predict(nls_X_test)
print('Actual labels of NonLinSep testing dataset:', nls_y_test)
print('Predicted labels by LinSep Perceptron:', ls_y_pred)
print('\tNonLinSep Perceptron accuracy %age:', nls_accuracy)
print('Predicted labels of NonLinSep Baseline:', nls_bl_y_pred)
nls_bl_accuracy = calculateAccuracy(nls_y_test, nls_bl_y_pred)
print('\tNonLinSep Baseline accuracy %age:', nls_bl_accuracy)

print()

titanic_bl = Baseline(titanic_ada.w_[1:])
titanic_bl_y_pred = titanic_bl.predict(titanic_X_test)
print('Actual labels of Titanic testing dataset:\n', titanic_y_test)
print('\nPredicted labels by Titanic Adaline random_state seed 8:\n', titanic_y_pred)
print('\n\tTitanic Adaline random_state seed 8 accuracy %age:', titanic_accuracy)
print('\nPredicted labels by Titanic Baseline:\n', titanic_bl_y_pred)
titanic_bl_accuracy = calculateAccuracy(titanic_y_test, titanic_bl_y_pred)
print('\n\tTitanic Baseline accuracy %age:', titanic_bl_accuracy)

print()