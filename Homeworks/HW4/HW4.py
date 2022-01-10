"""
Homework 4
COMP 379
Brian Nguyen
11/9/2021

----------

This program uses the following datasets:

    - wine.data: https://archive.ics.uci.edu/ml/datasets/wine 

References:

    - https://towardsdatascience.com/build-knn-from-scratch-python-7b714c47631a
    - https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
    - https://towardsdatascience.com/grid-search-in-python-from-scratch-hyperparameter-tuning-3cca8443727b
    - https://www.geeksforgeeks.org/grid-searching-from-scratch-using-python/

"""

# Import applicable modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from random import seed
from random import randrange
import warnings
warnings.filterwarnings("ignore")

########## Data Pre-processing ##########

print('\n\t----------  Dataset(s)  ----------\n')

wine_df = pd.read_csv('wine/wine.data', names = ['class', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'od280/od315 of diluted wines', 'proline'])
print('\'wine.data\':\n', wine_df)

y = wine_df.iloc[:, [0]].values
X = wine_df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scalar = StandardScaler()
scalar.fit(X_train)
X_train_std = scalar.transform(X_train)
X_test_std = scalar.transform(X_test)

train = np.concatenate((y_train, X_train_std), axis = 1)

########## k-fold Cross-validation ##########

class kFoldCV:

    def __init__(self):
        pass

    def crossValSplit(self, dataset, numFolds):
        dataSplit = []
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)

        for _ in range(numFolds):
            fold = []

            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))

            dataSplit.append(fold)

        return dataSplit
    
    def kFCVEvaluate(self, dataset, classifier, numFolds, *args):
        folds = self.crossValSplit(dataset, numFolds)
        scores = []

        for fold in folds:
            trainSet = list(folds)
            #trainSet.remove(fold)
            trainSet.pop(0)
            trainSet = sum(trainSet, [])
            testSet = []

            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)

            y_train_kFCV = [row[0] for row in trainSet]
            X_train_kFCV = [train[1:] for train in trainSet]

            classifier.fit(X_train_kFCV, y_train_kFCV)

            y_test_kFCV = [row[0] for row in testSet]
            X_test_kFCV = [test[1:] for test in testSet]

            pred = lr.predict(X_test_kFCV)

            accuracy = lr.score(X_test_kFCV, y_test_kFCV)
            scores.append(accuracy)

        print('k = ' + str(numFolds) + ':')
        print('\tAccuracies:', scores)
        print('\tMax:', max(scores))
        print('\tMean:', (sum(scores) / float(len(scores))))
        
########## Question 2 ##########

print('\n\t----------  Question 2  ----------\n')

seed(10)

lr = LogisticRegression()
kfcv = kFoldCV()

print('LR models w/ k-fold CV accuracies:')

kfcv.kFCVEvaluate(train, lr, 2)
kfcv.kFCVEvaluate(train, lr, 3)
kfcv.kFCVEvaluate(train, lr, 4)
kfcv.kFCVEvaluate(train, lr, 5)
kfcv.kFCVEvaluate(train, lr, 10)

########## Question 3 ##########

print('\n\t----------  Question 3  ----------\n')

c_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000]
penalties = ['l1', 'l2', 'elasticnet', 'none']

parameters = []
for c_value in c_values:
    for penalty in penalties:
         parameters.append((c_value, penalty))

accuracies = []
print('C-value and penalty parameter combination on training set accuracies:')
for k in range(len(parameters)):
    # Chose to use 'saga' solving algorithm because it is the only one that supports with all four penalty params
    lr = LogisticRegression(C = parameters[k][0], penalty = parameters[k][1], solver = 'saga', l1_ratio = 1)

    lr.fit(X_train_std, y_train)

    accuracy = lr.score(X_train_std, y_train)
    accuracies.append(accuracy)

    print(str(parameters[k]) + ':', accuracy)

print('\nGrid search CV on training set:')
print('\tCombinations:', len(accuracies))
max_index = accuracies.index(max(accuracies))
print('\tMax accuracy:', parameters[max_index], '=>', max(accuracies))
print('\tMean accuracy:', (sum(accuracies) / float(len(accuracies))))

########## Question 4 ##########

print('\n\t----------  Question 4  ----------\n')

# Theorhetically, training and testing on the training set will give 100% because it is essentially 
# memorizing the dataset, but training on the training set and testing on a training set that an 
# ML model has never seen before is not guaranteed to perform as well. This is an indicator of 
# whether or not the ML model is actually performing well or not with the holdout method.

lr = LogisticRegression(C = parameters[max_index][0], penalty = parameters[max_index][1], solver = 'saga', l1_ratio = 1)
lr.fit(X_train_std, y_train)
print('Grid search CV on testing set with parameter combination', str(parameters[max_index]) + ':')
accuracy = lr.score(X_test_std, y_test)
print('\tAccuracy:', accuracy)
pred = lr.predict(X_test_std)
f1 = f1_score(y_test, pred, average = 'macro')
print('\tMacro F1 score:', f1)

lr = LogisticRegression(C = parameters[max_index][0], penalty = parameters[max_index][1], solver = 'saga', l1_ratio = 1)
lr.fit(X_train_std, y_train)
print('\nGrid search CV on training set with parameter combination', str(parameters[max_index]) + ':')
accuracy = lr.score(X_test_std, y_test)
print('\tAccuracy:', accuracy)
pred = lr.predict(X_test_std)
f1 = f1_score(y_test, pred, average = 'macro')
print('\tMacro F1 score:', f1)

print()