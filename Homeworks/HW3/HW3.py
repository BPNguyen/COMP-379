"""
Homework 3
COMP 379
Brian Nguyen
10/18/2021

----------

This program uses the following datasets:

    - wine.data: https://archive.ics.uci.edu/ml/datasets/wine 

References:

    - https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2
    - https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ 

"""

# Import applicable modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

########## K-Nearest Neighbors Implementation ##########

# Manhattan -> p = 1
# Euclidean -> p = 2
def minkowski_distance(a, b, p = 2):

    dims = len(a)
    distance = 0

    for dim in range(dims):

        distance += abs(a[dim] - b[dim]) ** p

    distance = distance ** (1 / p)

    return distance

def knn_predict(X_train, X_test, y_train, y_test, k, p):

    predictions = []

    for test_point in X_test:

        distances = []

        for train_point in X_train:

            distance = minkowski_distance(test_point, train_point, p = p)
            distances.append(distance)

        dists_df = pd.DataFrame(data = distances, columns = ['dist'])
        
        nn_df = dists_df.sort_values(by = ['dist'], axis = 0)[:k]

        counter = Counter(y_train[nn_df.index])

        prediction = counter.most_common()[0][0]

        predictions.append(prediction)

    return predictions

########## Data Pre-processing ##########

print('\n\t----------  Dataset(s)  ----------\n')

wine_df = pd.read_csv('wine/wine.data', names = ['class', 'alcohol', 'malic acid', 'ash', 'alcalinity of ash', 'magnesium', 'total phenols', 'flavanoids', 'nonflavanoid phenols', 'proanthocyanins', 'color intensity', 'hue', 'od280/od315 of diluted wines', 'proline'])
print('\'wine.data\':\n', wine_df)

y = wine_df.iloc[:, 0].values
X = wine_df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

scalar = StandardScaler()
scalar.fit(X_train)
X_train_std = scalar.transform(X_train)
X_dev_std = scalar.transform(X_dev)
X_test_std = scalar.transform(X_test)

########## Question 1 ##########

print('\n\t----------  Question 1  ----------\n')

lr = LogisticRegression()
lr.fit(X_train_std, y_train)

print('Default SKLearn LR model (w/ C = 0.1) prediction metrics on \'X_dev_std\':')

dev_lr_accuracy = lr.score(X_dev_std, y_dev)
print('\tAccuracy:', dev_lr_accuracy)

# Calculated macro F1 score to accomodate for potential class imbalance, as class distribution varies (see wine.names for dist)
dev_lr_pred = lr.predict(X_dev_std)
dev_lr_f1_score = f1_score(y_dev, dev_lr_pred, average = 'macro')
print('\tMacro F1 score:', dev_lr_f1_score)

print('\nClassfication report of default SKLearn LR model (w/ C = 0.1) on \'X_dev_std\':\n', classification_report(y_dev, dev_lr_pred))
print('Confusion matrix of default SKLearn LR model (w/ C = 0.1) on \'X_dev_std\':\n', confusion_matrix(y_dev, dev_lr_pred))

########## Question 2 ##########

print('\n\t----------  Question 2  ----------\n')

print('Tweaking of LR model\'s C-values on \'X_dev_std\':')

c_values = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
for value in c_values:

    lr = LogisticRegression(C = value, random_state = 42)
    lr.fit(X_train_std, y_train)

    print('C =', value, '\b:')

    dev_lr_accuracy = lr.score(X_dev_std, y_dev)
    print('\tAccuracy:', dev_lr_accuracy)

    dev_lr_pred = lr.predict(X_dev_std)
    dev_lr_f1_score = f1_score(y_dev, dev_lr_pred, average = 'macro')
    print('\tMacro F1 score:', dev_lr_f1_score)

    if ((dev_lr_accuracy == 1) or (dev_lr_f1_score == 1)):

        print('\n', classification_report(y_dev, dev_lr_pred))
        print(confusion_matrix(y_dev, dev_lr_pred), '\n')

########## Question 3 ##########

print('\n\t----------  Question 3  ----------\n')

dev_knn_pred = knn_predict(X_train_std, X_dev_std, y_train, y_dev, k = 23, p = 2)
dev_knn_pred_np = np.array(dev_knn_pred)

print('KNN model (w/ k = 23) prediction metrics on \'X_dev_std\':')

dev_knn_accuracy = accuracy_score(y_dev, dev_knn_pred_np)
print('\tAccuracy:', dev_knn_accuracy)

dev_knn_f1_score = f1_score(y_dev, dev_knn_pred_np, average = 'macro')
print('\tMacro F1 score:', dev_knn_f1_score)

print('\nClassfication report of KNN model (w/ k = 23) on \'X_dev_std\':\n', classification_report(y_dev, dev_knn_pred_np))
print('Confusion matrix of KNN model (w/ k = 23) on \'X_dev_std\':\n', confusion_matrix(y_dev, dev_knn_pred_np))

print('\n* See figure for tweaking of KNN Model\'s k-values on \'X_dev_std\' *')

kkn_accuracies = []
for k in range(1, 125):

    dev_knn_pred = knn_predict(X_train_std, X_dev_std, y_train, y_dev, k, p = 2)
    kkn_accuracies.append(accuracy_score(y_dev, dev_knn_pred))

#print(kkn_accuracies.index(max(kkn_accuracies)), ':', max(kkn_accuracies))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 125), kkn_accuracies)
ax.set_xlabel('# of Nearest Neighbors (k)')
ax.set_ylabel('Accuracy (%)');
ax.set_title('Tweaking of KNN Model\'s k-values on \'X_dev_std\'')

########## Question 4 ##########

print('\n\t----------  Question 4  ----------\n')

print('Dummy Classifier (w/ various strategies) prediction accuracies on \'X_dev_std\':')

dc_strats = ['stratified', 'most_frequent', 'prior', 'uniform']
for s in dc_strats:

    dc = DummyClassifier(strategy = s, random_state = 42)
    dc.fit(X_train_std, y_train)

    print(s, '\b:')

    dev_dc_accuracy = dc.score(X_dev_std, y_dev)
    print('\tAccuracy:', dev_dc_accuracy)

    dev_dc_pred = dc.predict(X_dev_std)
    dev_dc_f1_score = f1_score(y_dev, dev_dc_pred, average = 'macro')
    print('\tMacro F1 score:', dev_dc_f1_score)

    print('\n', classification_report(y_dev, dev_dc_pred))
    print(confusion_matrix(y_dev, dev_dc_pred), '\n')

########## Question 5 ##########

print('\t----------  Question 5  ----------\n')

lr = LogisticRegression(C = 0.1, random_state = 42)
lr.fit(X_train_std, y_train)

print('LR model (w/ C = 0.1) prediction metrics on \'X_test_std\':')

test_lr_accuracy = lr.score(X_test_std, y_test)
print('\tAccuracy:', test_lr_accuracy)

test_lr_pred = lr.predict(X_test_std)
test_lr_f1_score = f1_score(y_test, test_lr_pred, average = 'macro')
print('\tMacro F1 score:', test_lr_f1_score)

test_knn_pred = knn_predict(X_train_std, X_test_std, y_train, y_test, k = 23, p = 2)
test_knn_pred_np = np.array(test_knn_pred)

print('\nKNN model (w/ k = 23) prediction metrics on \'X_test_std\':')

test_knn_accuracy = accuracy_score(y_test, test_knn_pred_np)
print('\tAccuracy:', test_knn_accuracy)

test_knn_f1_score = f1_score(y_test, test_knn_pred_np, average = 'macro')
print('\tMacro F1 score:', test_knn_f1_score)

print('\nDummy Classifier (w/ various strategies) prediction accuracies on \'X_test_std\':')
for s in dc_strats:

    dc = DummyClassifier(strategy = s, random_state = 42)
    dc.fit(X_train_std, y_train)

    print(s, '\b:')
    
    test_dc_accuracy = dc.score(X_test_std, y_test)
    print('\tAccuracy:', test_dc_accuracy)

    test_dc_pred = dc.predict(X_test_std)
    test_dc_f1_score = f1_score(y_test, test_dc_pred, average = 'macro')
    print('\tMacro F1 score:', test_dc_f1_score)

plt.tight_layout()
plt.show()
print()