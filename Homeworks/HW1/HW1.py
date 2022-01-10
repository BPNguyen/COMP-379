"""
Homework 1
COMP 379
Brian Nguyen
9/13/2021

----------

This program uses the following dataset:

    https://www.kaggle.com/c/titanic/data 

"""

import pandas as pd

def algorithm(data, type):
    
    if (type == 'train'):
        algo_df = pd.DataFrame(columns = ['PassengerId', 'Prediction', 'Actual', 'WasIncorrect'])
        algo_df.PassengerId = data.PassengerId
        algo_df.Actual = data.Survived
    elif (type == 'test'):
        algo_df = pd.DataFrame(columns = ['PassengerId', 'Prediction'])
        algo_df.PassengerId = data.PassengerId

    MAX_SCORE = 10
    THRESHOLD = MAX_SCORE / 2
    for index, row in data.iterrows():

        score = 0
        if (row.Sex == 'female'):
            score += 4

        if (row.Pclass == 1):
            score += 3
        elif (row.Pclass == 2):
            score += 2

        if (row.Age <= 18):
            score += 1

        if (row.SibSp >= 1):
            score += 1
        
        if (row.Parch >= 1):
            score += 1

        if (score >= THRESHOLD):
            algo_df.iloc[index, 1] = 1
        else:
            algo_df.iloc[index, 1] = 0

    if (type == 'train'):
        algo_df.WasIncorrect = abs(algo_df.Prediction - algo_df.Actual)
        print(algo_df)

        accuracy = (1 - (sum(algo_df.WasIncorrect) / len(algo_df))) * 100
        print('\n[train.csv] Algorithm accuracy %age:', accuracy)
    elif (type == 'test'):
        print(algo_df)

        accuracy = ((sum(algo_df.Prediction) / len(algo_df))) * 100
        print('\n[test.csv] Predicted %age of passengers survived:', accuracy)

# NOTE: Filepath to 'train.csv' may need to be updated to reflect actual location
train_data = pd.read_csv('train.csv')

# NOTE: Filepath to 'test.csv' may need to be updated to reflect actual location
test_data = pd.read_csv('test.csv')

print('\n\t----------  train.csv  ----------\n')
algorithm(train_data, 'train')

print('\n\t----------  test.csv  ----------\n')
algorithm(test_data, 'test')

print('\n\t----------  Assumption Value Survivability Rates  ----------\n')

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print('[Sex] Female:', rate_women)

pclass1 = train_data.loc[train_data.Pclass == 1]['Survived']
rate_pclass1 = sum(pclass1)/len(pclass1)
print('[Pclass] Ticket class 1/Upper socio-economic status:', rate_pclass1)

pclass2 = train_data.loc[train_data.Pclass == 2]['Survived']
rate_pclass2 = sum(pclass2)/len(pclass2)
print('[Pclass] Ticket class 2/Middle socio-economic status:', rate_pclass2)

pclass3 = train_data.loc[train_data.Pclass == 3]['Survived']
rate_pclass3 = sum(pclass3)/len(pclass3)
print('[Pclass] Ticket class 3/Lower socio-economic status:', rate_pclass3)

age = train_data.loc[train_data.Age <= 18]['Survived']
rate_age = sum(age)/len(age)
print('[Age] <= 18', rate_age)

sibsp = train_data.loc[train_data.SibSp >= 1]['Survived']
rate_sibsp = sum(sibsp)/len(sibsp)
print('[SibSp] >= 1:', rate_sibsp)

parch = train_data.loc[train_data.Parch >= 1]['Survived']
rate_parch = sum(parch)/len(parch)
print('[Parch] >= 1:', rate_parch)
print()