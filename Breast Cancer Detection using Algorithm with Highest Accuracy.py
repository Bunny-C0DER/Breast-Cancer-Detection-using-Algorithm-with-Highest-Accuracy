#importing libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot

#loading dataset

dataset = pd.read_csv('data.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#mapping 'Class' string values to numbers

dataset['diagnosis'] = dataset['diagnosis'].map({'B':0, 'M':1}).astype(int)
print(dataset.head)

#segregating dataset to X and Y

X = dataset.iloc[:, 2:32].values
X
Y = dataset.iloc[:,1].values
Y

#splitting dataset into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y, test_size = 0.25, random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#validating some ML algorithm by its accuracy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.01)))
models.append(('KNN', KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)))
models.append(('CART', DecisionTreeClassifier(criterion='entropy', max_depth=20, random_state=0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale', kernel='rbf', C=1.0 )))
models.append(('RFC', RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=2)))

results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f' % (name, cv_results.mean()))

pyplot.ylim(.900, .999)
pyplot.bar(names, res, color ='maroon', width = 0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()

#training and prediction using the algorithm with highest accuracy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model=LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))