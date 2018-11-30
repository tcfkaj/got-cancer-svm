""" Skin Cancer MNIST Project """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

classes = pd.read_csv("classes.csv")
#fulldata = pd.read_csv(r"C:\Users\steven\Downloads\scaled.csv")
fulldata = pd.read_csv("scaled_88L.csv")
#fulldata = pd.read_csv("hmnist_8_8_L.csv")
list1 = ["Actual 1", "Actual 2"]
list2 = ["Predicted 1", "Predicted 2"]
X = fulldata.iloc()[:,1:].drop(['label'], axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
class1, class2, class3, class4 = classes.keys()[2:]

'''================================================================'''
# SVM for 1.akiec vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class1
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[1e-300, 3, 100]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid_train = grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
model.support_vectors_

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 1.akiec
kernel: rbf
best: {'C': 1e-300}
from: {'C':[1e-300, 1e-100, 3, 100]}
get:          Predicted 1  Predicted 2
Actual 1            0           61
Actual 2            0         1942

cl: 1.akiec
kernel: poly, d=2
best: {'C': 1e-300}
from: {'C':[1e-300, 1e-100, 3, 100]}
get:          Predicted 1  Predicted 2
Actual 1            0           61
Actual 2            0         1942

"""
# SVM for 2.bcc vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class2
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[1e-300, 1, 3, 3.5]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 2.bcc
kernel: rbf
best:  {'C': 3}
from: {'C':[1e-300, 1, 3, 3.5]}
get:          Predicted 1  Predicted 2
Actual 1            0           96
Actual 2            0         1907

cl: 2.bcc
kernel: poly, d=2
best:  {'C': 1e-300}
from: {'C':[1e-300, 1e-10, 1e-5, 3, 100]}
get:          Predicted 1  Predicted 2
Actual 1            0           96
Actual 2            0         1907

"""
# SVM for 3.mel vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class3
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[1e-300, 3.6, 3.7, 3.8]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 3.mel
kernel: rbf
best:  {'C': 3.6}
from: {'C':[1e-300, 2, 2.5, 3, 3.5, 3.6]}
get:          Predicted 1  Predicted 2
Actual 1           17          205
Actual 2            7         1774

cl: 3.mel
kernel: poly, d=2
best:  {'C': 3.6}
from: {'C':[1e-300, 3.5, 3.6, 3.7]}
get:          Predicted 1  Predicted 2
Actual 1            7          215
Actual 2            7         1774

"""

# SVM for 4.ben vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class4
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[1e-300, 3, 3.5, 4]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 4.ben
kernel: rbf
best:  {'C': 3}
from: {'C':[1e-300, 2, 3, 3.5]}
get:          Predicted 1  Predicted 2
Actual 1         1589           35
Actual 2          318           61

cl: 4.ben
kernel: poly, d=2
best:  {'C': 3.5}
from: {'C':[1e-300, 3, 3.5, 4]}
get:          Predicted 1  Predicted 2
Actual 1         1612           12
Actual 2          363           16


"""


'''================================================================'''






'''SKIP BELOW'''
skin = pd.read_csv("hmnist_8_8_L.csv")
#skin.describe()

# Creating 3 classes: 1:"nv"="melanocytic nevi" 67%, 2:"mel"="melanoma" 11% and 3:"Other" 22%
for i in range(len(skin['label'])):
    if skin['label'][i] not in [4, 6]:
        skin['label'][i] = 3
    elif skin['label'][i] == 4:
        skin['label'][i] = 1
    else:
        skin['label'][i] = 2

skin['label'] = pd.Categorical(skin['label'])
#skin['label'].value_counts()
#skin.info()

# Split train/test
from sklearn.model_selection import train_test_split
#y = skin['label']
y = np.array(skin['label'])  # Grid search does not take categorical, use array
X = skin.drop(['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SVM modeling
from sklearn.svm import SVC
model = SVC()
#model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

list1 = ["Actual 1", "Actual 2", "Actual 3"]
list2 = ["Predicted 1", "Predicted 2", "Predicted 3"]

pd.DataFrame(confusion_matrix(y_test, predictions), list1, list2)
print(classification_report(y_test, predictions))

from sklearn.model_selection import GridSearchCV
param_grid = {'gamma':[0.000001, 0.00001, 0.0001], 'C':[8, 10, 12]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
best: {'C': 1, 'gamma': 1e-05}
from: {'gamma':[0.000001, 0.00001, 0.0001, 0.001], 'C':[0.0001, 0.001, 0.01, 1]}
get:          Predicted 1  Predicted 2  Predicted 3
Actual 1         1933            8           62
Actual 2          291           17           49
Actual 3          409            1          235


best: {'C': 4, 'gamma': 1e-05}
from: {'gamma':[0.000001, 0.00001, 0.0001], 'C':[1, 2, 4]}
get:          Predicted 1  Predicted 2  Predicted 3
Actual 1         1898           17           88
Actual 2          260           43           54
Actual 3          346           14          285


best: {'C': 10, 'gamma': 1e-05}
from: {'gamma':[0.000001, 0.00001, 0.0001], 'C':[5, 10, 15]}
get:          Predicted 1  Predicted 2  Predicted 3
Actual 1         1869           31          103
Actual 2          243           60           54
Actual 3          330           25          290

"""



"""
precision
Precision is the ability of a classiifer not to label an instance positive that is actually negative. For each class it is defined as as the ratio of true positives to the sum of true and false positives. Said another way, “for all instances classified positive, what percent was correct?”
recall
Recall is the ability of a classifier to find all positive instances. For each class it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, “for all instances that were actually positive, what percent was classified correctly?”
f1 score
The F1 score is a weighted harmonic mean of precision and recall such that the best score is 1.0 and the worst is 0.0. Generally speaking, F1 scores are lower than accuracy measures as they embed precision and recall into their computation. As a rule of thumb, the weighted average of F1 should be used to compare classifier models, not global accuracy.
support
Support is the number of actual occurrences of the class in the specified dataset. Imbalanced support in the training data may indicate structural weaknesses in the reported scores of the classifier and could indicate the need for stratified sampling or rebalancing. Support doesn’t change between models but instead diagnoses the evaluation process.
"""




"""
1.akiec, 2.bk1, 3.df, 4.nv, 5.vasc, 6.mel
"""




metadata = pd.read_csv("HAM10000_metadata.csv")
metadata.describe()
metadata.info()

skin.ix()[1,1]
metadata.iloc()[:,2]
# Merging Response variable "dx"="pigmented lesions" with skin images, by labels
full_data = pd.merge(pd.DataFrame(metadata.iloc()[:,2]), skin, left_index=True, right_index=True)
full_data.head()

cato=pd.Series(full_data.iloc()[:,0], dtype="category")
cato.value_counts()
cato2=pd.Series(full_data.iloc()[:,-1], dtype="category")
cato2.value_counts()














