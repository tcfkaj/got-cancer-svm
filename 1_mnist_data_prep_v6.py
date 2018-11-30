""" Skin Cancer MNIST Project """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

classes = pd.read_csv("classes_3cl.csv")
#fulldata = pd.read_csv(r"C:\Users\steven\Downloads\scaled.csv")
fulldata = pd.read_csv("scaled_8_8_RGB.csv")
#fulldata = pd.read_csv("hmnist_8_8_L.csv")
list1 = ["Actual 1", "Actual 2"]
list2 = ["Predicted 1", "Predicted 2"]
X = fulldata.iloc()[:,1:].drop(['label'], axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
class1, class2, class3 = classes.keys()[2:]

'''================================================================'''
# SVM for 1.ab vs others
#model = SVC(gamma='auto', class_weight="balanced")
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class1
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[690, 695, 698, 700]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 1.ab
kernel: rbf
best: {'C': 65}
from: {'C':[64, 65, 66]}
get:          Predicted 1  Predicted 2
Actual 1          480            3
Actual 2           58         1944

              precision    recall  f1-score   support

          ab       0.89      0.99      0.94       483
         not       1.00      0.97      0.98      2002

   micro avg       0.98      0.98      0.98      2485
   macro avg       0.95      0.98      0.96      2485
weighted avg       0.98      0.98      0.98      2485

cl: 1.ab
kernel: poly, d=2
best: {'C': 695}
from: {'C':[690, 695, 698, 700]} 
get:          Predicted 1  Predicted 2
Actual 1          481            2
Actual 2           84         1918

              precision    recall  f1-score   support

          ab       0.85      1.00      0.92       483
         not       1.00      0.96      0.98      2002

   micro avg       0.97      0.97      0.97      2485
   macro avg       0.93      0.98      0.95      2485
weighted avg       0.97      0.97      0.97      2485

"""
# SVM for 2.mel vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class2
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[390, 400, 410, 420]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 2.mel
kernel: rbf
best:  {'C': 200}
from: {'C':[200, 210, 205]} 
get:          Predicted 1  Predicted 2
Actual 1          680            3
Actual 2          105         1697

              precision    recall  f1-score   support

         mel       0.87      1.00      0.93       683
         not       1.00      0.94      0.97      1802

   micro avg       0.96      0.96      0.96      2485
   macro avg       0.93      0.97      0.95      2485
weighted avg       0.96      0.96      0.96      2485

cl: 2.mel
kernel: poly, d=2
best:  {'C': 400}
from: {'C':[390, 400, 410, 420]}
get:          Predicted 1  Predicted 2
Actual 1          616           67
Actual 2          146         1656

              precision    recall  f1-score   support

         mel       0.81      0.90      0.85       683
         not       0.96      0.92      0.94      1802

   micro avg       0.91      0.91      0.91      2485
   macro avg       0.88      0.91      0.90      2485
weighted avg       0.92      0.91      0.92      2485

"""
# SVM for 3.ben vs others
#model = SVC(gamma='auto')
model = SVC(kernel='poly', degree=2, gamma='auto')
cl = class3
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

param_grid = {'C':[385, 390, 395, 398]}  # 'gamma':[]
grid = GridSearchCV(model, param_grid, verbose=2)
grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2)
print(classification_report(y_test, grid_predictions))

"""
cl: 3.ben
kernel: rbf
best:  {'C': 105}
from: {'C':[100, 105, 107, 110]}
get:          Predicted 1  Predicted 2
Actual 1         1143          176
Actual 2           22         1144

              precision    recall  f1-score   support

         ben       0.98      0.87      0.92      1319
         not       0.87      0.98      0.92      1166

   micro avg       0.92      0.92      0.92      2485
   macro avg       0.92      0.92      0.92      2485
weighted avg       0.93      0.92      0.92      2485

cl: 3.ben
kernel: poly, d=2
best:  {'C': 395}
from: {'C':[385, 390, 395, 398]} 
get:          Predicted 1  Predicted 2
Actual 1         1069          250
Actual 2           84         1082

              precision    recall  f1-score   support

         ben       0.93      0.81      0.86      1319
         not       0.81      0.93      0.87      1166

   micro avg       0.87      0.87      0.87      2485
   macro avg       0.87      0.87      0.87      2485
weighted avg       0.87      0.87      0.87      2485

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












