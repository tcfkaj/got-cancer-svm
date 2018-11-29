import pandas as pd
import scipy as scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# classes = pd.read_csv("classes.csv")
# print(classes.head())
# print(classes['y'].head())
# print(len(classes['y']))
#
# fulldata = pd.read_csv("scaled.csv")
# print(fulldata.iloc[:,1:5].head())


classes = pd.read_csv("classes_SER.csv")
#fulldata = pd.read_csv(r"C:\Users\steven\Downloads\scaled.csv")
fulldata = pd.read_csv("SER_8_8_RGB.csv")
#fulldata = pd.read_csv("hmnist_8_8_L.csv")
list1 = ["Actual 1", "Actual 2"]
list2 = ["Predicted 1", "Predicted 2"]
X = fulldata.iloc()[:,1:].drop(['label'], axis=1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
class1, class2, class3= classes.keys()[2:]

# Benign vs. Others

model = SVC(kernel='rbf', gamma='auto')
cl = class3
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

randgrid = {'C': scipy.stats.expon(scale=100), 'gamma':
scipy.stats.expon(scale=.25),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}
grid = RandomizedSearchCV(model, randgrid, verbose=2, cv=5)

# param_grid = {'C':[1e-300, 3.6, 3.7, 3.8]}  # 'gamma':[]
# grid = GridSearchCV(model, param_grid, verbose=2)

grid.fit(X_train, y_train)

print("Best params:\n")
print(grid.best_params_)
print()
print("Best score:\n")
print(grid.best_score_)
print()
print("Best estimator:\n")
print(grid.best_estimator_)
print()
print("Results:\n")
print(grid.cv_results_)
print()

grid_predictions = grid.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2))
print(classification_report(y_test, grid_predictions))

# Melanoma vs. Not

model = SVC(kernel='rbf', gamma='auto')
cl = class2
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

randgrid = {'C': scipy.stats.expon(scale=100), 'gamma':
scipy.stats.expon(scale=.25),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}
grid = RandomizedSearchCV(model, randgrid, verbose=2, cv=5)

# param_grid = {'C':[1e-300, 3.6, 3.7, 3.8]}  # 'gamma':[]
# grid = GridSearchCV(model, param_grid, verbose=2)

grid.fit(X_train, y_train)

print("Best params:\n")
print(grid.best_params_)
print()
print("Best score:\n")
print(grid.best_score_)
print()
print("Best estimator:\n")
print(grid.best_estimator_)
print()
print("Results:\n")
print(grid.cv_results_)
print()

grid_predictions = grid.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2))
print(classification_report(y_test, grid_predictions))

# ab vs Not

model = SVC(kernel='rbf', gamma='auto')
cl = class1
pd.Categorical(classes[cl]).value_counts()
y = classes[cl]  # Grid search does not take categorical data, use df or array
y_train, y_test = train_test_split(y, test_size=0.2, random_state=1)

randgrid = {'C': scipy.stats.expon(scale=100), 'gamma':
scipy.stats.expon(scale=.25),
  'kernel': ['rbf'], 'class_weight':['balanced', None]}
grid = RandomizedSearchCV(model, randgrid, verbose=2, cv=5)

# param_grid = {'C':[1e-300, 3.6, 3.7, 3.8]}  # 'gamma':[]
# grid = GridSearchCV(model, param_grid, verbose=2)

grid.fit(X_train, y_train)

print("Best params:\n")
print(grid.best_params_)
print()
print("Best score:\n")
print(grid.best_score_)
print()
print("Best estimator:\n")
print(grid.best_estimator_)
print()
print("Results:\n")
print(grid.cv_results_)
print()

grid_predictions = grid.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, grid_predictions), list1, list2))
print(classification_report(y_test, grid_predictions))

