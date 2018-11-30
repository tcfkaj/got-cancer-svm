import pandas as pd
import scipy as scipy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import time as time
import pickle

classes = pd.read_csv("data/classes_3cl.csv")
fulldata = pd.read_csv("data/scaled_8_8_RGB.csv")
list1 = ["Actual 1", "Actual 2"]
list2 = ["Predicted 1", "Predicted 2"]
list3 = ["Actual 1", "Actual 2", "Actual 3"]
list4 = ["Predicted 1", "Predicted 2", "Predicted 3"]
X = fulldata.iloc()[:,1:].drop(['label'], axis=1)
classes = classes.drop(classes.columns[0], axis=1)

# C_mel = 70.6780
# C_ben = 18.3646
# C_ab = 163.3178
model = SVC(kernel='poly', degree=3, gamma='auto', class_weight='balanced')

ben_scores  = []
mel_scores = []
ab_scores = []
multi_scores = []
ben_rats  = []
mel_rats = []
ab_rats = []
multi_rats = []


start_time = time.time()

## N-Fold Cross-Validation
n = 5
for i in range(1,n+1):

    print('Fold ', i, '... \n')
    print('........ \n')
    print('........ \n')
    print('........ \n')
    print()
    print()
    print()

    X_train, X_test, cl_train, cl_test = train_test_split(X, classes,
    test_size=1/n, random_state=i)
    # print(X_train.iloc[:,1:5].head())
    # print(cl_train.head())

    # Inflate training set
    # index = cl_train.index[cl_train['y'] != 'ben'].tolist()
    # cl_rep, X_rep = cl_train.loc[index,:], X_train.loc[index,:]
    # cl_train = pd.concat([cl_train, pd.concat([cl_rep]*3)])
    # X_train = pd.concat([X_train, pd.concat([X_rep]*3)])

    # Ben vs rest
    print('Benign vs rest... \n')
    y_train, y_test = cl_train['ben'], cl_test['ben']
    model.fit(X_train, y_train)
    nsv = sum(model.n_support_)
    ltr = len(X_train)
    print('Number of support vectors: ', nsv)
    print('Size of training set: ', ltr)
    print('Ratio: ', nsv/ltr)
    print(model.dual_coef_)
    ben_coef = model.dual_coef_

    model_pred = model.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, model_pred), list1, list2))
    print(classification_report(y_test, model_pred))
    ben_scores.append(model.score(X_test, y_test))
    ben_rats.append(nsv/ltr)

    if i == 1:
        ben_conf = confusion_matrix(y_test, model_pred)
    else:
        ben_conf += confusion_matrix(y_test, model_pred)


    # Mel vs rest
    print('Mel vs rest... \n')
    y_train, y_test = cl_train['mel'], cl_test['mel']
    model.fit(X_train, y_train)
    nsv = sum(model.n_support_)
    ltr = len(X_train)
    print('Number of support vectors: ', nsv)
    print('Size of training set: ', ltr)
    print('Ratio: ', nsv/ltr)
    print(model.dual_coef_)
    mel_coef = model.dual_coef_

    model_pred = model.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, model_pred), list1, list2))
    print(classification_report(y_test, model_pred))
    mel_scores.append(model.score(X_test, y_test))
    mel_rats.append(nsv/ltr)

    if i == 1:
        mel_conf = confusion_matrix(y_test, model_pred)
    else:
        mel_conf += confusion_matrix(y_test, model_pred)

    # AB vs rest
    print('AB vs rest... \n')
    y_train, y_test = cl_train['ab'], cl_test['ab']
    model.fit(X_train, y_train)
    nsv = sum(model.n_support_)
    ltr = len(X_train)
    print('Number of support vectors: ', nsv)
    print('Size of training set: ', ltr)
    print('Ratio: ', nsv/ltr)
    print(model.dual_coef_)
    ab_coef = model.dual_coef_

    model_pred = model.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, model_pred), list1, list2))
    print(classification_report(y_test, model_pred))
    ab_scores.append(model.score(X_test, y_test))
    ab_rats.append(nsv/ltr)

    if i == 1:
        ab_conf = confusion_matrix(y_test, model_pred)
    else:
        ab_conf += confusion_matrix(y_test, model_pred)



    # Save alphas
    filename = "data/alphas_run_" + str(i)
    with open(filename, 'wb') as f:
        pickle.dump(ben_coef, f)
        pickle.dump(mel_coef, f)
        pickle.dump(ab_coef, f)

    # 3 class
    print('Multiclass RBF... \n')
    y_train, y_test = cl_train['y'], cl_test['y']
    model.fit(X_train, y_train)
    nsv = sum(model.n_support_)
    ltr = len(X_train)
    print('Number of support vectors: ', nsv)
    print('Size of training set: ', ltr)
    print('Ratio: ', nsv/ltr)

    model_pred = model.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, model_pred), list3, list4))
    print(classification_report(y_test, model_pred))
    multi_scores.append(model.score(X_test, y_test))
    multi_rats.append(nsv/ltr)

    if i == 1:
        multi_conf = confusion_matrix(y_test, model_pred)
    else:
        multi_conf += confusion_matrix(y_test, model_pred)



end_time = time.time()

## Scores

ben_score = sum(ben_scores)/n
mel_score = sum(mel_scores)/n
ab_score = sum(ab_scores)/n
multi_score = sum(multi_scores)/n

print()
print('CV scores: \n')
print()
print('Ben vs not: \n')
print(ben_score)
print()
print(ben_scores)
print()
print(ben_rats)
print()
print(ben_conf)
print()
print('Mel vs not: \n')
print(mel_score)
print()
print(mel_scores)
print()
print(mel_rats)
print()
print(mel_conf)
print()
print('AB vs not: \n')
print(ab_score)
print()
print(ab_scores)
print()
print(ab_rats)
print()
print(ab_conf)
print()
print('Multiclass: \n')
print(multi_score)
print()
print(multi_scores)
print()
print(multi_rats)
print()
print(multi_conf)
print()
print()
run_time = end_time - start_time
print('Finished in  ', run_time, ' seconds')
