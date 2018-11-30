""" Histograms, means and std dev, alpha ranking"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

classes = pd.read_csv("classes_3cl.csv")
#fulldata = pd.read_csv(r"C:\Users\steven\Downloads\scaled.csv")
fulldata = pd.read_csv("scaled_8_8_RGB.csv")
#fulldata = pd.read_csv("hmnist_8_8_L.csv")

X = fulldata.iloc()[:,1:].drop(['label'], axis=1)
class1, class2, class3 = classes.keys()[2:]

#pd.Categorical(classes[class1]).value_counts()
#pd.Categorical(classes[class2]).value_counts()
#pd.Categorical(classes[class3]).value_counts()

# X for each class
X_cl1 = X.iloc[np.where(classes[class1]!="not")[0],:]
X_cl2 = X.iloc[np.where(classes[class2]!="not")[0],:]
X_cl3 = X.iloc[np.where(classes[class3]!="not")[0],:]

# Boxplots of each descriptor in all classes
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    plt.boxplot([aaa, bbb, ccc], 0, '')
    plt.title("Boxplots of descriptor " + str(i) + " for all classes")
    plt.xticks(range(1,4), classes.keys()[2:])
    #plt.savefig("var" + str(i) + "_boxplot.png")
    plt.show()
    
# Histogram comparison 
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    plt.hist([aaa, bbb, ccc])
    plt.title("Histogram comparison of descriptor " + str(i) + " for all classes")
    plt.legend(classes.keys()[2:])
    plt.ylabel('Frequency')
    #plt.savefig("var" + str(i) + "_hist_compare.png")
    plt.show()

# Bar chart of means and std dev for each descriptor in all classes
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    plt.bar(range(3), [np.mean(aaa), np.mean(bbb), np.mean(ccc)])
    plt.title("Mean comparison of descriptor " + str(i) + " for all classes")
    plt.xticks(range(0,4), classes.keys()[2:])
    plt.ylabel('Mean')
    #plt.savefig("var" + str(i) + "_mean_bar.png")
    plt.show()
    
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    plt.bar(range(3), [np.std(aaa), np.std(bbb), np.std(ccc)])
    plt.title("Std dev comparison of descriptor " + str(i) + " for all classes")
    plt.xticks(range(0,4), classes.keys()[2:])
    plt.ylabel('Standard Dev')
    #plt.savefig("var" + str(i) + "_std_bar.png")
    plt.show()

# Saving mean and std data
df_mean = pd.DataFrame()
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    row1 = pd.DataFrame([np.mean(aaa), np.mean(bbb), np.mean(ccc)])
    df_mean = pd.concat([df_mean, row1], axis=1)
df_mean.columns = np.array(X.keys()[range(10)])

df_std = pd.DataFrame()
for i in range(10):
    aaa=X_cl1.iloc[:,i]
    bbb=X_cl2.iloc[:,i]
    ccc=X_cl3.iloc[:,i]
    row1 = pd.DataFrame([np.std(aaa), np.std(bbb), np.std(ccc)])
    df_std = pd.concat([df_std, row1], axis=1)
df_std.columns = np.array(X.keys()[range(10)])

#df_mean.to_csv("mean_3cl_10var.csv")
#df_std.to_csv("std_3cl_10var.csv")

'''================================================================'''
cl = class3  # class1 ~ 3
'''================================================================'''

# 3. Histograms for each descriptor in each class
X_cl = X.iloc[np.where(classes[cl]!="not")[0],:]
for i in range(10):
    plt.hist(X_cl.iloc[:,i], edgecolor="k")
    plt.title("Descriptor " + str(X_cl.keys()[i]) + ' in class ' + cl)
    plt.ylabel('Frequency')
    #plt.savefig("var" + str(i) + "_" + cl + ".png")
    plt.show()

'''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'''
# Plotting reordered alpha associated to the support vectors
import pickle
for i in range(1,6):
    filename = "data/RBF_alphas/alphas_run_" + str(i)
    with open(filename, 'rb') as f:
        ben_coef = pickle.load(f)
        mel_coef = pickle.load(f)
        ab_coef = pickle.load(f)
        
        ben_coef.sort()
        mel_coef.sort()
        ab_coef.sort()
        
        plt.plot(ben_coef[0])
        plt.plot(mel_coef[0])
        plt.plot(ab_coef[0])
        plt.legend(["ben", "mel", "ab"])
        plt.title("Ranked alpha for the support vectors in 3 classes (rbf fold #" + str(i) + ")")
        plt.ylabel('Alpha')
        plt.xlabel('Support Vectors')
        #plt.savefig("alpha_rbf_fold" + str(i) + ".png")
        plt.show()

for i in range(1,6):
    filename = "data/poly degree2/alphas_run_" + str(i)
    with open(filename, 'rb') as f:
        ben_coef = pickle.load(f)
        mel_coef = pickle.load(f)
        ab_coef = pickle.load(f)
        
        ben_coef.sort()
        mel_coef.sort()
        ab_coef.sort()
        
        plt.plot(ben_coef[0])
        plt.plot(mel_coef[0])
        plt.plot(ab_coef[0])
        plt.legend(["ben", "mel", "ab"])
        plt.title("Ranked alpha for the support vectors in 3 classes (poly2 fold #" + str(i) + ")")
        plt.ylabel('Alpha')
        plt.xlabel('Support Vectors')
        #plt.savefig("alpha_poly2_fold" + str(i) + ".png")
        plt.show()
        
for i in range(1,6):
    filename = "data/poly degree3/alphas_run_" + str(i)
    with open(filename, 'rb') as f:
        ben_coef = pickle.load(f)
        mel_coef = pickle.load(f)
        ab_coef = pickle.load(f)
        
        ben_coef.sort()
        mel_coef.sort()
        ab_coef.sort()
        
        plt.plot(ben_coef[0])
        plt.plot(mel_coef[0])
        plt.plot(ab_coef[0])
        plt.legend(["ben", "mel", "ab"])
        plt.title("Ranked alpha for the support vectors in 3 classes (poly3 fold #" + str(i) + ")")
        plt.ylabel('Alpha')
        plt.xlabel('Support Vectors')
        #plt.savefig("alpha_poly3_fold" + str(i) + ".png")
        plt.show()


