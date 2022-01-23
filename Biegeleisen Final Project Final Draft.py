import pandas as pd
import numpy as np

heart_failure_data = pd.read_excel('heart_failure.xlsx')
# Reads the heart failure data to a pandas dataframe

heart_failure = heart_failure_data.to_numpy()
# Converts the heart failure data to a numpy array

X = heart_failure[:, 0:12]
# X contains the first 12 features of the heart failure data

y = heart_failure[:, 12]
# y contains the last feature of the heart failure data (whether death occurred)


"""
LASSO
"""

numLambdas = 350
LambdaRatio = 1.05
LambdaStart = 0.00015

from sklearn import linear_model

Lambda = LambdaStart
for i in range(numLambdas):
    clf = linear_model.Lasso(alpha = Lambda)
    clf.fit(X, y)
    print(clf.coef_)
    Lambda = LambdaRatio * Lambda

# Features 2, 6, and 11 appear to be the last features to go to 0
# Features 2, 6, and 11 correspond to creatinine phosphokinase, platelets, and time
# (Features are numbered from 0 to 11)
    
# The last feature to go to zero is 6 (platelets)
# The second to last feature to go to zero is 2 (creatinine phosphokinase)
# The third to last feature to go to zero is 11 (time)
# The fourth to last feature to go to zero is 4 (ejection fraction)
# The fifth to last feature to go to zero is 0 (age)
# The sixth to last feature to go to zero is 8 (serum sodium)
# The seventh to last feature to go to zero is 7 (serum creatinine)   
# The eigth to last feature to go to zero is 9 (sex)
# The nineth to last feature to go to zero is 3 (diabetes)
# The tenth to last feature to go to zero is 5 (high_blood_pressure)
# The eleventh to last feature to go to zero is 10 (smoking)
# The first feature to go to zero is 1 (anaemia)


XBest2 = X[:, [2, 6]]    
XBest3 = X[:, [2, 6, 11]]
XBest4 = X[:, [2, 4, 6, 11]]
XBest5 = X[:, [0, 2, 4, 6, 11]]
XBest6 = X[:, [0, 2, 4, 6, 8, 11]]
XBest7 = X[:, [0, 2, 4, 6, 7, 8, 11]]


C0 = heart_failure[0, :]
# Initializes Class 0 to be the first row of heart_failure in which
# the "death event" column (the final column) has a value of 0

C1 = heart_failure[14, :]
# Initializes Class 1 to be the first row of heart_failure in which
# the "death event" column (the final column) has a value of 1

for i in range(299):
    if heart_failure[i, 12] == 0:
        C0 = np.vstack((C0, heart_failure[i, :]))
        
        # If the "death event column" of row i in heart_failure has
        # a value of 0, row i gets added to Class 0
    
    if heart_failure[i, 12] == 1:
        C1 = np.vstack((C1, heart_failure[i, :]))   

        # If the "death event column" of row i in heart_failure has
        # a value of 1, row i gets added to Class 1
        

C0 = C0[1:, :]
# The first row of Class 0 is removed to account for the first row of 
# Class 0 being entered into Class 0 twice

C1 = C1[1:, :]
# The first row of Class 1 is removed to account for the first row of 
# Class 1 being entered into Class 1 twice

print('Mean platelets (Survived):', np.mean(C0[:, 6]))
print('Mean platelets (Did not survive):', np.mean(C1[:, 6]))

print('Mean creatinine phosphokinase (Survived):', np.mean(C0[:, 2]))
print('Mean creatinine phosphokinase (Did not survive):', np.mean(C1[:, 2]))

print('Mean time (Survived):', np.mean(C0[:, 11]))
print('Mean time (Did not survive):', np.mean(C1[:, 11]))

print('Mean ejection fraction (Survived):', np.mean(C0[:, 4]))
print('Mean ejection fraction (Did not survive):', np.mean(C1[:, 4]))

print('Mean age (Survived):', np.mean(C0[:, 0]))
print('Mean age (Did not survive):', np.mean(C1[:, 0]))

print('Mean serum sodium (Survived):', np.mean(C0[:, 8]))
print('Mean serum sodium (Did not survive):', np.mean(C1[:, 8]))

print('Mean serum creatinine (Survived):', np.mean(C0[:, 7]))
print('Mean serum creatinine (Did not survive):', np.mean(C1[:, 7]))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig1 = plt.figure()
ax = fig1.add_subplot(111, projection = '3d')
ax.scatter(C0[:, 2], C0[:, 6], C0[:, 11], marker = '+', c = 'b', label = 'Survived')
# Plots Class 0 by its three best features

ax.scatter(C1[:, 2], C1[:, 6], C1[:, 11], marker = '*', c = 'g', label = 'Did not Survive')
# Plots Class 1 by its three best features

ax.set_xlabel('Creatinine Phosphokinase')
ax.set_ylabel('Platelets')
ax.set_zlabel('Time')
ax.legend()

"""
Logistic Regression (All 12 Features, 1 Iteration)
"""

LR_Estimate = np.zeros(299)
LR_Accuracy = np.zeros(299)

# LR_Estimate and LR_Accuracy are initialized as zero arrays with 299 entries 

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X, y)

for i in range(len(X)):
    LR_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of LR_Estimate is the predicted value of y given X using Logistic Regression
    
    for j in range(len(X)):
        if LR_Estimate[j] == y[j]:
            LR_Accuracy[j] = 1

    # If the jth entry of LR_Estimate equals the jth entry of y, then
    # LR_Estimate correctly predicted the jth entry of y and the jth
    # entry of LR_Accuracy becomes 1
    
    # Otherwise the jth entry of LR_Accuracy remains 0

print('Logistic Regression Accuracy (12 Features, 1 Iteration):', sum(LR_Accuracy)/len(LR_Accuracy))
# prints the proportion of data correctly predicted using Logistic Regression


"""
SVM Gaussian (All 12 Features, 1 Iteration)
"""

SVMG_Estimate = np.zeros(299)
SVMG_Accuracy = np.zeros(299)

# SVMG_Estimate and SVMG_Accuracy are initialized as zero arrays with 299 entries 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma = 'auto'))
clf.fit(X, y)

for i in range(len(X)):
    SVMG_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of SVMG_Estimate is the predicted value of y given X using Gaussian SVM
    
    for j in range(len(X)):
        if SVMG_Estimate[j] == y[j]:
            SVMG_Accuracy[j] = 1

    # If the jth entry of SVMG_Estimate equals the jth entry of y, then
    # SVMG_Estimate correctly predicted the jth entry of y and the jth
    # entry of SVMG_Accuracy becomes 1
    
    # Otherwise the jth entry of SVMG_Accuracy remains 0

print('SVM Gaussian Accuracy (12 Features, 1 Iteration):', sum(SVMG_Accuracy)/len(SVMG_Accuracy))
# prints the proportion of data correctly predicted using Gaussian SVM


"""
SVM Linear (All 12 Features, 1 Iteration)
"""

SVML_Estimate = np.zeros(299)
SVML_Accuracy = np.zeros(299)

# SVML_Estimate and SVML_Accuracy are initialized as zero arrays with 299 entries 

clf = make_pipeline(StandardScaler(), SVC(kernel = 'linear', gamma = 'auto'))
clf.fit(X, y)

for i in range(len(X)):
    SVML_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of SVML_Estimate is the predicted value of y given X using Linear SVM
    
    for j in range(len(X)):
        if SVML_Estimate[j] == y[j]:
            SVML_Accuracy[j] = 1

    # If the jth entry of SVML_Estimate equals the jth entry of y, then
    # SVML_Estimate correctly predicted the jth entry of y and the jth
    # entry of SVML_Accuracy becomes 1
    
    # Otherwise the jth entry of SVML_Accuracy remains 0

print('SVM Linear Accuracy (12 Features, 1 Iteration):', sum(SVML_Accuracy)/len(SVML_Accuracy))
# prints the proportion of data correctly predicted using Linear SVM


"""
SVM Polynomial (All 12 Features, 1 Iteration)
"""

SVMP_Estimate = np.zeros(299)
SVMP_Accuracy = np.zeros(299)

# SVMP_Estimate and SVMP_Accuracy are initialized as zero arrays with 299 entries 

clf = make_pipeline(StandardScaler(), SVC(kernel = 'poly', gamma = 'auto'))
clf.fit(X, y)

for i in range(len(X)):
    SVMP_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of SVMP_Estimate is the predicted value of y given X using Polynomial SVM
    
    for j in range(len(X)):
        if SVMP_Estimate[j] == y[j]:
            SVMP_Accuracy[j] = 1

    # If the jth entry of SVMP_Estimate equals the jth entry of y, then
    # SVMP_Estimate correctly predicted the jth entry of y and the jth
    # entry of SVMP_Accuracy becomes 1
    
    # Otherwise the jth entry of SVMP_Accuracy remains 0

print('SVM Polynomial Accuracy (12 Features, 1 Iteration):', sum(SVMP_Accuracy)/len(SVMP_Accuracy))
# prints the proportion of data correctly predicted using Polynomial SVM


"""
SVM Sigmoid (All 12 Features, 1 Iteration)
"""

SVMS_Estimate = np.zeros(299)
SVMS_Accuracy = np.zeros(299)

# SVMS_Estimate and SVMS_Accuracy are initialized as zero arrays with 299 entries 

clf = make_pipeline(StandardScaler(), SVC(kernel = 'sigmoid', gamma = 'auto'))
clf.fit(X, y)

for i in range(len(X)):
    SVMP_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of SVMS_Estimate is the predicted value of y given X using Sigmoid SVM
    
    for j in range(len(X)):
        if SVMS_Estimate[j] == y[j]:
            SVMS_Accuracy[j] = 1

    # If the jth entry of SVMS_Estimate equals the jth entry of y, then
    # SVMS_Estimate correctly predicted the jth entry of y and the jth
    # entry of SVMS_Accuracy becomes 1
    
    # Otherwise the jth entry of SVMS_Accuracy remains 0

print('SVM Sigmoid Accuracy (12 Features, 1 Iteration):', sum(SVMS_Accuracy)/len(SVMS_Accuracy))
# prints the proportion of data correctly predicted using Polynomial SVM


"""
Random Forest (All 12 Features, 1 Iteration)
"""

RF_Estimate = np.zeros(299)
RF_Accuracy = np.zeros(299)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 2, random_state = 0)
clf.fit(X, y)

for i in range(len(X)):
    RF_Estimate[i] = clf.predict([X[i, :]])
    
    # The ith entry of RF_Estimate is the predicted value of y given X using Random Forest
    
    for j in range(len(X)):
        if RF_Estimate[j] == y[j]:
            RF_Accuracy[j] = 1

    # If the jth entry of SVM_Estimate equals the jth entry of y, then
    # SVM_Estimate correctly predicted the jth entry of y and the jth
    # entry of SVM_Accuracy becomes 1

    # Otherwise the jth entry of SVM_Accuracy remains 0
    
print('Random Forest Accuracy (12 Features, 1 Iteration):', sum(RF_Accuracy)/len(RF_Accuracy))
# prints the proportion of data correctly predicted using Random Forest


"""
Logistic Regression ROC and AUC (All 12 Features, 1 Iteration)
Information on coding ROC and AUC from: 
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
"""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.1, random_state = 2)
# split into train/test sets

ns_probs = [0 for _ in range(len(testy))]
# generate a no skill prediction (majority class)

model = LogisticRegression()
model.fit(trainX, trainy)
# fit a model

lr_probs = model.predict_proba(testX)
# predict probabilities

lr_probs = lr_probs[:, 1]
# keep probabilities for the positive outcome only

ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# calculate scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# summarize scores

ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# calculate roc curves

fig2 = plt.figure()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label = 'No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label = 'Logistic Regression')
# plot the roc curve for the model

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# axis labels

pyplot.legend(loc = 'lower right')
# show the legend


"""
Random Forest ROC and AUC (All 12 Features, 1 Iteration)
Information on coding ROC and AUC from: 
https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
"""

trainX, testX, trainy, testy = train_test_split(X, y, test_size = 0.1, random_state = 2)
# split into train/test sets

ns_probs = [0 for _ in range(len(testy))]
# generate a no skill prediction (majority class)

model = RandomForestClassifier()
model.fit(trainX, trainy)
# fit a model

lr_probs = model.predict_proba(testX)
# predict probabilities

lr_probs = lr_probs[:, 1]
# keep probabilities for the positive outcome only

ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# calculate scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (lr_auc))
# summarize scores

ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# calculate roc curves

fig3 = plt.figure()
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label = 'No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label = 'Random Forest')
# plot the roc curve for the model

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# axis labels

pyplot.legend(loc = 'lower right')
# show the legend


"""
Logistic Regression with 10-Fold Cross Validation (All 12 Features)
Information on coding Repeated K-Fold Cross Validation from:
https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/
"""
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_12 = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_12 = cross_val_score(model, X, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (12 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_12), np.std(LR_accuracy_scores_12)))
print('Logistic Regression ROC AUC (12 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_12), np.std(LR_roc_auc_scores_12)))


"""
SVM with 10-Fold Cross Validation (All 12 Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_12 = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_12 = cross_val_score(model, X, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (12 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_12), np.std(SVM_accuracy_scores_12)))
print('SVM Gaussian ROC AUC (12 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_12), np.std(SVM_roc_auc_scores_12)))


"""
Random Forest with 10-Fold Cross Validation (All 12 Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_12 = cross_val_score(model, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_12 = cross_val_score(model, X, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (12 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_12), np.std(RF_accuracy_scores_12)))
print('Random Forest ROC AUC (12 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_12), np.std(RF_roc_auc_scores_12)))


"""
Logistic Regression with 10-Fold Cross Validation (7 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_7 = cross_val_score(model, XBest7, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_7 = cross_val_score(model, XBest7, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (7 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_7), np.std(LR_accuracy_scores_7)))
print('Logistic Regression ROC AUC (7 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_7), np.std(LR_roc_auc_scores_7)))


"""
SVM with 10-Fold Cross Validation (7 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_7 = cross_val_score(model, XBest7, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_7 = cross_val_score(model, XBest7, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (7 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_7), np.std(SVM_accuracy_scores_7)))
print('SVM Gaussian ROC AUC (7 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_7), np.std(SVM_roc_auc_scores_7)))


"""
Random Forest with 10-Fold Cross Validation (7 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_7 = cross_val_score(model, XBest7, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_7 = cross_val_score(model, XBest7, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (7 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_7), np.std(RF_accuracy_scores_7)))
print('Random Forest ROC AUC (7 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_7), np.std(RF_roc_auc_scores_7)))


"""
Logistic Regression with 10-Fold Cross Validation (6 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_6 = cross_val_score(model, XBest6, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_6 = cross_val_score(model, XBest6, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (5 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_6), np.std(LR_accuracy_scores_6)))
print('Logistic Regression ROC AUC (5 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_6), np.std(LR_roc_auc_scores_6)))


"""
SVM with 10-Fold Cross Validation (6 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_6 = cross_val_score(model, XBest6, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_6 = cross_val_score(model, XBest6, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (6 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_6), np.std(SVM_accuracy_scores_6)))
print('SVM Gaussian ROC AUC (6 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_6), np.std(SVM_roc_auc_scores_6)))


"""
Random Forest with 10-Fold Cross Validation (6 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_6 = cross_val_score(model, XBest6, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_6 = cross_val_score(model, XBest6, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (6 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_6), np.std(RF_accuracy_scores_6)))
print('Random Forest ROC AUC (6 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_6), np.std(RF_roc_auc_scores_6)))


"""
Logistic Regression with 10-Fold Cross Validation (5 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_5 = cross_val_score(model, XBest5, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_5 = cross_val_score(model, XBest5, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (5 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_5), np.std(LR_accuracy_scores_5)))
print('Logistic Regression ROC AUC (5 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_5), np.std(LR_roc_auc_scores_5)))


"""
SVM with 10-Fold Cross Validation (5 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_5 = cross_val_score(model, XBest5, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_5 = cross_val_score(model, XBest5, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (5 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_5), np.std(SVM_accuracy_scores_5)))
print('SVM Gaussian ROC AUC (5 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_5), np.std(SVM_roc_auc_scores_5)))


"""
Random Forest with 10-Fold Cross Validation (5 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_5 = cross_val_score(model, XBest5, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_5 = cross_val_score(model, XBest5, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (5 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_5), np.std(RF_accuracy_scores_5)))
print('Random Forest ROC AUC (5 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_5), np.std(RF_roc_auc_scores_5)))


"""
Logistic Regression with 10-Fold Cross Validation (4 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_4 = cross_val_score(model, XBest4, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_4 = cross_val_score(model, XBest4, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (4 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_4), np.std(LR_accuracy_scores_4)))
print('Logistic Regression ROC AUC (4 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_4), np.std(LR_roc_auc_scores_4)))


"""
SVM with 10-Fold Cross Validation (4 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_4 = cross_val_score(model, XBest4, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_4 = cross_val_score(model, XBest4, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (4 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_4), np.std(SVM_accuracy_scores_4)))
print('SVM Gaussian ROC AUC (4 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_4), np.std(SVM_roc_auc_scores_4)))


"""
Random Forest with 10-Fold Cross Validation (4 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_4 = cross_val_score(model, XBest4, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_4 = cross_val_score(model, XBest4, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (4 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_4), np.std(RF_accuracy_scores_4)))
print('Random Forest ROC AUC (4 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_4), np.std(RF_roc_auc_scores_4)))


"""
Logistic Regression with 10-Fold Cross Validation (3 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_3 = cross_val_score(model, XBest3, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_3 = cross_val_score(model, XBest3, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (3 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_3), np.std(LR_accuracy_scores_3)))
print('Logistic Regression ROC AUC (3 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_3), np.std(LR_roc_auc_scores_3)))


"""
SVM with 10-Fold Cross Validation (3 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_3 = cross_val_score(model, XBest3, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_3 = cross_val_score(model, XBest3, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (3 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_3), np.std(SVM_accuracy_scores_3)))
print('SVM Gaussian ROC AUC (3 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_3), np.std(SVM_roc_auc_scores_3)))


"""
Random Forest with 10-Fold Cross Validation (3 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_3 = cross_val_score(model, XBest3, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_3 = cross_val_score(model, XBest3, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (3 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_3), np.std(RF_accuracy_scores_3)))
print('Random Forest ROC AUC (3 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_3), np.std(RF_roc_auc_scores_3)))


"""
Logistic Regression with 10-Fold Cross Validation (2 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = LogisticRegression()
LR_accuracy_scores_2 = cross_val_score(model, XBest2, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
LR_roc_auc_scores_2 = cross_val_score(model, XBest2, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Logistic Regression Accuracy (2 Features): %.3f (%.3f)' % (np.mean(LR_accuracy_scores_2), np.std(LR_accuracy_scores_2)))
print('Logistic Regression ROC AUC (2 Features): %.3f (%.3f)' % (np.mean(LR_roc_auc_scores_2), np.std(LR_roc_auc_scores_2)))


"""
SVM with 10-Fold Cross Validation (2 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = SVC(kernel = 'rbf', gamma = 'auto')
SVM_accuracy_scores_2 = cross_val_score(model, XBest2, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
SVM_roc_auc_scores_2 = cross_val_score(model, XBest2, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('SVM Gaussian Accuracy (2 Features): %.3f (%.3f)' % (np.mean(SVM_accuracy_scores_2), np.std(SVM_accuracy_scores_2)))
print('SVM Gaussian ROC AUC (2 Features): %.3f (%.3f)' % (np.mean(SVM_roc_auc_scores_2), np.std(SVM_roc_auc_scores_2)))


"""
Random Forest with 10-Fold Cross Validation (2 Best Features)
"""

cv = RepeatedKFold(n_splits = 10, n_repeats = 100, random_state = 1)
model = RandomForestClassifier()
RF_accuracy_scores_2 = cross_val_score(model, XBest2, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
RF_roc_auc_scores_2 = cross_val_score(model, XBest2, y, scoring = 'roc_auc', cv = cv, n_jobs = -1)
print('Random Forest Accuracy (2 Features): %.3f (%.3f)' % (np.mean(RF_accuracy_scores_2), np.std(RF_accuracy_scores_2)))
print('Random Forest ROC AUC (2 Features): %.3f (%.3f)' % (np.mean(RF_roc_auc_scores_2), np.std(RF_roc_auc_scores_2)))


"""

DeBonis Classifier: (10 iterations):


12 Features (Accuracy):
Error Percentage: 23.7858 (100 - Error for Accuracy Percentage)
Standard Deviation: 7.7700

Error Percentage for Class 0: 14.5804
Standard Deviation for Class 0: 8.8660

Error Percentage for Class 1: 43.2444
Standard Deviation for Class 1: 17.7705


12 Features (ROC AUC):
ROC AUC: 0.7845
Standard Deviation: 0.0809


7 Features (Accuracy):
Error Percentage: 20.7459 (100 - Error for Accuracy Percentage)
Standard Deviation: 7.2750

Error Percentage for Class 0: 12.4239
Standard Deviation for Class 0: 8.3493

Error Percentage for Class 1: 38.4889
Standard Deviation for Class 1: 15.5504


7 Features (ROC AUC):
ROC AUC: 0.8204
Standard Deviation: 0.0878


3 Features (Accuracy):
Error Percentage: 22.6387 (100 - Error for Accuracy Percentage)
Standard Deviation: 7.3879

Error Percentage for Class 0: 11.4870
Standard Deviation for Class 0: 8.1694

Error Percentage for Class 1: 46.4178
Standard Deviation for Class 1: 15.4872


3 Features (ROC AUC):
ROC AUC: 0.7779
Standard Deviation: 0.0949

"""