#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from tools.email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn import svm

# clf = svm.SVC(kernel='linear')
clf = svm.SVC(C=10000.0)
print(clf.kernel)
print(clf)
# features_train_end_index = int(len(features_train) / 100)
# labels_train_end_index = int(len(labels_train) / 100)
# print(features_train_end_index, labels_train_end_index)
#
# features_train = features_train[:features_train_end_index]
# labels_train = labels_train[:labels_train_end_index]

t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")

t1 = time()
# predict = clf.predict(features_test)
predict = clf.predict(features_test)
p = predict[10]
p2 = predict[26]
p3 = predict[50]

print(p, p2, p3)
print("predicting time:", round(time() - t1, 3), "s")

# mismatch_count = (predict != labels_test).sum()
mismatch_count = (predict != labels_test).sum()
accuracy = 1 - mismatch_count / len(labels_test)
print(predict)
p_list = list(predict)

print(labels_test)
print(mismatch_count, accuracy)
#########################################################
