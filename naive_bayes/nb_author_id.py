#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

from tools.email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
print(gnb)

t0 = time()
gnb.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")

t1 = time()
predict = gnb.predict(features_test)
print("predicting time:", round(time() - t1, 3), "s")

mismatch_count = (predict != labels_test).sum()
accuracy = 1 - mismatch_count / len(labels_test)
print(predict)
print(labels_test)
print(mismatch_count,accuracy)


#########################################################
