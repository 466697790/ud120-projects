#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from time import time

sys.path.append("../tools/")
from tools.feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  

from sklearn import tree

clf = tree.DecisionTreeClassifier()
print(clf)

t0 = time()
clf = clf.fit(features, labels)
print("training time:", round(time() - t0, 3), "s")

# t1 = time()
# predict = clf.predict(features)
# print("predicting time:", round(time() - t1, 3), "s")
#
# mismatch_count = (predict != labels_test).sum()
# accuracy = 1 - mismatch_count / len(labels_test)
#
# print(mismatch_count,accuracy)
