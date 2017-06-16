# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
from model_stacker import *
import numpy as np
import math
from math import log
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from time import gmtime, strftime
import scipy
import sys
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
from string import punctuation
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
import time
from scipy import sparse
from matplotlib import *
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import operator
from sklearn import svm
import pickle

# <codecell>


test = np.array(p.read_table('./test.csv', sep = ","))

################################################################################
# READ IN THE TEST DATA
################################################################################
# all data from opening 1 to straight to opening 10
X_test_stockdata = normalize10day(test[:,range(2, 48)]) # load in test data
X_test_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(25)))

#X_test = np.hstack((X_test_stockindicators, X_test_stockdata))
X_test = X_test_stockdata


#load model
filename = 'allmodel.sav'
best_model = pickle.load(open(filename, 'rb'))
best_model.fit(new_X, new_Y) = best_model.score(new_X, new_Y)


print "prediction"
# do a prediction and save it
pred = best_model.predict_proba(new_X)[:,1]
testfile = p.read_csv('./test.csv', sep=",", na_values=['?'], index_col=[0,1])

# submit as D multiplied by 100 + stock id
testindices = [100 * D + StId for (D, StId) in testfile.index]

pred_df = p.DataFrame(np.vstack((testindices, pred)).transpose(), columns=["Id", "Prediction"])
pred_df.to_csv('./predictions/' + modelname + '/' + modelname + ' ' + strftime("%m-%d %X") + " C-" + str(round(best_C,4)) + " CV-" + str(round(best_cv, 4)) + ".csv", index = False)

print "submission file created"
