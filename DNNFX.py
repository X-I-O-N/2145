#sklearn main for voting ensemble 
#optional step using mlxtend stack multiple DL classifiers before passing to voting ensemble (3 DL models with 1000 different DL models each are sent to voting ensemble)
#sklflow for DL classifier
#sklearn-deap for optimizing DL classifier parameters with a genetic algorithm like how kai uses genetic programming to optimize a linear curves parameters
#**NEW METHOD** optimize seperate models using sklearn-deap and then stack them together and predict_proba
#**NEW METHOD2** duplicate the perfect stacked model and feed to votingclassifier. Lets check the results on this.
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>
#import autosklearn.classification
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from evolutionary_search import EvolutionaryAlgorithmSearchCV
from skflow import *
from mlxtend.classifier import StackingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from itertools import chain
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
from sklearn import *
# <codecell>
def tied_rank(x):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    x : list of numbers, numpy array

    Returns
    -------
    score : list of numbers
            The tied rank f each element in x

    """
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i): 
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1): 
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

def auc(actual, posterior):
    """
    This function is by Ben Hamner and taken from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    
    Computes the area under the receiver-operater characteristic (AUC)

    This function computes the AUC error metric for binary classification.

    Parameters
    ----------
    actual : list of binary numbers, numpy array
             The ground truth value
    posterior : same type as actual
                Defines a ranking on the binary numbers, from most likely to
                be positive to least likely to be positive.

    Returns
    -------
    score : double
            The mean squared error between actual and posterior

    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x==1])
    num_negative = len(actual)-num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i]==1])
    auc = ((sum_positive - num_positive*(num_positive+1)/2.0) /
           (num_negative*num_positive))
    sys.stdout.write('.')
    return auc

def auc_scorer(estimator, X, y):
    predicted = estimator.predict_proba(X)[:,1]
    return auc(y, predicted)
       
def normalize10day(stocks):
    def process_column(i):
        if operator.mod(i, 5) == 1:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 2:
            return stocks[:,i] * 0
        if operator.mod(i, 5) == 4:
            return stocks[:,i] * 0
            #return np.log(stocks[:,i] + 1)
        else:
            return stocks[:,i] / stocks[:,0]
    n = stocks.shape[0]
    stocks_dat =  np.array([ process_column(i) for i in range(46)]).transpose()
    #stocks_movingavgO9O10 = np.array([int(i > j) for i,j in zip(stocks_dat[:,45], stocks_dat[:,40])]).reshape((n, 1))
    #stocks_movingavgC9O10 = np.array([int(i > j) for i,j in zip(stocks_dat[:,45], stocks_dat[:,43])]).reshape((n, 1))
    #return np.hstack((stocks_dat, stocks_movingavgO9O10, stocks_movingavgC9O10))
    return stocks_dat
    

# <codecell>


# <codecell>

print "loading data.."
train = np.array(p.read_table('./FXGLORY5MIN.csv', sep = ","))
test = np.array(p.read_table('./bintest.csv', sep = ","))

################################################################################
# READ IN THE TEST DATA
################################################################################
# all data from opening 1 to straight to opening 10
X_test_stockdata = normalize10day(test[:,range(2, 48)]) # load in test data
X_test_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(25)))

#X_test = np.hstack((X_test_stockindicators, X_test_stockdata))
X_test = X_test_stockdata

#np.identity(94)[:,range(93)]

################################################################################
# READ IN THE TRAIN DATA
################################################################################
n_windows = 490
windows = range(n_windows)

X_windows = [train[:,range(1 + 5*w, 47 + 5*w)] for w in windows]
X_windows_normalized = [normalize10day(w) for w in X_windows]
X_stockdata = np.vstack(X_windows_normalized)
X_stockindicators = np.vstack((np.identity(94)[:,range(93)] for i in range(n_windows)))

#X = np.hstack((X_stockindicators, X_stockdata))
X = X_stockdata
#X = preprocessing.normalize(X1, norm='l2')

# read in the response variable
y_stockdata = np.vstack([train[:, [46 + 5*w, 49 + 5*w]] for w in windows])
y = (y_stockdata[:,1] - y_stockdata[:,0] > 0) + 0

# chain.from_iterable is basically a "flatten" function, that takes a list of lists and 
# converts it to one list
# columns we want are just the opening and closing prices
#columns_we_want = list(chain.from_iterable([[5 * x, 5 * x + 3] for x in range(10)]))[:-1]
# we get our matrix of open and close prices, and normalize the data such that all data
# is divided by the opening price on the first day
#X = np.array([l/l[0] for l in train[:, columns_we_want]])
    
# we make indicators of whether or not the stock went up that day.
#y = (train[:, 48] > train[:, 45]) + 0

print "this step done"

print "tuning models using genetic algo"
# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(1288, input_dim=1288, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# larger model
def create_larger():
	# create model
	model = Sequential()
	model.add(Dense(1288, input_dim=1288, kernel_initializer='normal', activation='relu'))
	model.add(Dense(644, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# smaller model
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(644, input_dim=1288, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# <codecell>

print "preparing models"

modelname = "keras"

if modelname == "keras":
	C = np.linspace(300, 5000, num = 10)[::-1]
	estimators = []
	estimators.append(('standardize', StandardScaler()))
	estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
	models = Pipeline(estimators)

	

if modelname == "voteDNN3nostack": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    clf1 = TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, steps=200)
    clf2 = TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, steps=200)
    clf3 = TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3, steps=200)
    models = [sklearn.ensemble.VotingClassifier(estimators=[('DNN1', clf1), ('DNN2', clf2), ('DNN3', clf3)], voting='soft', weights=[1, 1, 1])]


if modelname == "asl": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [autosklearn.classification.AutoSklearnClassifier()]

if modelname == "stack": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    clf1 = sklearn.tree.DecisionTreeClassifier(max_depth=4)
    clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
    clf3 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200)
    clf4 = lm.LogisticRegression()
    clf5 = RandomForestClassifier(n_estimators = 100)
    clf6 = lm.LogisticRegression(penalty = "l1", C = 5000)
    clf11 = sklearn.tree.DecisionTreeClassifier(max_depth=50)
    clf21 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7000)
    clf31 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=2000)
    clf41 = sklearn.ensemble.ExtraTreesClassifier(max_depth=None)
    clf51 = RandomForestClassifier(n_estimators = 1000)
    clf61 = lm.LogisticRegression(penalty = "l1", C = 50000)
    gbc = sklearn.ensemble.GradientBoostingClassifier()
    #lr = lm.LogisticRegression()
    models = [StackingClassifier(classifiers=[clf1,clf2,clf3,clf4,clf5,clf6,clf11,clf21,clf31,clf41,clf51,clf61], meta_classifier=gbc)]

if modelname == "GBC": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)]

if modelname == "ridge": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l2", C = c) for c in C]

if modelname == "ada": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [sklearn.ensemble.AdaBoostClassifier(base_estimator=lm.LogisticRegression(penalty = "l1", C = 3433), n_estimators=50, learning_rate=1.0)]

if modelname == "vote": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    clf1 = sklearn.tree.DecisionTreeClassifier(max_depth=4)
    clf2 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=7)
    clf3 = sklearn.ensemble.GradientBoostingClassifier(n_estimators=200)
    clf4 = lm.LogisticRegression()
    clf5 = RandomForestClassifier(n_estimators = 100)
    clf6 = lm.LogisticRegression(penalty = "l1", C = 5000)
    models = [sklearn.ensemble.VotingClassifier(estimators=[('dtc', clf1), ('knc', clf2), ('gbc', clf3), ('lr', clf4), ('rfc', clf5), ('lass', clf6)], voting='soft', weights=[2, 3, 2, 1, 2, 1])]

if modelname == "lasso": 
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty = "l1", C = c) for c in C]

if modelname == "sgd": 
    C = np.linspace(0.00005, .01, num = 5)
    models = [lm.SGDClassifier(loss = "log", penalty = "l2", alpha = c, warm_start = False) for c in C]
    
if modelname == "randomforest":
    C = np.linspace(50, 300, num = 10)
    models = [RandomForestClassifier(n_estimators = int(c)) for c in C]

if modelname == "Perceptron":
    C = np.linspace(50, 300, num = 10)
    models = [lm.Perceptron()]

if modelname == "mlp":
    C = np.linspace(50, 300, num = 10)
    models = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1,max_iter=100)]

if modelname == "KNC":
    C = np.linspace(50, 300, num = 10)
    models = [sklearn.neighbors.KNeighborsClassifier()]

if modelname == "nblend":
    C = np.linspace(50, 300, num = 10)
    models = [sklearn.neighbors.KNeighborsClassifier(n_jobs=-1,)]

if modelname == "svc":
    C = np.linspace(50, 300, num = 10)
    models = [svm.SVC()]

if modelname == "blend":
    C = np.linspace(300, 5000, num = 10)[::-1]
    models = [lm.LogisticRegression(penalty='l2', C = 5000),
          lm.LogisticRegression(penalty='l1', C = 500),
          RandomForestClassifier(n_estimators = 100),
          GradientBoostingClassifier(n_estimators = 200),  sklearn.neighbors.KNeighborsClassifier(),
          ]

#print "calculating cv scores"
#cv_scores = [0] * len(models)
#for i, model in enumerate(models):
    # for all of the models, save the cross-validation scores into the array cv_scores
 #   cv_scores[i] = np.mean(cross_validation.cross_val_score(model, X, y, cv=5, scoring = auc_scorer, n_jobs=-1))
    #cv_scores[i] = np.mean(cross_validation.cross_val_score(model, X, y, cv=5, score_func = auc))
  #  print " (%d/%d) C = %f: CV = %f" % (i + 1, len(C), C[i], cv_scores[i])

# find which model and C is the best
#best = cv_scores.index(max(cv_scores))
#best_model = models[best]
#best_cv = cv_scores[best]
#best_C = C[best]
#print "BEST %f: %f" % (best_C, best_cv)

print "SCORING DNN"
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
best_model = models

print "training on full data"
# fit the best model on the full data
best_model.fit(X, y)
#save model to disk
filename = 'model.sav'
pickle.dump(best_model, open(filename, 'wb'))
print "all done Teerth"