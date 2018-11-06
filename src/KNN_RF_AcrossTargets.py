import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sys


import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

listColumnsCompoundFingerPrintsAndProtSeqs = ["tid","molregno","AFFINITY_FLAG"]

knnNeihbhors = 5
rf_estimators = 200

for i in range(0,512) :
    listColumnsCompoundFingerPrintsAndProtSeqs.append("Compound_"+str(i))

for i in range(0,150):
    listColumnsCompoundFingerPrintsAndProtSeqs.append(str(i))


counter = 0

dictALLResults = {}


for i in range(0, 4):

    dfResults = pd.DataFrame()

    counter = str(i)

    strTrainSetFile = "data/baselineTrainSet_WithCmpFingerPrints_AndProtienSeq_"
    strTestSetFile = "data/baselineValidationSet_WithCmpFingerPrints__AndProtienSeq_"

    strTrainSetFile = strTrainSetFile + str(i) + ".csv"
    strTestSetFile = strTestSetFile + str(i) + ".csv"

    print "Starting Iteration " + counter

    dfTrainSet = pd.read_csv(strTrainSetFile)
    dfValidationSet = pd.read_csv(strTestSetFile)

    listTarget = dfValidationSet.tid.unique()

    dfTrainSet = dfTrainSet[listColumnsCompoundFingerPrintsAndProtSeqs]
    dfValidationSet = dfValidationSet[listColumnsCompoundFingerPrintsAndProtSeqs]
    listTarget = dfValidationSet.tid.unique()

    dfTrainSet.dropna(inplace=True)
    dfValidationSet.dropna(inplace=True)

    # dfTrainSet = dfTrainSet.sample(5000)
    # dfValidationSet = dfValidationSet.sample(500)

    yTrain = dfTrainSet.AFFINITY_FLAG
    xTrain = dfTrainSet

    if "tid" in xTrain.columns:
        del xTrain["tid"]

    if "AFFINITY_FLAG" in xTrain.columns:
        del xTrain["AFFINITY_FLAG"]

    if "molregno" in xTrain.columns:
        del xTrain["molregno"]

    clf = RandomForestClassifier(n_estimators=rf_estimators)

    clf.fit(xTrain, yTrain)

    yTest = dfValidationSet.AFFINITY_FLAG
    xTest = dfValidationSet.copy()

    if "tid" in xTest.columns:
        del xTest["tid"]

    if "AFFINITY_FLAG" in xTest.columns:
        del xTest["AFFINITY_FLAG"]

    if "molregno" in xTest.columns:
        del xTest["molregno"]

    yPredRnd = clf.predict(xTest)
    yPredProbRnd = clf.predict_proba(xTest)

    dfResults["molregno"] = dfValidationSet.molregno
    dfResults["yPredRnd"] = list(yPredRnd)
    dfResults["yPredProbRnd"] = list(yPredProbRnd[:, 0])
    dfResults["yTest_iter"] = dfValidationSet.AFFINITY_FLAG
    dfResults["tid_iter"] = dfValidationSet.tid

    knnclf = KNeighborsRegressor(n_neighbors=knnNeihbhors)
    knnclf.fit(xTrain, yTrain)

    yPredKNN = knnclf.predict(xTest)

    dfResults["yPredKNN"] = list(yPredKNN)

    dictALLResults["Results" + counter] = dfResults

    print "Done with iteration"


import pickle
with open("results/Results_3_RndFrst_%d_KNN__%d_CompoundFingerprints_ProteinSeqs_32.dict.raw"%(rf_estimators,knnNeihbhors),"w") as fp :
    pickle.dump(dictALLResults,fp)
    #fp.write(str(dictALLResults))


