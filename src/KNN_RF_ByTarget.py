import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split

knnNeihbhors = 10
rf_estimators = 200


## Get the list of column names for expliciti FPs
listColumnsCompoundFingerPrints = ["tid","molregno","AFFINITY_FLAG"]

for i in range(0,512) :
    listColumnsCompoundFingerPrints.append("Compound_"+str(i))


strCurrBaseFolder = "/Users/raghuramsrinivas/localdrive/education/deepbind/src/CollabFilteringV2/"

results_RndFrst_KNN_by_Target = pd.DataFrame()
counter = 0

dictALLResults = {}

for i in range(0, 4):

    counter = str(i)

    strTrainSetFile = strCurrBaseFolder+"data/baselineTrainSet_WithCmpFingerPrints_%d.csv"%i
    strTestSetFile = strCurrBaseFolder+"data/baselineValidationSet_WithCmpFingerPrints_%d.csv"%i


    dfTrainSet = pd.read_csv(strTrainSetFile)
    dfValidationSet = pd.read_csv(strTestSetFile)

    print('Shape of training file %s'%(str(dfTrainSet.shape)))
    print('Shape of testing file %s'%str(dfValidationSet.shape))

    listTarget = dfValidationSet.tid.unique()

    print("Total Number of Unique Targets %d"%len(listTarget))

    dfTrainSet = dfTrainSet[listColumnsCompoundFingerPrints]
    dfValidationSet = dfValidationSet[listColumnsCompoundFingerPrints]
    listTarget = dfValidationSet.tid.unique()

    dfTrainSet.dropna(inplace=True)
    dfValidationSet.dropna(inplace=True)

    cnt = 0
    dfResults = pd.DataFrame()
    for target in listTarget[:25]:

        dfResultsByTarget = pd.DataFrame()

        dfTrainSetByTarget = dfTrainSet[dfTrainSet.tid == target]
        dfValidationSetByTarget = dfValidationSet[dfValidationSet.tid == target]

        if dfValidationSetByTarget.shape[0] < 10:
            continue

        yTrain = dfTrainSetByTarget.AFFINITY_FLAG
        xTrain = dfTrainSetByTarget

        if "tid" in xTrain.columns:
            del xTrain["tid"]

        if "AFFINITY_FLAG" in xTrain.columns:
            del xTrain["AFFINITY_FLAG"]

        if "molregno" in xTrain.columns:
            del xTrain["molregno"]

        clf = RandomForestClassifier(n_estimators=rf_estimators)

        clf.fit(xTrain, yTrain)

        yTest = dfValidationSetByTarget.AFFINITY_FLAG
        xTest = dfValidationSetByTarget.copy()
        if "tid" in xTest.columns:
            del xTest["tid"]

        if "AFFINITY_FLAG" in xTest.columns:
            del xTest["AFFINITY_FLAG"]

        if "molregno" in xTest.columns:
            del xTest["molregno"]

        yPredRnd = clf.predict(xTest)
        yPredProbRnd = clf.predict_proba(xTest)

        dfResultsByTarget["molregno"] = dfValidationSetByTarget.molregno
        dfResultsByTarget["yPredRnd"] = list(yPredRnd)
        dfResultsByTarget["yPredProbRnd"] = list(yPredProbRnd[:, 0])
        dfResultsByTarget["yTest"] = dfValidationSetByTarget.AFFINITY_FLAG
        dfResultsByTarget["tid"] = target
        cnt += 1

        knnclf = KNeighborsRegressor(n_neighbors=knnNeihbhors)
        knnclf.fit(xTrain, yTrain)

        try:
            yPredKNN = knnclf.predict(xTest)
            dfResultsByTarget["yPredKNN_iter_" + counter] = list(yPredKNN)
        except:
            dfResultsByTarget["yPredKNN_iter_" + counter] = .5

        dfResults = dfResults.append(dfResultsByTarget)
        sys.stdout.write('.')
        sys.stdout.flush()

    dictALLResults["Results" + counter] = dfResults
    print "Done with iteration"



import pickle
with open("results/Results_2_RndFrst_%d_Estimator_KNN_%d_Neigh_CompoundFingerprints_2.dict.raw"%(rf_estimators,knnNeihbhors)
        ,"w") as fp :
    pickle.dump(dictALLResults,fp)


'''
for i in range(0, 1):
    cnt = str(i)

    dfTemp = dictALLResults["Results" + str(cnt)].copy()
    yp = dfTemp["yPredRnd"]
    yt = dfTemp["yTest"]
    print("RandomForestMethod")
    print roc_auc_score(yt ,yp)

    yp = dfTemp["yPredKNN_iter_" + str(cnt)]

    print "KNN"
    print roc_auc_score(yt, yp)


'''
