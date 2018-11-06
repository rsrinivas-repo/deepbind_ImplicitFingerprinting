import graphlab as gl
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd




strBaseFolder = "/Users/raghuramsrinivas/localdrive/education/deepbind/src/CollabFilteringV2/"

def runBasicCFModels(counter):

    results_CollabFiltering = pd.DataFrame()
    counter = str(counter)

    attrUser = "tid"
    attrItem = "molregno"

    training_data = gl.SFrame.read_csv(strBaseFolder+"data/baselineTrainSet_ByTarget_" + counter + ".csv")
    validation_data =  gl.SFrame.read_csv(strBaseFolder+"data/baselineValidationSet_ByTarget" + counter + ".csv")

    print(training_data.shape)


    model_factorization_reco = gl.ranking_factorization_recommender.create(training_data,
                                                                           binary_target=True,
                                                                           target="AFFINITY_FLAG",
                                                                           item_id="tid", random_seed=int(counter),
                                                                           user_id="molregno", num_factors=50,
                                                                           verbose=False ,
                                                                           user_data=MolWTs[["molregno","ScaledCounts"]] ,
                                                                           item_data=targetWTs[["tid","ScaledCounts"]]
                                                                           )

    predictedOutput = model_factorization_reco.predict(validation_data)

    # cast affinity scores from test set into numpy"
    npTest_y = np.asarray(validation_data['AFFINITY_FLAG'])

    npPred_y = np.asarray(predictedOutput)

    results_CollabFiltering["yTest_" + counter] = npTest_y
    results_CollabFiltering["yPred_" + counter] = npPred_y
    results_CollabFiltering["tid_iter_" + counter] = validation_data["tid"]
    results_CollabFiltering["molregno_iter_" + counter] = validation_data["molregno"]

    fpr, tpr, thresholds = roc_curve(npTest_y, npPred_y, pos_label=1)

    roc_auc = roc_auc_score(npTest_y, npPred_y)
    print("AUC : %.2f " % roc_auc)
    dictResults["Iteration_"+(counter)] = results_CollabFiltering



targetWTs = gl.SFrame.read_csv("/Users/raghuramsrinivas/localdrive/education/deepbind/src/CollabFilteringV2/data/TargetByCounts.csv")
MolWTs = gl.SFrame.read_csv("/Users/raghuramsrinivas/localdrive/education/deepbind/src/CollabFilteringV2/data/MolsByCounts.csv")


dictResults = {}

for count in range(0, 4):
    print ("\n\n******************************************************************************************")
    print ("Iteration %d" % count)
    runBasicCFModels(count)



import pickle as pkl

with open(strBaseFolder+"results/dictResultsCF_ByTarget_Run_sidefeatures_targetonly.pkl","w") as fp :
    pkl.dump(dictResults,fp)

#dictResults.keys()

