import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import sys



def calcEmptySpaceForDelta(validData,trainData,d,metric) :

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto' ,metric=similarityMeasure).fit(trainData[listFPCols])

    
    for i in range(0,validData.shape[0]) :
        point = validData[listFPCols].as_matrix()[i]
        point = point.reshape(1,512)
        
        distance = nbrs.kneighbors(point)[0][0][0]
        
        if (distance < d) :
            listDistances.append(1) 
        else :
            listDistances.append(0) 
        
    #retVal = np.sum(listDistances) / np.linalg.norm(validData[listFPCols].as_matrix())
    retVal = float(np.sum(listDistances)) / validData.shape[0]
    
    return retVal
        

evalMetrics = ["jaccard","euclidean","rogerstanimoto"]
listBiasByTarget =[]
listDistances = []
d = np.arange(0,1,.01)
for target in dfTrain.tid.unique():
    
 
    
    dictBiasByTarget = {}

    trainData = dfTrain[dfTrain.tid==target]
    validData = dfValid[dfValid.tid==target]
    
    Va = validData[validData.AFFINITY_FLAG==1]
    Vi = validData[validData.AFFINITY_FLAG==0]

    Ti = trainData[trainData.AFFINITY_FLAG==0]
    Ta = trainData[trainData.AFFINITY_FLAG==1]
    
    AA = 0 
    AI = 0 
    II = 0
    IA = 0 

    try:
    	for metric in range(0,len(evalMetric)) :
			for i in range(0,d.shape[0]) :
				AA+=calcEmptySpaceForDelta(Va,Ta,d[i],metric)
				AI+=calcEmptySpaceForDelta(Va,Ti,d[i],metric)
				II+=calcEmptySpaceForDelta(Vi,Ti,d[i],metric)
				IA+=calcEmptySpaceForDelta(Vi,Ta,d[i],metric)


			AVEBias =  ((AA-AI)  + (II- IA)) / d.shape[0]
		
			dictBiasByTarget["Target"]  = target
			dictBiasByTarget["AVEBias"]  = AVEBias
			dictBiasByTarget["ActivityCount"] = trainData.shape[0]
			dictBiasByTarget["AA"]  = AA
			dictBiasByTarget["AI"]  = AI
			dictBiasByTarget["II"]  = II
			dictBiasByTarget["IA"]  = IA
		
			listBiasByTarget.append(dictBiasByTarget)
		except:
			continue
		sys.stdout.write(".")
