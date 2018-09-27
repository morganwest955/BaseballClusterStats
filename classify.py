import sys
import pandas as pd
import numpy as np
from scipy.cluster.vq import whiten
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

### Enter data to search ###

##playerType = 'pitcher'
##player = "Kevin Gausman"

### read in and whiten data ###

batters = pd.read_csv("batters.csv")
pitchers = pd.read_csv("pitchers.csv")
pFeatures = pitchers.iloc[:,13:17].copy()
pWhitened = whiten(pFeatures)
pWhitened = pd.DataFrame(data=pWhitened, columns=['AVG','OBP','SLG','OPS'])
bFeatures = batters.iloc[:,15:19].copy()
bWhitened = whiten(bFeatures)
bWhitened = pd.DataFrame(data=bWhitened, columns=['AVG','OBP','SLG','OPS'])
batters = batters['Name']
batters = pd.concat([batters, bWhitened], axis=1, join='inner')
pitchers = pitchers['Name']
pitchers = pd.concat([pitchers, pWhitened], axis=1, join='inner')
##batters = batters.groupby(['Name']).mean()
##pitchers = pitchers.groupby(['Name']).mean()

### read in clusters ###

pClusters = pd.read_csv("pitcherClusters.csv")
bClusters = pd.read_csv("batterClusters.csv")

def calcCloseCluster(playerType, player):
    valid = True
    if (playerType == 'pitcher'):
        dataset = pitchers
        clusters = pClusters
    elif (playerType == 'batter'):
        dataset = batters
        clusters = bClusters
    else:
        print("Invalid player type argument")
        valid = False
        
    if (valid and player in dataset.Name.values):
        dataset = dataset.groupby(['Name']).mean()
        for index, row in dataset.iterrows():
            if (index == player):
                playerRow = [row[0],row[1],row[2],row[3]]
                exit
        minDist = 100
        minCluster = 10
        for i, row in clusters.iterrows():
            clusterRow = [row[1],row[2],row[3],row[4]]
            dist = distance.euclidean(clusterRow, playerRow)
            if (dist < minDist):
                minDist = dist
                minCluster = i
        return[player, minCluster, minDist]
    else:
        print("Player does not exist in data set")

pitchersClusterTable = []
catPitchers = pitchers.groupby(['Name']).mean()
for name, etc in catPitchers.iterrows():
    pitchersClusterTable.append(calcCloseCluster('pitcher', name))
battersClusterTable = []
catBatters = batters.groupby(['Name']).mean()
for name, etc in catBatters.iterrows():
    battersClusterTable.append(calcCloseCluster('batter', name))

cols = (['Name'],['Cluster'],['DistToCluster'])
ClusteredPitchers = pd.DataFrame(data = pitchersClusterTable, columns = cols)
ClusteredBatters = pd.DataFrame(data = battersClusterTable, columns = cols)
print(ClusteredPitchers)
print(ClusteredBatters)
