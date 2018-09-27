import pandas as pd
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

batters = pd.read_csv("batters.csv")
pitchers = pd.read_csv("pitchers.csv")

NUM_CLUSTERS = 5
np.random.seed((1000,2000))

### clustering pitchers ###

pFeatures = pitchers.iloc[:,13:17].copy()
pWhitened = whiten(pFeatures)
pClusters = kmeans(pWhitened, NUM_CLUSTERS)
print(pClusters)

### clustering batters ###

bFeatures = batters.iloc[:,15:19].copy()
bWhitened = whiten(bFeatures)
bClusters = kmeans(bWhitened, NUM_CLUSTERS)
print(bClusters)

### organizing data for readability ###

cols = ['AVG','OBP','SLG','OPS']
pClustersRead = pd.DataFrame(data = pClusters[0], columns = cols)
print(pClustersRead)
bClustersRead = pd.DataFrame(data = bClusters[0], columns = cols)
print(bClustersRead)
pClustersRead.to_csv("pitcherClusters.csv")
bClustersRead.to_csv("batterClusters.csv")
