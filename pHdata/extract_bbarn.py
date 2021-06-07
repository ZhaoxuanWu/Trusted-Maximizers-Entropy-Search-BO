
import sys
import numpy as np 

def getX(name):
    X = np.loadtxt("/home/qphong/Data/Workspace/hmmbo/src/pHdata/X_{}.txt".format(name))
    return X

def getY(name):
    Y = np.loadtxt("/home/qphong/Data/Workspace/hmmbo/src/pHdata/Y_{}.txt".format(name))
    return Y
