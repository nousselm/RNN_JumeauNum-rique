import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(A, B):

    treeA = cKDTree(A)
    treeB = cKDTree(B)

    distA,_ = treeA.query(B)
    distB,_ = treeB.query(A)

    return np.mean(distA**2) + np.mean(distB**2)