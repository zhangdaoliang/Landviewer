import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import *
import operator


def creatDataSet():
    group = ([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    print(group)
    print(labels)
    return group, labels


creatDataSet()
