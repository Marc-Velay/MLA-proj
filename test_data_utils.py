
import numpy as np
from collections import Counter
from random import randrange

# Utilitaire pour CNN_test: récupère les données de db_test

class DataSet(object):
    def __init__(self, filename_data, nbdata, L2normalize=False, batchSize=128):
        self.nbdata = nbdata
        # taille des images 56*56 pixels en couleurs RBG
        self.dim = 9408
        self.imgSize = 56
        self.data = None
        self.batchSize = batchSize
        self.curPos = 0
        self.x = None

        f = open(filename_data, 'rb')
        self.data = np.empty([nbdata, self.dim], dtype=np.float32)
        for i in range(nbdata):
            self.data[i,:] = np.fromfile(f, dtype=np.uint8, count=self.dim)
        f.close()

    def GetTestBase(self):
        return self.data
