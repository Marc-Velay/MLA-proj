
import numpy as np
from collections import Counter
from random import randrange


class DataSet(object):
    def __init__(self, filename_data, filename_labels, nbdata, L2normalize=False, batchSize=128, splitRatio=0.7):
        self.nbdata = nbdata
        # taille des images 56*56 pixels en couleurs RBG
        self.dim = 9408
        self.imgSize = 56
        self.data = None
        self.labels = None
        self.test_data = None
        self.test_labels = None
        self.batchSize = batchSize
        self.curPos = 0
        self.x = None
        self.y_desired = None
        self.PosClassWeight = 0.1369

        f = open(filename_data, 'rb')
        self.data = np.empty([nbdata, self.dim], dtype=np.float32)
        for i in range(nbdata):
            self.data[i,:] = np.fromfile(f, dtype=np.uint8, count=self.dim)
        f.close()

        f = open(filename_labels, 'r')
        self.labels = np.empty([nbdata, 2], dtype=np.int)
        for i in range(nbdata):
            line=int(f.readline())
            self.labels[i,line] = 1
            #self.labels[i] = line
        f.close()

        print ('nb data : ', self.nbdata)

        tmpdata = np.empty([1, self.dim], dtype=np.float32)
        tmplabel = np.empty([1, 2], dtype=np.float32)
        arr = np.arange(nbdata)
        np.random.shuffle(arr)
        tmpdata = self.data[arr[0],:]
        tmplabel = self.labels[arr[0],:]
        for i in range(nbdata-1):
            self.data[arr[i],:] = self.data[arr[i+1],:]
            self.labels[arr[i],:] = self.labels[arr[i+1],:]
        self.data[arr[nbdata-1],:] = tmpdata
        self.labels[arr[nbdata-1],:] = tmplabel


        self.data, self.test_data = self.data[:int(len(self.data)*splitRatio)], self.data[int(len(self.data)*splitRatio):]
        self.labels, self.test_labels = self.labels[:int(len(self.labels)*splitRatio)], self.labels[int(len(self.labels)*splitRatio):]

        if L2normalize:
            self.data /= np.sqrt(np.expand_dims(np.square(self.data).sum(axis=1), 1))

        class_counts = list(np.argmax(self.labels,1)).count(1)
        print("total members : ", len(self.labels), " class 1 : ",class_counts)

    def NextTrainingBatch(self):
        '''if self.curPos + self.batchSize > self.nbdata:
            self.curPos = 0
        xs = self.data[self.curPos:self.curPos+self.batchSize,:]
        ys = self.labels[self.curPos:self.curPos+self.batchSize,:]
        self.curPos += self.batchSize'''
        xs = list()
        ys = list()
        nb_z, nb_o = 0, 0
        n_sample = round(len(self.data))
        while nb_z < self.batchSize/2 or nb_o < self.batchSize/2:
            index = randrange(n_sample)
            if self.labels[index,0] == 1 and nb_z < self.batchSize/2:
                xs.append(self.data[index])
                ys.append(self.labels[index])
                nb_z+=1
            elif self.labels[index,1] == 1 and nb_o < self.batchSize/2:
                xs.append(self.data[index])
                ys.append(self.labels[index])
                nb_o+=1
        #print(np.array(ys).shape)
        #input()

        return xs,ys
