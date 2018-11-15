
import numpy as np
from collections import Counter
from random import randrange


class DataSet(object):
    def __init__(self, filename_data, filename_labels, nbdata, batchSize=128, splitRatio=0.7):
        self.nbdata = nbdata
        # taille des images 56*56 pixels en couleurs RBG
        self.dim = 9408
        self.imgSize = 56
        self.data = None
        self.labels = None
        self.val_data = None
        self.val_labels = None
        self.test_data = None
        self.test_labels = None
        self.batchSize = batchSize
        self.curPos = 0
        self.x = None
        self.y_desired = None

        f = open(filename_data, 'rb')
        self.data = np.empty([nbdata, self.dim], dtype=np.float32)
        #Lecture des bytes block par block, de la taille de l'image.
        for i in range(nbdata):
            self.data[i,:] = np.fromfile(f, dtype=np.uint8, count=self.dim)
        f.close()

        f = open(filename_labels, 'r')
        self.labels = np.empty([nbdata, 2], dtype=np.int)
        for i in range(nbdata):
            line=int(f.readline())
            # Les labels sont de la forme [0., 1.]. On attribue donc la valeur 1 à la colonne de la classe.
            self.labels[i,line] = 1
        f.close()

        print ('nb data : ', self.nbdata)

        tmpdata = np.empty([1, self.dim], dtype=np.float32)
        tmplabel = np.empty([1, 2], dtype=np.float32)
        #On mélange toutes les données, afin de ne pas obtenir le même ordre à chaque execution
        arr = np.arange(nbdata)
        np.random.shuffle(arr)
        tmpdata = self.data[arr[0],:]
        tmplabel = self.labels[arr[0],:]
        for i in range(nbdata-1):
            self.data[arr[i],:] = self.data[arr[i+1],:]
            self.labels[arr[i],:] = self.labels[arr[i+1],:]
        self.data[arr[nbdata-1],:] = tmpdata
        self.labels[arr[nbdata-1],:] = tmplabel

        #Creation de la base de test
        self.data, self.test_data = self.data[:int(len(self.data)*splitRatio)], self.data[int(len(self.data)*splitRatio):]
        self.labels, self.test_labels = self.labels[:int(len(self.labels)*splitRatio)], self.labels[int(len(self.labels)*splitRatio):]

        #Creation de la base de validation
        self.data, self.val_data = self.data[:int(len(self.data)*0.8)], self.data[int(len(self.data)*0.8):]
        self.labels, self.val_labels = self.labels[:int(len(self.labels)*0.8)], self.labels[int(len(self.labels)*0.8):]

        class_counts = list(np.argmax(self.labels,1)).count(1)
        print("total members : ", len(self.labels), " class 1 : ",class_counts)

    def NextTrainingBatch_resample(self):
        #Le but de cette fonction est d'obtenir un batch équilibré.
        #On tire donc au hasard des images tant que les deux classes n'ont pas
        #chacune batch/2 échantillons.
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

    def NextTrainingBatch(self):
        if self.curPos + self.batchSize > self.nbdata:
            self.curPos = 0
        xs = self.data[self.curPos:self.curPos+self.batchSize,:]
        ys = self.labels[self.curPos:self.curPos+self.batchSize,:]
        self.curPos += self.batchSize

        return xs,ys
