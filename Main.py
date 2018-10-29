import tensorflow as tf
import numpy as np
import Datasets as ds
import Layers
from matplotlib import pyplot as plt

def get_dict(database):
	xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}

LoadModel = False

experiment_name = 'face_classification'
train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)
#test = ds.DataSet('../DataBases/data_test10k.bin','../DataBases/gender_test10k.bin',10000)

img = train.data[0].reshape((56,56,3)).astype(int)

plt.imshow(img)
plt.show()
