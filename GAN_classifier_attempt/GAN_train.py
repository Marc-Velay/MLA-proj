import GAN_layers
import Datasets as ds
from sklearn.utils import class_weight
import numpy as np

TRAIN = True
model_filename = 'save/gan.h5'

experiment_name = 'face_classification-gan'
train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)

train.data = train.data.reshape(len(train.data), 56,56,3)
train.val_data = train.val_data.reshape(len(train.val_data), 56,56,3)
train.test_data = train.test_data.reshape(len(train.test_data), 56,56,3)
'''
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.labels[:,1]),
                                                 train.labels[:,1])
'''


X_train = (train.data.astype(np.float32)) / 255

gan = GAN_layers.GAN()
gan.train(X_train, train.labels, epochs=4000, batch_size=32, save_interval=500)
#gan.train(X_train, train.labels)
