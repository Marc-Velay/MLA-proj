import GAN_layers
import Datasets as ds
from sklearn.utils import class_weight
import numpy as np

TRAIN = True
model_filename = 'save/gan.h5'

experiment_name = 'face_classification-gan'
train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)
'''
train.data = train.data.reshape(len(train.data), 56,56,3)
train.val_data = train.val_data.reshape(len(train.val_data), 56,56,3)
train.test_data = train.test_data.reshape(len(train.test_data), 56,56,3)
'''
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.labels[:,1]),
                                                 train.labels[:,1])

print(train.data[0].shape)
print(train.labels[0])



X_train = (train.data.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)


gan = GAN_layers.GAN()
gan.train(X_train)
