from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, LeakyReLU
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.utils import class_weight
import numpy as np
import Datasets as ds
import Layers
from matplotlib import pyplot as plt

#
# Ce fichier est utilisé pour entraîner un CNN avec la librairie Keras.
#

TRAIN = True
model_filename = 'save/gan.h5'

experiment_name = 'face_classification-gan'
train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)

#transforme un vecteur en 3 matrices RGB de 56x56px
train.data = train.data.reshape(len(train.data), 56,56,3)
train.val_data = train.val_data.reshape(len(train.val_data), 56,56,3)
train.test_data = train.test_data.reshape(len(train.test_data), 56,56,3)

#On attribue à chaque classe un poids afin d'équilibrer leur importance, basé sur leur fréquence respective dans la distribution
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train.labels[:,1]),
                                                 train.labels[:,1])


#On utilise un modèle séquentiel car un seul mode et un seul vecteur cible.
#TODO: Implémenter avec l'api fonctionnelle. Permettrai un fully connected CNN.
model = Sequential()
#Couche de convolution
model.add(Conv2D(128, kernel_size=3, input_shape=(56,56,3)))
#Activation leaky relu: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
model.add(LeakyReLU(alpha=0.2))
#Réduction de la taille des matrices
model.add(MaxPooling2D(pool_size=(2,2)))
#Réinitialise des poids au hasard, afin d'éviter l'overfitting
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=3))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(128, kernel_size=3))
model.add(LeakyReLU(alpha=0.2))
model.add(Flatten())
#Attention layer
model.add(Dense(32, activation='softmax'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



if TRAIN:

    #Sauvegarde le modèle avec la plus haute précision
	checkpoint = ModelCheckpoint(model_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]

	model.fit(train.data, train.labels, validation_data=(train.val_data, train.val_labels),
				epochs=2, class_weight=class_weights, callbacks=callbacks_list, batch_size=train.batchSize)

    #On prédit toutes les valeures de la base de test, afin de générer des métriques
	predictions = model.predict(train.test_data)
	matrix = metrics.confusion_matrix(train.test_labels.argmax(axis=1), predictions.argmax(axis=1))
	print(matrix)
