import tensorflow as tf
import numpy as np
import Datasets as ds
import Layers
from matplotlib import pyplot as plt
import pickle


#
#	Ce fichier est utilisé afin d'entraîner le modèle.
#	Ce code a été développé sous un linux, les paths sont peut-etre a modifier sous windows.
#

def get_dict(database):
	xs,ys = database.NextTrainingBatch_resample()
	#xs,ys = database.NextTrainingBatch()
	return {x:xs,y_desired:ys}

LoadModel = False

experiment_name = 'face_classification'
#On charge les données depuis les .raw, en retournant un objet dataset
train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)

#On transforme la première image en vecteur numpy de dimensions (56,56,3)
#Puis on utilise matplotlib afin d'afficher l'image
img = train.data[0].reshape((56,56,3)).astype(int)
plt.imshow(img)
plt.show()

#Calcule la précision sur un ensemble de batchs OU une base de tests.
def mean_accuracy(TFsession,loc_acc, train, x_loc, y_loc, TRAIN):
		acc = 0
		if TRAIN:
			for i in range(0, len(train.data), train.batchSize):
				curBatchSize = min(train.batchSize, len(train.data)-i)
				dict = {x_loc:train.data[i:i+curBatchSize,:],y_loc:train.labels[i:i+curBatchSize,:]}
				acc += TFsession.run(loc_acc, dict) * curBatchSize
			acc /= len(train.data)
		else:
			for i in range(0, len(train.test_data), train.batchSize):
				curBatchSize = min(train.batchSize, len(train.test_data)-i)
				dict = {x_loc:train.test_data[i:i+curBatchSize,:],y_loc:train.test_labels[i:i+curBatchSize,:]}
				acc += TFsession.run(loc_acc, dict) * curBatchSize
			acc /= len(train.test_data)
		return acc

with tf.name_scope('input'):
	# Défini une liste d'images que le modèle prends en entrée. Format (9408,1) pour chaque image.
	x = tf.placeholder(tf.float32, [None, train.dim],name='x')
	# La sortie attendue est un one-hot vecteur pour chaque classe. format [0., 1.] pour chaque élément
	y_desired = tf.placeholder(tf.float32, [None, 2],name='y_desired')

with tf.name_scope('CNN'):
	#Transforme le vecteur de (9408,1) à (56,56,3) pour chaque img du batch
	t = Layers.unflat(x,56,56,3)
	#Convolutions par une matrice de taille (3,3) avec un pas de 1
	t = Layers.conv(t,3,3,1,'conv_1')
	#Maxpooling afin de réduire la dimension de l'image. Divise par 2 les dimensions.
	t = Layers.maxpool(t,2,'pool_2')
	t = Layers.conv(t,6,3,1,'conv_3')
	t = Layers.maxpool(t,2,'pool_4')
	t = Layers.conv(t,12,3,1,'conv_5')
	t = Layers.maxpool(t,2,'pool_6')
	# Applatie l'image en un vecteur.
	t = Layers.flat(t)
	# Dense afin d'obtenir un vecteur des classes.
	y = Layers.fc(t,2,'fc_7',tf.nn.log_softmax)

with tf.name_scope('cross_entropy'):
	#Calcule la différence entre la prédiction et les classes réelles.
	diff = y_desired * y
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.summary.scalar('cross entropy', cross_entropy)

with tf.name_scope('accuracy'):
	#Calcule la précision du modèle: effectif d'échantillons correctement classifiés.
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('confusion'):
	#Calcule une matrice de confusion, afin d'observer graphiquement les erreurs de classification
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_desired, 1))
	with tf.name_scope('confusion'):
		cm = tf.confusion_matrix(labels=tf.argmax(y_desired, 1), predictions=tf.argmax(y, 1), num_classes=2)

with tf.name_scope('learning_rate'):
	#Défini la quantité variation des poids lors de l'apprentissage.
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(1e-3,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate', learning_rate)

#On utilise l'algorithme Adam, un des optimiseur les plus efficaces disponibles.
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
#Aggrégation des variables afin d'output une présentation claire du CNN
merged = tf.summary.merge_all()

Acc_Train = tf.placeholder("float", name='Acc_Train');
Acc_Test = tf.placeholder("float", name='Acc_Test');
MeanAcc_summary = tf.summary.merge([tf.summary.scalar('Acc_Train', Acc_Train),tf.summary.scalar('Acc_Test', Acc_Test)])



print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")

sess = tf.Session()
#On initialise les variables.
sess.run(tf.global_variables_initializer())
#Utilitaire afin d'écrire les résumés destinés à tensorboard
writer = tf.summary.FileWriter(experiment_name, sess.graph)
#Utilitaire afin de sauvegarder le modèle après entraînement.
saver = tf.train.Saver()
#Permet de charger un modèle entraîné, si on le souhaite.
if LoadModel:
	saver.restore(sess, "./save/modelCNN.ckpt")

#Nous utilisons un oversampling de la base d'entraînement. Il est donc équivalent de définir un nombre
# subjectif de batchs sur lesquels apprendre qu'un nombre d'itérations.
# Entraîne le modèle sur 10000 batchs de 256 images.
nbIt = 10000
for it in range(nbIt):
	#On récupère le 1er batch généré sur demande.
	trainDict = get_dict(train)
	#On entraîne le modèle
	sess.run(train_step, feed_dict=trainDict)

	#Calcule de métriques afin de déterminer l'apprentissage du modèle.
	if it%100 == 0:
		acc,ce,lr = sess.run([accuracy,cross_entropy,learning_rate], feed_dict=trainDict)
		print ("it= %6d - rate= %f - cross_entropy= %f - acc= %f" % (it,lr,ce,acc ))
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)

	#On applique les métriques sur une base de test régulièrement, afin de vérifier l'apprentissage
	# sur une base inconnue.
	if it%1000 == 0:
		Acc_Train_value = mean_accuracy(sess,accuracy,train,x, y_desired, True)
		Acc_Test_value = mean_accuracy(sess,accuracy,train, x, y_desired, False)
		print ("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		conf = sess.run(cm, feed_dict={x:train.test_data, y_desired:train.test_labels})
		print("confusion matrix: ")
		print(conf)


writer.close()
#Sauvegarde du modèle entraîné.
if not LoadModel:
	saver.save(sess, "./save/modelCNN.ckpt")
sess.close()
