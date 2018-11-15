import tensorflow as tf
import numpy as np
import test_data_utils as ds
import Layers
from matplotlib import pyplot as plt
import pickle

def get_dict(database):
	xs = database.GetTestBase()
	return {x:xs}

LoadModel = True

experiment_name = 'face_classification'
print("------------- loading data")
#train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)
test = ds.DataSet('./data/db_test.raw',10130)


img = test.data[0].reshape((56,56,3)).astype(int)
plt.imshow(img)
plt.show()

print("------------- defining CNN")

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, test.dim],name='x')

with tf.name_scope('CNN'):
	t = Layers.unflat(x,56,56,3)
	t = Layers.conv(t,3,3,1,'conv_1')
	t = Layers.maxpool(t,2,'pool_2')
	t = Layers.conv(t,6,3,1,'conv_3')
	t = Layers.maxpool(t,2,'pool_4')
	t = Layers.conv(t,12,3,1,'conv_5')
	t = Layers.maxpool(t,2,'pool_6')
	t = Layers.flat(t)
	y = Layers.fc(t,2,'fc_7',tf.nn.log_softmax)

with tf.name_scope('predict'):
	predictions = tf.argmax(y,1)

print ("-----------------------------------------------------")
print ("------------- GENERATING PREDICTION.TXT")
print ("-----------------------------------------------------")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./save/modelCNN.ckpt")

	trainDict = get_dict(test)
	predictions = sess.run(predictions, feed_dict=trainDict)
	print("------------- created",len(predictions), "predictions")
	print("------------- writing to file")
	with open("predictions.txt", "w") as f:
		for line in predictions:
			f.write(str(line))
	print("------------- DONE")
sess.close()
