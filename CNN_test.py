import tensorflow as tf
import numpy as np
import Datasets as ds
import Layers
from matplotlib import pyplot as plt
import pickle

def get_dict(database):
	xs = database.NextTrainingBatch()
	return {x:xs}

LoadModel = True

experiment_name = 'face_classification'
#train = ds.DataSet('./data/db_train.raw','./data/label_train.txt',111430, splitRatio=0.9)
test = ds.DataSet('../DataBases/db_test.raw',10130)

#img = train.data[0].reshape((56,56,3)).astype(int)
#plt.imshow(img)
#plt.show()


with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, [None, train.dim],name='x')
	y_desired = tf.placeholder(tf.float32, [None, 2],name='y_desired')

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


print ("-----------------------------------------------------")
print ("----------- GENERATING PREDICTION.TXT")
print ("-----------------------------------------------------")

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./save/CNN1_model.ckpt")

for item in range(ds.nbdata):
	trainDict = get_dict(train)
	sess.run(train_step, feed_dict=trainDict)

	if it%10 == 0:
		acc,ce,lr = sess.run([accuracy,cross_entropy,learning_rate], feed_dict=trainDict)
		print ("it= %6d - rate= %f - cross_entropy= %f - acc= %f" % (it,lr,ce,acc ))
		summary_merged = sess.run(merged, feed_dict=trainDict)
		writer.add_summary(summary_merged, it)

	if it%100 == 50:
		Acc_Train_value = mean_accuracy(sess,accuracy,train,x, y_desired, True)
		Acc_Test_value = mean_accuracy(sess,accuracy,train, x, y_desired, False)
		print ("mean accuracy train = %f  test = %f" % (Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)
		conf = sess.run(cm, feed_dict={x:train.test_data, y_desired:train.test_labels})
		print("confusion matrix: ")
		print(conf)


writer.close()
sess.close()
