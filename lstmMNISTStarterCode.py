import tensorflow as tf 
from tensorflow.python.ops import rnn
import numpy as np 

from tensorflow.examples.tutorials.mnist import input_data

print('Importing MNIST Data...')
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # call mnist function
print('Data imported')

learningRate = 1e-4
trainingIters = 1000000
batchSize = 150
displayStep = 10

nInput = 28 # 28 pixels in each row
nSteps = 28 # 28 rows of pixels
nHidden = 10 #number of neurons for the RNN
nClasses = 10 # 10 image classes in MNIST

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	x = tf.split(x, nSteps, 0) # configuring so you can get it as needed for the 28 pixels

	lstmCell = tf.contrib.rnn.BasicRNNCell(nHidden) # parameter is the number of units in the LSTM cell

	outputs, states = tf.contrib.rnn.static_rnn(lstmCell, x, dtype=tf.float32) #for the rnn where to get the output and hidden state

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learningRate,0.01).minimize(cost)

correctPred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss_summary = tf.summary.scalar('Loss', cost)
test_summary = tf.summary.scalar('Test Validation', accuracy)
validation_summary = tf.summary.scalar('Cross Validation', accuracy)

# Instantiate a SummaryWriter to output summaries and the Graph.
result_dir1 = './results/graph' # directory where the results from the training are saved
result_dir2 = './results/other'  # directory where the results from the training are saved
summary_writer = tf.summary.FileWriter(result_dir1, tf.Session().graph)
other_writer = tf.summary.FileWriter(result_dir2)

init = tf.initialize_all_variables()

step = 1

with tf.Session() as sess:
	sess.run(init)
	print('Session initialized')

	trainData = mnist.train.images.reshape((-1, nSteps, nInput))
	trainLabel = mnist.train.labels
	testData = mnist.test.images.reshape((-1, nSteps, nInput))
	testLabel = mnist.test.labels

	feed_dict_train = {x: trainData, y: trainLabel}
	feed_dict_test = {x: testData, y: testLabel}

	while step * batchSize < trainingIters:
		batch = mnist.train.next_batch(batchSize)
		batchX, batchY = batch[0], batch[1]
		batchX = batchX.reshape((batchSize, nSteps, nInput))
		batch_dict = {x: batchX, y: batchY}

		sess.run(optimizer, feed_dict={x:batchX, y:batchY}) # do I need keep_prob here?

		# Batch accuracy and loss
		if step % displayStep == 0:
			acc = accuracy.eval(feed_dict=batch_dict)
			loss = cost.eval(feed_dict=batch_dict)
			print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + str(loss)+ ", Training Accuracy= " + str(acc))

		if step % 100 == 0:
			# To log training accuracy.
			train_acc, train_summ = sess.run([accuracy, test_summary], feed_dict=feed_dict_train)
			other_writer.add_summary(train_summ, step)

			# To log validation accuracy.
			valid_acc, valid_summ = sess.run([accuracy, validation_summary], feed_dict=feed_dict_test)
			other_writer.add_summary(valid_summ, step)

			# To log loss
			loss, loss_summ = sess.run([cost, loss_summary], feed_dict=batch_dict)
			other_writer.add_summary(loss_summ, step)
			other_writer.flush()
			print('training and validation accuracy logged')

		step += 1

	print('Optimization finished')
	print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData, y: testLabel}))
