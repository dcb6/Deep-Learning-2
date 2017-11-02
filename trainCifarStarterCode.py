from scipy import misc
import numpy as np
import tensorflow as tf

# import random
# import matplotlib.pyplot as plt
# import matplotlib as mp

def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)

    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)

    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_max

ntrain = 1000  # per class
ntest = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 500

Train = np.zeros((ntrain * nclass, imsize, imsize, nchannels))
Test = np.zeros((ntest * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((ntrain * nclass, nclass))
LTest = np.zeros((ntest * nclass, nclass))

itrain = -1
itest = -1

for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itrain += 1
        Train[itrain, :, :, 0] = im
        LTrain[itrain, iclass] = 1  # 1-hot label
    for isample in range(0, ntest):
        path = './CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        im = misc.imread(path);  # 28 by 28
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :, 0] = im
        LTest[itest, iclass] = 1  # 1-hot label

print("Images loaded.")

sess = tf.InteractiveSession()

# placeholders for input data and input labels
tf_data = tf.placeholder(tf.float32,shape=[None,imsize,imsize,nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels]
tf_labels = tf.placeholder(tf.float32,shape=[None,nclass]) #tf variable for labels

# reshape the input image
x_image = tf.reshape(tf_data, [-1, 28, 28, 1])

# --------------------------------------------------
# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32]) # width height numchannels numfilters
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # input 7*7*64, output 1024
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, nclass])
b_fc2 = bias_variable([nclass])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# create summaries
summary_items = [W_conv1,b_conv1,conv2d(x_image, W_conv1) + b_conv1,h_conv1,h_pool1,
    W_conv2,b_conv2,conv2d(h_pool1, W_conv2) + b_conv2,h_conv2,h_pool2,
    W_fc1,b_fc1,tf.matmul(h_pool2_flat, W_fc1) + b_fc1,h_fc1,
    W_fc2,b_fc2,tf.matmul(h_fc1_drop, W_fc2) + b_fc2,y_conv]

summary_titles = ['Conv_1_Weights','Conv_1_Bias','Conv_1_Input','Conv_1_Activations_post-Tanh','Conv_1_Activations_post-Pooling',
        'Conv_2_Weights','Conv_2_Bias','Conv_2_Input','Conv_2_Activations_post-Tanh','Conv_2_Activations_post-Pooling',
        'Dense_1_Weights','Dense_1_Bias','Dense_1_Inputs','Dense_1_Activations',
        'Softmax_Weights','Softmax_Bias','Softmax_Inputs','Softmax_Activations']

for i in range(len(summary_items)):
    variable_summaries(summary_items[i],summary_titles[i])

# --------------------------------------------------
# setup training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
optimizer = tf.train.MomentumOptimizer(1e-4, 0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # Add a scalar summary for the snapshot loss.
tf.summary.scalar('Cross Entropy', cross_entropy)
#
# # Build the summary operation based on the TF collection of Summaries.
tf.summary.image('Conv1 Weights Image', tf.transpose(W_conv1,[3,1,0,2]),32)
summary_op = tf.summary.merge_all()
test_summary = tf.summary.scalar('Test Validation', accuracy)
validation_summary = tf.summary.scalar('Cross Validation', accuracy)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
result_dir1 = './results/graph' # directory where the results from the training are saved
result_dir2 = './results/other'  # directory where the results from the training are saved
summary_writer = tf.summary.FileWriter(result_dir1, sess.graph)
other_writer = tf.summary.FileWriter(result_dir2)

# --------------------------------------------------
# optimization

sess.run(tf.global_variables_initializer())
batch_xs = np.zeros([batchsize,imsize,imsize,nchannels]) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize,nclass]) #setup as [batchsize, the how many classes]

nsamples = ntrain * nclass

feed_dict_train = {tf_data: Train, tf_labels: LTrain, keep_prob: 0.5}
feed_dict_test = {tf_data: Test, tf_labels: LTest, keep_prob: 0.5}

for i in range(10000): # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]

    # Batch accuracy
    if i%10 == 0:
        # output the training accuracy every 100 iterations (batch)
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob:0.5})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    # Test/Validation Accuracy
    if i%1000 == 0:
        # Log training accuracy.
        train_acc, train_summ = sess.run([accuracy, test_summary],feed_dict=feed_dict_train)
        other_writer.add_summary(train_summ, i)

        # Log validation accuracy.
        valid_acc, valid_summ = sess.run([accuracy, validation_summary],feed_dict=feed_dict_test)
        other_writer.add_summary(valid_summ, i)
        other_writer.flush()

    optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob:0.5}) # dropout only during training

summary = sess.run(summary_op,feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob:0.5})
other_writer.add_summary(summary)
sess.close()