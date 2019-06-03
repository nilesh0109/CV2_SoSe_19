import numpy as np
import os
import tensorflow as tf
import re
"""
unpickle function from CIFAR-10 website
"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin')
    return d

def create_dataset_from_files(files):
	rawdata = []
	labels = []
	for f in files:
		d = unpickle(f)
		rawdata.extend(d["data"])
		labels.extend(d["labels"])

	rawdata = np.array(rawdata)

	red = rawdata[:,:1024].reshape((-1,32,32))
	green = rawdata[:,1024:2048].reshape((-1,32,32))
	blue = rawdata[:,2048:].reshape((-1,32,32))

	data = np.stack((red,green,blue), axis=3)
	labels = np.array(labels)

	return data, labels

cifar_data_path = 'data/cifar10/cifar-10-batches-py/' # replace with your path
cifar_training_files = [os.path.join(cifar_data_path, 'data_batch_{:d}'.format(i)) for i in range(1,6)]
cifar_testing_files = [os.path.join(cifar_data_path, 'test_batch')]

train_data, train_labels = create_dataset_from_files(cifar_training_files)
test_data, test_labels = create_dataset_from_files(cifar_testing_files)

vgg_weight_file = 'Model/VGG16/vgg16-conv-weights.npz'

# Load VGG16 weights from file
weights = np.load(vgg_weight_file)

# Set up net
batchsize = 128
input_images = tf.placeholder(tf.uint8, [None, 32, 32, 3])
labels = tf.placeholder(tf.int64, [None, ])

# As preprocessing, the average RGB value must be subtracted.
with tf.name_scope('preprocess') as scope:
	imgs = tf.image.convert_image_dtype(input_images, tf.float32) * 255.0
	mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
	imgs_normalized = imgs - mean

with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(imgs_normalized, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv1_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_2_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool1') as scope:
	pool = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv2_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv2_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv2_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv2_2_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool2') as scope:
	pool = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(pool, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_2') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_2_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_2_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('conv3_3') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_3_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_3_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(act, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)

with tf.name_scope('pool3') as scope:
	pool = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

# Our own trained layer
with tf.name_scope('fc1') as scope:
	flat = tf.layers.flatten(pool)
	logits = tf.layers.dense(flat, units=10)

with tf.name_scope('loss') as scope:
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
	minimize_op = optimizer.minimize(loss)

# Statistics
predictions = tf.argmax(logits, axis=1)
correct_prediction = tf.equal(labels, predictions)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Summaries
loss_summary = tf.summary.scalar(name="loss", tensor=loss)
accuracy_summary = tf.summary.scalar(name="accuracy", tensor=accuracy)

# for saving checkpoints
saver = tf.train.Saver()
with tf.Session() as sess:
	writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
	sess.run(tf.global_variables_initializer())
	# Instead of running initializer, we could load a checkpoint:
	#saver.restore(sess, './my-model-299')
	# for batch in range(300):
	# 	idx = np.random.choice(train_data.shape[0], batchsize, replace=False) # sample random indices
	# 	_, l, a, batch_acc, batch_loss = sess.run([minimize_op, loss_summary, accuracy_summary, accuracy, loss], 
	# 		feed_dict={input_images: train_data[idx,...], labels: train_labels[idx]})

	# 	writer.add_summary(l, global_step=batch)
	# 	writer.add_summary(a, global_step=batch)
	# 	if batch % 100 == 0:
	# 		print('Batch {:d} done: batch loss {:f}, batch accuracy {:f}'.format(batch, batch_loss, batch_acc))


	# run testing in smaller batches so we don't run out of memory.
	test_batch_size = 100
	num_test_batches = int(test_data.shape[0]/test_batch_size)
	test_losses = []
	test_accs = []
	for test_batch in range(num_test_batches):
		start_idx = test_batch * test_batch_size
		stop_idx = start_idx + test_batch_size
		test_idx = np.arange(start_idx, stop_idx)

		test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={input_images: test_data[test_idx], labels: test_labels[test_idx]})
		print('Test batch {:d} done: batch loss {:f}, batch accuracy {:f}'.format(test_batch, test_loss, test_accuracy))
		test_losses.append(test_loss)
		test_accs.append(test_accuracy)

	print('Test loss: {:f} -- test accuracy: {:f}'.format(np.average(test_losses), np.average(test_accs)))

	# we could save at end of training, for example
	save_path = saver.save(sess, "./my-model", global_step=batch)  