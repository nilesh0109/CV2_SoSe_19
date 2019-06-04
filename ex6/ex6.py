import numpy as np
import os
import tensorflow as tf
import imageio

IN_TRAINING_MODE = True

def load_train_data():
	training_img_directory = 'data/train/images'
	training_fixation_directory = 'data/train/fixations'

	train_imgs = np.zeros((1200, 180, 320, 3), dtype=np.uint8)
	train_fixations = np.zeros((1200, 180, 320, 1), dtype=np.uint8)
	for i in range(1, 1201):
		img_file = os.path.join(training_img_directory, '{:04d}.jpg'.format(i))
		fixation_file = os.path.join(training_fixation_directory, '{:04d}.jpg'.format(i))
		train_imgs[i-1] = imageio.imread(img_file)
		fixation = imageio.imread(fixation_file)
		train_fixations[i-1] = np.expand_dims(fixation, -1) # adds singleton dimension so fixation size is (180,320,1)
	
	return train_imgs, train_fixations

# Generator function will output one (image, target) tuple at a time,
# and shuffle the data for each new epoch
def data_generator(imgs, targets):
	while True: # produce new epochs forever
		# Shuffle the data for this epoch
		idx = np.arange(imgs.shape[0])
		np.random.shuffle(idx)

		imgs = imgs[idx]
		targets = targets[idx]
		for i in range(imgs.shape[0]):
			yield imgs[i], targets[i]

def get_batch_from_generator(gen, batchsize):
	batch_imgs = []
	batch_fixations = []
	for i in range(batchsize):
		img, target = gen.__next__()
		batch_imgs.append(img)
		batch_fixations.append(target)
	return np.array(batch_imgs), np.array(batch_fixations)

# Set up
num_batches = 100 # you can experiment with this value, but remember training a large network requires a lot of iterations!
batchsize = 64 # define your batch size

# load entire data to memory (this dataset is small, so we can do it)
train_imgs, train_fixations = load_train_data()

###
### At minimum, add preprocessing to convert image and target to tf.float32.
### Then enter your CNN definition.
### Name the target fixation map after preprocessing "fixations_normalized", 
### and name the output "saliency_raw" so they fit the code afterward.
### 


vgg_weight_file = 'Model/VGG16/vgg16-conv-weights.npz'

# Load VGG16 weights from file
weights = np.load(vgg_weight_file)
# for k in weights.keys():
# 	print(k+'shape : {0}'.format(weights[k].shape))
# Set up net
batchsize = 128
input_images_placeholder = tf.placeholder(tf.uint8, [None, 180, 320, 3])
target_images_placehodler = tf.placeholder(tf.int64, [None, 180, 320, 1])

with tf.name_scope('preprocessing') as scope:
	input_imgs = tf.image.convert_image_dtype(input_images_placeholder, tf.float32) * 255.0
	fixations_normalized = tf.image.convert_image_dtype(target_images_placehodler, tf.float32) * 255.0

with tf.name_scope('conv1_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv1_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv1_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(input_imgs, kernel, [1, 1, 1, 1], padding='SAME')
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
	pool2 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(2,2), padding='same')

with tf.name_scope('conv3_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv3_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv3_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
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
	pool3 = tf.layers.max_pooling2d(act, pool_size=(2,2), strides=(1,1), padding='same')

with tf.name_scope('conv4_1') as scope:
	kernel = tf.Variable(initial_value=weights['conv4_1_W'], trainable=False, name="weights")
	biases = tf.Variable(initial_value=weights['conv4_1_b'], trainable=False, name="biases")
	conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
	out = tf.nn.bias_add(conv, biases)
	act = tf.nn.relu(out, name=scope)
	conv4_1 = act



with tf.name_scope('concat_featuremaps') as scope:
	concatenated_feature_maps = tf.concat([pool2, pool3, conv4_1], axis=3)
	regularized_feature_maps = tf.layers.dropout(concatenated_feature_maps, rate=0.5, training= IN_TRAINING_MODE)

with tf.name_scope('featuremaps_conv1') as scope:
	act_featuremaps_conv1 = tf.layers.conv2d(regularized_feature_maps, filters=64, kernel_size=(3,3), activation=tf.nn.relu, name='featuremaps_conv1')

with tf.name_scope('featuremaps_conv2') as scope:
	saliency_raw = tf.layers.conv2d(act_featuremaps_conv1, filters=1, kernel_size=(1,1), activation=tf.nn.relu, name='featuremaps_conv2')

with tf.name_scope('loss') as scope:
	# normalize saliency
	max_value_per_image = tf.reduce_max(saliency_raw, axis=[1,2,3], keepdims=True)
	predicted_saliency = (saliency_raw / max_value_per_image)
	#print('predicted_saliency shape is ', predicted_saliency.shape)
	# Prediction is smaller than target, so downscale target to same size
	target_shape = predicted_saliency.shape[1:3]
	target_downscaled = tf.image.resize_images(fixations_normalized, target_shape)
	
	#print('fixations_normalized shape is ', fixations_normalized.shape)
	
	# Loss function from Cornia et al. (2016) [with higher weight for salient pixels]
	alpha = 1.01
	weights = 1.0 / (alpha - target_downscaled)
	loss = tf.losses.mean_squared_error(labels=target_downscaled, 
										predictions=predicted_saliency, 
										weights=weights)
	
	# Optimizer settings from Cornia et al. (2016) [except for decay]
	optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True)
	minimize_op = optimizer.minimize(loss)
loss_summary = tf.summary.scalar(name="loss", tensor=loss)

with tf.Session() as sess:
	writer = tf.summary.FileWriter(logdir="./", graph=sess.graph)
	sess.run(tf.global_variables_initializer())

	gen = data_generator(train_imgs, train_fixations)

	for b in range(num_batches):
		batch_imgs, batch_fixations = get_batch_from_generator(gen, batchsize)
		batch_loss,l_summary  = sess.run([loss, loss_summary], feed_dict={input_images_placeholder: batch_imgs, target_images_placehodler: batch_fixations})
		writer.add_summary(l_summary, global_step=b)
		if b % 10 == 0:
			print('Batch {:d} done: batch loss {:f}'.format(b, batch_loss))

