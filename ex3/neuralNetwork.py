import tensorflow as tf
import numpy as np

#Training the Neural Network
def train_network(train_data, train_label, batch_size, learning_rate):
    imgs_placeholder = tf.placeholder(tf.uint8, shape=(None, 32, 32, 3))
    imgs_label_placeholder = tf.placeholder(tf.int64, shape=(None,))
    preprocessed_imgs = tf.image.convert_image_dtype(imgs_placeholder, tf.float32)
    
    ######################### first convolutional layer ################################
    conv1_1out = tf.layers.conv2d(preprocessed_imgs, filters=32, kernel_size=(3,3), activation=tf.nn.relu, name='conv1_1')
    conv1_2out = tf.layers.conv2d(conv1_1out, filters=32, kernel_size=(3,3), activation=tf.nn.relu, name='conv1_2')
    pool1_out = tf.layers.max_pooling2d(conv1_2out, pool_size=(2,2), strides=2, name='pool1')
    
    ######################### Second convolutional layer ################################
    conv2_1out = tf.layers.conv2d(pool1_out, filters=64, kernel_size=(3,3), activation=tf.nn.relu, name='conv2_1')
    conv2_2out = tf.layers.conv2d(conv2_1out, filters=64, kernel_size=(3,3), activation=tf.nn.relu, name='conv2_2')
    pool2_out = tf.layers.max_pooling2d(conv2_2out, pool_size=(2,2), strides=2, name='pool2')
    
    ######################### Fully connected layer ################################
    conv2_1out = tf.layers.flatten(pool2_out, name='flatten')
    logits = tf.layers.dense(conv2_1out, units=10, name='fc1')
    
    ######################### Define Loss function ################################
    loss =  tf.losses.sparse_softmax_cross_entropy(imgs_label_placeholder, logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate= learning_rate)
    minimizer_op = optimizer.minimize(loss)
    
    ######################### Accuracy check ################################
    predictions = tf.argmax(logits, axis=1)
    correct_pred = tf.equal(imgs_label_placeholder, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_pred , tf.float32))
    
    ################## Adding summary for consoling through tensorboard ###############
    loss_summary = tf.summary.scalar(name="loss", tensor=loss) #writes loss summary
    acc_summary = tf.summary.scalar(name="accuracy", tensor=accuracy) #writes accuracy summary
    
    ######################### Start Training ################################
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for k in range(5):
            j = 0

            unseen_data = train_data[:]
            while(len(unseen_data)):
                idx = np.random.choice( unseen_data.shape[0], batch_size, replace=False )
                indices_to_keep = [i for i in range(len(unseen_data)) if i not in idx]
                unseen_data = unseen_data[indices_to_keep]
                j = j + 1
                training_data_batch = train_data[idx]
                train_label_batch = train_label[idx]
                _, l, acc, l_summary, a_summary =sess.run([minimizer_op, loss, accuracy, loss_summary, acc_summary], {imgs_placeholder: training_data_batch, imgs_label_placeholder: train_label_batch})
                writer = tf.summary.FileWriter('./graphs', sess.graph)
                writer.add_summary(l_summary, global_step= j) # writes loss summary
                writer.add_summary(a_summary, global_step=j) # writes loss summary
            
            print('After iteration {} loss is {} and accuracy is {}'.format(k, l, acc))
                
        

    