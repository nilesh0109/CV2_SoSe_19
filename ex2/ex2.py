import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.load('data/linreg_x.npy')
y_data = np.load('data/linreg_y.npy')

print(len(x_data))
print(x_data.shape)

plt.plot(x_data, y_data, 'bo')


###########  Setting up the computational graph  ################

x_placeholder = tf.placeholder(tf.float32, shape=x_data.shape)
y_placeholder = tf.placeholder(tf.float32, shape=y_data.shape)

w = tf.get_variable("w", shape=(x_data.shape[1],1), initializer=tf.random_normal_initializer(0,1))
b = tf.get_variable("b", shape=(x_data.shape[1],1), initializer=tf.random_normal_initializer(0,1))

y_pred = tf.matmul(x_placeholder, w) + b

########### setting up the tensor operations #######################

loss = tf.losses.mean_squared_error(y_placeholder, y_pred)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
minimizer_op = optimizer.minimize(loss)

#adding summary for consoling through tensorboard
l_summary = tf.summary.scalar(name="loss", tensor=loss)
w_summary = tf.summary.histogram(name="weights", values=w)
b_summary = tf.summary.histogram(name="bias", values=b)

########### computing in a session ##########################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for k in range(100):
       _, l, wk, bk = sess.run([minimizer_op, loss, w, b], {x_placeholder: x_data, y_placeholder: y_data})
       print('Iteration {%d}: Loss {%f}, w = {%f}, b = {%f}' % (k,l,wk,bk))
       ls, ws, bs = sess.run([l_summary, w_summary, b_summary], {x_placeholder: x_data, y_placeholder: y_data})

        # Writing summary to tensorboard
       writer = tf.summary.FileWriter('./graphs', sess.graph)
       writer.add_summary(ls, global_step=k) # writes loss summary
       writer.add_summary(ws, global_step=k) # writes weights summary
       writer.add_summary(bs, global_step=k) # writes biases summary

############### visualizing the output #######################

plt.plot(x_data, wk * x_data + bk, 'g')
plt.show()

# use this to launch tensorboard
# python3 -m tensorboard.main --logdir=./graphs