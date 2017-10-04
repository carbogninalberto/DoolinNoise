# As I said earlier, TensorFlow follows a lazy approach. The usual workflow of running a program in TensorFlow is as follows:
# 1 Build a computational graph, this can be any mathematical operation TensorFlow supports.
# 2 Initialize variables, to compile the variables defined previously
# 3 Create session, this is where the magic starts!
# 4 Run graph in session, the compiled graph is passed to the session, which starts its execution.
# 5 Close session, shutdown the session.
#
#
# placeholder: A way to feed data into the graphs
# feed_dict: A dictionary to pass numeric values to computational graph
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#import tensorflow
import tensorflow as tf
#import modules
import numpy as np
import pandas as pd
import urllib.request
from numpy import genfromtxt

#To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)

#set directiory path for safekeeping
root_dir = os.path.abspath('../..')
data_dir = os.path.join(root_dir, 'data')
sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

#data elaboration
mydata = genfromtxt('data.csv', delimiter=',')
mydata = np.delete(mydata, 0, 1)
mydata = np.delete(mydata, 4, 1)
nrow = mydata.shape[0]-1
outputArr = np.delete(mydata, nrow, 0)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
outputArr = np.delete(outputArr, 0, 1)
mydata = np.delete(mydata, 0, 0)

mydata = np.array(mydata, dtype=float)
outputArr = np.array(outputArr, dtype=float)

# Process to Normalize
X = mydata
y = outputArr
# Normalize
recover = np.amax(y)-np.amin(y)
min_sum = np.amin(y)
X = (X-np.amin(X))/(np.amax(X)-np.amin(X))
Y = (y-np.amin(y))/(np.amax(y)-np.amin(y))

print (X)
print (Y)

## set all ANN variables

# number of neurons in each layer
input_num_units = 4
hidden_num_units = 20000
output_num_units = 1

#define placeholders
x = tf.placeholder(tf.float32, [1, input_num_units])
y = tf.placeholder(tf.float32, [1, output_num_units])

#set remaining variables
epochs = 500
batch_size = 1
learning_rate = 0.0001

## define weights and biases
weights = {
    'hidden': tf.Variable(tf.truncated_normal([input_num_units, hidden_num_units], stddev=0.0001)),
    'output': tf.Variable(tf.truncated_normal([hidden_num_units, output_num_units], stddev=0.0001))
}

biases = {
    'hidden': tf.Variable(tf.ones([hidden_num_units])),
    'output': tf.Variable(tf.ones([output_num_units]))
}

#neural network computational graph
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']

#neural network cost
#model = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y)

#cost = 0.5*sum((y-output_layer)**2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
#cost = -tf.reduce_sum(y*tf.log(output_layer))
#set optimizer (ex. backpropogation algorithm)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

##initialize variables
init = tf.global_variables_initializer()
print("weights: ", weights['hidden'])

with tf.Session() as sess:
    #create initialized variables
    sess.run(init)

    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize

    for epoch in range(epochs):
        avg_cost = 0
        total_batch = mydata.shape[0]
        #print(mydata.shape)
        for i in range(int(total_batch)):
            x_tmp = np.array([X[i]], dtype=np.float32)#.transpose()
            y_tmp = np.array([Y[i]], dtype=np.float32)#.transpose()
            #print(x_tmp)
            #x_tmp, y_tmp = tf.train.batch(batch_size=batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: x_tmp, y: y_tmp})
            #print(sess.run([optimizer, cost], feed_dict={x: x_tmp, y: y_tmp}))

            #print("{:.10f}".format(c))
            avg_cost += c/total_batch
        print("Epoch: ", (epoch+1), " cost = ", "{:.5f}".format(avg_cost))

    print("\nTraining Complete! cost: ", avg_cost)

    predict = tf.argmax(output_layer, 1)
    x_tmp = np.array([X[0]], dtype=np.float32)
    print(x_tmp)
    print(predict.eval({x: x_tmp})*1.000000)
    print(sess.run(output_layer, feed_dict={x: x_tmp})*recover+min_sum)



#build computational graph
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

addition = tf.add(a, b)

#initialize variables
init = tf.global_variables_initializer()

#create session and run the graph

with tf.Session() as sess:
    sess.run(init)
    print("addition: %i" % sess.run(addition, feed_dict={a: 2, b: 3}))

#close session
sess.close()
