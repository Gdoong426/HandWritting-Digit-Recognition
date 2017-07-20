import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import csv
import math
import cv2

with open("RNN_ori.csv") as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    data = list(spamreader)


# import testing data set for further prediction
with open("RNN_test.csv") as testfile:
    spamreader = csv.reader(testfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    test_ka = list(spamreader)

for i in range(len(test_ka)):
    test_ka[i] = test_ka[i][1:]

num = int(len(data)/2)
xy = [0 for x in range(2)]
number = []
number_ka = []  

for i in range(num):
    xy[0] = data[i*2][:]
    xy[1] = data[i*2+1][:]
    number.append(list(np.reshape(np.asarray(xy), len(data[i*2])*2, order = 'F')))

for i in range(int(len(test_ka)/2)):
    xy[0] = test_ka[i*2][:]
    xy[1] = test_ka[i*2+1][:]
    number_ka.append(list(np.reshape(np.asarray(xy), len(test_ka[i*2])*2, order = 'F')))

from random import shuffle
shuffle(number)

label = [i[0] for i in number]
number_ls = [i[2:] for i in number]

max_length_train = max(len(row) for row in number_ls)
max_length_ka = max(len(row) for row in number_ka)
if max_length_ka < max_length_train:
    max_length = max_length_train

elif max_length_ka >= max_length_train:
    max_length = max_length_ka

number_np = np.array([row + [0]* (max_length - len(row)) for row in number_ls])
number_ka_e = np.array([row + [0]* (max_length - len(row)) for row in number_ka])


number_np = np.array(number_np, dtype = int)
number_ka_ts = np.array(number_ka_e, dtype=int)


number_3D = []
number_3D_ka = []
for array in number_np:
    number_3D.append(array.reshape(2, int(max_length/2), order = 'F'))

for array in number_ka_ts:
    number_3D_ka.append(array.reshape(2, int(max_length/2), order = 'F'))

number_3D = np.array(number_3D)
number_3D_ka = np.array(number_3D_ka)


label_np = np.array(label, dtype = int)


# Make label in one-hot encoding
label_onehot = np.eye(10, dtype = int)[np.array(label_np, dtype = int).reshape(-1)]

test_size = int(len(number)/100)
train_number = number_3D[test_size:]
test_number = number_3D[:test_size]
train_label = label_onehot[test_size:]
test_label = label_onehot[:test_size]
train_size = len(train_number)
batch_size = 1000


# Create input queues
train_input_queue = tf.train.slice_input_producer([number_3D, label_onehot], shuffle=False)
test_input_queue = tf.train.slice_input_producer([test_number, test_label], shuffle=False)
test_ka_input_queue = tf.train.slice_input_producer([number_3D_ka], shuffle=False)

train_data_number = train_input_queue[0]
train_data_label = train_input_queue[1]

batch_num, batch_label = tf.train.batch([train_data_number, train_data_label], batch_size = batch_size)
batch_test_num, batch_test_label = tf.train.batch([test_input_queue[0], test_input_queue[1]], batch_size= batch_size)
batch_num_ka = tf.train.batch(test_ka_input_queue, batch_size = batch_size)



learning_rate = 0.001
training_iters = 40
total_batch = int(len(train_number) / batch_size)
display_steps = 10

    

n_input = 2
n_hidden = 1024 # hidden layer num of features
n_classes = 10
n_layers = 4 # Number of layers of LSTM cell



x = tf.placeholder(tf.float32, [None, n_input, int(max_length/2)])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[52]:


weights = {
    'W1': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'O1': tf.Variable(tf.random_normal([n_classes]))
}


# In[53]:


def RNN(x, weights, biases):
    
    #Current input data shape: (batch_size, n_inputs, n_steps)
    # Required shape: 'n_steps' tensors list of shape(batch_size, n_input)

    # If using static_rnn, then uncomment tf.unstack
    # Unstack to get a list of 'n_steps' tensors of shape(batch_size, n_inputs)
    x = tf.unstack(x, int(max_length/2), 2) 

    print(x)

    gru_cell = rnn.GRUCell(n_hidden, input_size=None)
    #lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(gru_cell, x, dtype=tf.float32)
    
    ## Using dynamic Rnn ##
    #outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    #outputs = tf.transpose(outputs, [1, 0, 2])
    #print(outputs)
    return tf.matmul(outputs[-1], weights['W1']) + biases['O1']


# In[55]:


result = [] # The prediction result

config=tf.ConfigProto(log_device_placement=True)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # op to write logs to tensorboard
    log_path = '/Tensorboard_LSTM/'
    summary_writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    print("Initialized global variable...")
    for train_step in range(training_iters):
        
        print("train step:", train_step, "/", training_iters)
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = sess.run([batch_num, batch_label])
            
            print("batch number" , i, "/", total_batch)
            
            _, c = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
            #summary_writer.add_summary(summary, train_step * total_batch + i)
            
            avg_cost += c
            print("Average cost: ", avg_cost)

            print("---------------------------------------")

        if train_step % display_steps == 0:
            print("Epoch:", '%04d' % (train_step+1), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished...")
    correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    
    print(correct)

    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print("Accuracy:", accuracy.eval({x:test_number, y:test_label}))

    save_path = saver.save(sess, "D:/Machine Learning/model/model_BasicRNN_0610.ckpt")
    print("Model saved in file:", save_path)
    
    print("Total test iter:", int(len(number_3D_ka)/batch_size))
    prediction = tf.argmax(pred, 1)

    for i in range(int(len(number_3D_ka)/batch_size)):
        
    
        print("test iter:", i)
        print(batch_num_ka)
        result.append(prediction.eval(feed_dict = {x: sess.run(batch_num_ka)}, session = sess))
        
result = np.array(result, dtype=np.int64).reshape(-1)

result = result.reshape(-1,1)
print(len(result))

print("Predictions", result)


with open("prediction_lstm_2.csv", 'w', newline='') as out:

    for col in result:
        print(col)
        writer = csv.writer(out, delimiter = ',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(col)

coord.request_stop()
sess.close()


    
            