#imports
import tensorflow as tf
import pickle
import random
import os
print("Finished importing")


def model(input_data, weights, bias):
    y = 0
    for i in range(len(weights)):
        y = y + input_data[i] * weights[i]
    y = y + bias
    return y


#reset the graph
tf.reset_default_graph()

#set up the variables
input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

NUM_DIMENSIONS = 11

weights = []
for i in range(NUM_DIMENSIONS):
    weights.append(tf.Variable(0.0, dtype=tf.float32))

bias = tf.Variable(0.0, dtype=tf.float32)

#linear regression ftw
model_operation = model(input_data, weights, bias)
error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
train = optimizer.minimize(loss)

#checkpoint setup
path = "./mlr-model/"

load_checkpoint = True

#import the data
inputs = pickle.load(open("input.pkl", "rb"))
random.shuffle(inputs)
training_inputs = inputs[:29]
test_inputs = inputs[30:]
print(test_inputs)
print(len(test_inputs))

revs = pickle.load(open("output.pkl", "rb"))
random.shuffle(revs)
training_revs = revs[:29]
test_revs = revs[30:]
print(test_revs)
print(len(test_revs))


#run a session
saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    if load_checkpoint:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(init)
    sess.run(tf.local_variables_initializer())
    for epoch in range(90000):
        sess.run(train, feed_dict={input_data: training_inputs, output_data: training_revs})
        if (epoch%1000==0):
            print("loss", sess.run(loss, feed_dict={input_data: training_inputs, output_data: training_revs}))
            print("Saving checkpoint")
            saver.save(sess, path + "mlr", epoch)


    print("Training set loss:")
    print(sess.run(loss, feed_dict={input_data: training_inputs, output_data: training_revs}))


with tf.Session() as sess:
    print("Test set loss:")

    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    sess.run(tf.local_variables_initializer())

    for inp, label in zip(test_inputs, test_revs):
        tsl = sess.run(loss, feed_dict={input_data: inp, output_data: label})
    print(tsl)


