import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
input > weight > hidden layer 1 (activation function) > weights >
hidden layer 2 (activation function) > weights > output layer.

compare ouput to intended output > cost function (cross entropy)

optimization function (optimizer) > minimize cost (AdamOptimizer,
........SGD, AdaGrad) leads to backpropogation

feed forward + backprop = epoch(1 cycle)
'''

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]
5 = [0,0,0,0,0,1,0,0,0,0]
6 = [0,0,0,0,0,0,1,0,0,0]
7 = [0,0,0,0,0,0,0,1,0,0]
8 = [0,0,0,0,0,0,0,0,1,0]
9 = [0,0,0,0,0,0,0,0,0,1]
'''

n_nodes_hl1 = 300
n_nodes_hl2 = 300
n_nodes_hl3 = 300

n_classes = 10
batch_size = 100

# feature_size = 28*28 pixels = 784
# height * width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights'])
                , hidden_1_layer['biases'])
    # activation function = rectified linear (threshold function)
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights'])
                , hidden_2_layer['biases'])
    # activation function = rectified linear (threshold function)
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights'])
                , hidden_3_layer['biases'])
    # activation function = rectified linear (threshold function)
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights'])
                    , output_layer['biases'])

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=y))

    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles of feed forward + back prop
    no_of_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training the network (with training data (mnist.train))
        for epoch in range(no_of_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x,
                                                              y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, ' completed out of ', no_of_epochs, 'loss:',
                  epoch_loss)

        # testing the mnist_test_data which is different from the training data
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print ('Accuracy: ', accuracy.eval({x: mnist.test.images,
                                            y: mnist.test.labels}))

train_neural_network(x)
