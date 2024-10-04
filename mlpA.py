from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#-----------------------------------------------------------
# Globals

# Training Parameters
learning_rate = 0.001
num_steps = 200000
batch_size = 1000

display_step = 20

# Network Parameters
nout = 2
num_input = 25  # data input
num_hidden_1 = 50 # 1st layer
num_hidden_out = nout # output layer

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], stddev=0.5)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_out], stddev=0.5)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_out])),
}

#-----------------------------------------------------------
# data shuffle
def shuffle_data(data):
    #print("Shuffling data.")
    D = data['D']; E = data['E']
    I = np.random.permutation(D.shape[0])
    Dnew = D.copy(); Enew = E.copy()
    for i in range(D.shape[0]):
        Dnew[i,:] = D[I[i],:]
        Enew[i,:] = E[I[i],:]
    data['D'] = Dnew
    data['E'] = Enew

#-----------------------------------------------------------
# data import (train)
def import_data0(doshuffle):
    D = np.loadtxt('train_in0.txt')
    E = np.loadtxt('train_out0.txt')
    iepoch = 0; idata = 0; ndata = D.shape[0]
    data = {'D':D, 'E':E, 'iepoch':iepoch, 'idata':idata, 'ndata':ndata}
    if (doshuffle):
        np.random.seed(17);
        shuffle_data(data);
    return data


#-----------------------------------------------------------
# data import (validation)
def import_data1(doshuffle):
    D = np.loadtxt('train_in1.txt')
    E = np.loadtxt('train_out1.txt')
    iepoch = 0; idata = 0; ndata = D.shape[0]
    data = {'D':D, 'E':E, 'iepoch':iepoch, 'idata':idata, 'ndata':ndata}
    if (doshuffle):
        np.random.seed(17);
        shuffle_data(data);
    return data

#-----------------------------------------------------------
# get next batch
def next_batch(data, batch_size):
    idata = data['idata']; ndata = data['ndata']
    if (idata + batch_size > ndata):
        shuffle_data(data)
        data['idata'] = idata = 0
        data['iepoch'] += 1
    D = data['D']
    E = data['E']
    X = D[idata:(idata+batch_size),:]
    J = E[idata:(idata+batch_size),:]
    data['idata'] = idata + batch_size
    return X, J


#-----------------------------------------------------------
# Building the MLP network
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                     biases['encoder_b2'])    
    return layer_2


#-----------------------------------------------------------
def main():
    
    # Import Data
    input_data = import_data0(True)
    
    # tf input and output placeholders
    X = tf.placeholder("float", [None, num_input])
    J = tf.placeholder("float", [None, nout])
    
    # Construct model
    encoder_op = encoder(X)

    # Prediction
    y_pred = encoder_op
    y_true = J
    
    # Define loss and optimizer, minimize the squared error
    #loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    loss = tf.losses.mean_squared_error(y_true, y_pred)

    # TODO: divide out by true values before taking the MSE
    #loss = tf.losses.mean_pairwise_squared_error(y_true, y_pred)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    # Start a new TF session
    print("Starting training.")
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # For testing
        test_data = import_data1(False)

        # Training
        print('% Iteration  (Minibatch loss) (training set loss) (testing set loss)')
        for i in range(1, num_steps+1):
            # Prepare Data
            # Get the next batch of data
            batch_x, batch_j  = next_batch(input_data, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, J: batch_j})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                # display losses (testing too)
                #e,la = sess.run([encoder_op,loss], feed_dict={X:input_data['D'], J:input_data['E']})
                #e,lt = sess.run([encoder_op,loss], feed_dict={X:test_data['D'], J:test_data['E']})
                #print('%i %.5e %.5e %.5e'%(i, l, la, lt))
                print('%i %.5e'%(i, l))
                
        # Testing (predicted errors)
        print("Testing")
        e,l = sess.run([encoder_op,loss], feed_dict={X:test_data['D'], J:test_data['E']})
        print("Total validation loss=%f"%(l))
        ea = np.array(e)
        np.savetxt('pred_out.txt', ea, fmt='%.12E')

        print()
        np.savetxt('w1.txt', sess.run(weights['encoder_h1']), fmt='%.12E');
        np.savetxt('w2.txt', sess.run(weights['encoder_h2']), fmt='%.12E');
        np.savetxt('b1.txt', sess.run(biases['encoder_b1']), fmt='%.12E');
        np.savetxt('b2.txt', sess.run(biases['encoder_b2']), fmt='%.12E');

        
        
if __name__ == "__main__":
    main()
