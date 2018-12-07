import numpy as np
import tensorflow as tf
from utils import *
from prepare_data import *



def forward_prop(x):
    
    N_out = 1 + 6*M
    weights_y = tf.Variable(tf.truncated_normal([3*cells_number, N_out], 0.0, 0.075, dtype=tf.float32), name = "weights_y") #shape = [3*cell_numbers,N_out]
    biais_y = tf.Variable(tf.zeros([N_out]), name = "biais_y")
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm1")
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm2")
    lstm3 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm3")
    lstm1_state = lstm1.zero_state(1, dtype=tf.float32) 
    lstm2_state = lstm2.zero_state(1, dtype=tf.float32)
    lstm3_state = lstm3.zero_state(1, dtype=tf.float32)
    
    h1, lstm1_state = lstm1(x, lstm1_state) 
    h2, lstm2_state = lstm2(tf.concat([x, h1], axis=1), lstm2_state)
    h3, lstm3_state = lstm2(tf.concat([x, h2], axis=1), lstm3_state) 

    h_list = tf.concat([h1,h2,h3],axis=1) 
    y_hat = tf.nn.xw_plus_b(h_list,weights_y,biais_y) 
    return y_hat


tf.reset_default_graph()

x = tf.placeholder(shape = [1, 3], dtype=tf.float32)
y = tf.placeholder(shape = [1, 3], dtype=tf.float32)

y_hat = forward_prop(x) 

end_of_stroke = 1 / (1 + tf.exp(y_hat[0,0]))
pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[0,1:],6,axis=0) #shape = [20,]
pi = tf.exp(pi_hat) / tf.reduce_sum(tf.exp(pi_hat)) #shape = [20,]
sigma1 = tf.exp(sigma1_hat)#shape = [M,]
sigma2 = tf.exp(sigma2_hat)#shape = [M,]
mu1 = mu1_hat#shape = [M,]
mu2 = mu2_hat#shape = [M,]
rho = tf.tanh(rho_hat) #shape = [M,]

saver = tf.train.Saver() 

current_stroke = np.zeros((1,3))
strokes = [[0,0,0]]

sess = tf.Session()
saver.restore(sess,"saved/model.ckpt")
for i in range(200):
    Pi,Mu1,Mu2,sig1,sig2,Rho = sess.run([pi,mu1,mu2,sigma1,sigma2,rho], feed_dict = {x: current_stroke})
    rand = np.argmax(Pi)
    x1,x2= np.random.multivariate_normal([Mu1[rand], Mu2[rand]],
           [[np.square(sig1[rand]), Rho[rand]*sig1[rand] * sig2[rand]],[Rho[rand]*sig1[rand]*sig2[rand], np.square(sig2[rand])]])
    proba_end = sess.run(end_of_stroke, feed_dict = {x: current_stroke})
    if proba_end > 0.5:
        x0 = 1
    else:
        x0 = 0
    current_stroke[0,0] = x0
    current_stroke[0,1] = x1
    current_stroke[0,2] = x2
    strokes.append([x0,x1,x2])
sess.close()

strokes = np.array(strokes)
strokes = denormarlize(strokes, norm_params)

plot_stroke(strokes)
#
#from tensorflow.python.tools import inspect_checkpoint as chkp
#chkp.print_tensors_in_checkpoint_file("saved/model.ckpt", tensor_name='', all_tensors=True)
