import numpy as np
import tensorflow as tf
from utils import *
from prepare_data import *



def forward_prop(x):
    
    N_out = 1 + 6*M
    weights_y = tf.Variable(tf.truncated_normal([2*cells_number, N_out], 0.0, 0.075, dtype=tf.float32), name = "weights_y") #shape = [3*cell_numbers,N_out]
    biais_y = tf.Variable(tf.zeros([N_out]), name = "biais_y")
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm1")
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm2")
    lstm1_state = lstm1.zero_state(1, dtype=tf.float32) 
    lstm2_state = lstm2.zero_state(1, dtype=tf.float32)
    
    h1, lstm1_state = lstm1(x, lstm1_state) #[x_list[t] window].shape = [batch_size, 3 + character_number]
    h2, lstm2_state = lstm2(tf.concat([x, h1], axis=1), lstm2_state) #[x_list[t] h1 window].shape = [batch_size,3+cells_number+character_number]

    h_list = tf.concat([h1,h2],axis=1) #shape = [batch_size,T,2*cells_number]
    y_hat = tf.nn.xw_plus_b(h_list,weights_y,biais_y) #shape = [batch_size,T,N_out]
    return y_hat



def compute_loss(y_hat,y):

    y_end_of_stroke,y1,y2 = tf.split(y,3,axis=1) #shape = [1,1]
    y1 = tf.squeeze(y1,axis=1)
    y2 = tf.squeeze(y2,axis=1)
    y_end_of_stroke = tf.squeeze(y_end_of_stroke,axis=1) #shape = [1]
    end_of_stroke = 1 / (1 + tf.exp(y_hat[0,0]))
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[0,1:],6,axis=0) #shape = [20,]
    pi = tf.exp(pi_hat) / tf.reduce_sum(tf.exp(pi_hat)) #shape = [20,]
    sigma1 = tf.exp(sigma1_hat)#shape = [M,]
    sigma2 = tf.exp(sigma2_hat)#shape = [M,]
    mu1 = mu1_hat#shape = [M,]
    mu2 = mu2_hat#shape = [M,]
    rho = tf.tanh(rho_hat) #shape = [M,]   
    Z = tf.square((y1 - mu1) / sigma1) + tf.square((y2 - mu2) / sigma2) - 2 * rho * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2) #shape = [M,]
    gaussian = pi*tf.exp(-Z / (2 * (1 - tf.square(rho)))) / (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))) #shape = [M,]
    eps = 1e-20
    loss_gaussian = -tf.log(tf.reduce_sum(gaussian) + eps)
    loss_bernoulli = -(tf.log(end_of_stroke + eps)*y_end_of_stroke + tf.log(1-end_of_stroke + eps)*(1 - y_end_of_stroke))

    loss = (loss_gaussian+loss_bernoulli)

    return loss

tf.reset_default_graph()

x = tf.placeholder(shape = [1, 3], dtype=tf.float32) #shape = [batch_size,T,3]
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


cost = compute_loss(y_hat,y)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)



saver = tf.train.Saver() 
current_stroke = np.zeros((1,3))
strokes = [[0,0,0]]

#with tf.Session() as sess:
#    saver.restore(sess,"saved/weights.ckpt")
#    for i in range(T-1):
#        Pi,Mu1,Mu2,sig1,sig2,Rho = sess.run([pi,mu1,mu2,sigma1,sigma2,rho], feed_dict = {x: current_stroke})
#        rand = np.random.choice(np.arange(M), p = Pi)
#        x1,x2= np.random.multivariate_normal([Mu1[rand], Mu2[rand]],
#               [[np.square(sig1[rand]), Rho[rand]*sig1[rand] * sig2[rand]],[Rho[rand]*sig1[rand]*sig2[rand], np.square(sig2[rand])]])
#        proba_end = sess.run(end_of_stroke, feed_dict = {x: current_stroke})
#        x0 = np.random.binomial(1,proba_end)
#        current_stroke[0,0] = x0
#        current_stroke[0,1] = x1
#        current_stroke[0,2] = x2
#        strokes.append([x0,x1,x2])

sess = tf.Session()
saver.restore(sess,"saved/weights.ckpt")
for i in range(200):
    Pi,Mu1,Mu2,sig1,sig2,Rho = sess.run([pi,mu1,mu2,sigma1,sigma2,rho], feed_dict = {x: current_stroke})
    rand = np.random.choice(np.arange(M), p = Pi)
    x1,x2= np.random.multivariate_normal([Mu1[rand], Mu2[rand]],
           [[np.square(sig1[rand]), Rho[rand]*sig1[rand] * sig2[rand]],[Rho[rand]*sig1[rand]*sig2[rand], np.square(sig2[rand])]])
    proba_end = sess.run(end_of_stroke, feed_dict = {x: current_stroke})
    x0 = np.random.binomial(1,proba_end)
    current_stroke[0,0] = x0
    current_stroke[0,1] = x1
    current_stroke[0,2] = x2
    strokes.append([x0,x1,x2])
sess.close()

strokes = np.array(strokes)

plot_stroke(strokes)

#from tensorflow.python.tools import inspect_checkpoint as chkp
#chkp.print_tensors_in_checkpoint_file("saved/weights.ckpt", tensor_name='', all_tensors=True)
