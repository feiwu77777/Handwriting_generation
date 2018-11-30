import numpy as np
import tensorflow as tf
from utils import *
from prepare_data import *


############### part 2: training a model #########################
 

def forward_prop(x_list):
    
    N_out = 1 + 6*M
    weights_y = tf.Variable(tf.truncated_normal([2*cells_number, N_out], 0.0, 0.075, dtype=tf.float32), name = "weights_y") #shape = [3*cell_numbers,N_out]
    biais_y = tf.Variable(tf.zeros([N_out]), name = "biais_y")
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm1")
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number,name = "lstm2")
    lstm1_state = lstm1.zero_state(batch_size, dtype=tf.float32) 
    lstm2_state = lstm2.zero_state(batch_size, dtype=tf.float32)

    h1_list = []
    h2_list = []
    
    for t in range(T):
        h1, lstm1_state = lstm1(x_list[t], lstm1_state) #[x_list[t] window].shape = [batch_size, 3 + character_number]
        h1_list.append(h1) #h1.shape = [batch_size, cell_numbers]
        h2, lstm2_state = lstm2(tf.concat([x_list[t], h1], axis=1), lstm2_state) #[x_list[t] h1 window].shape = [batch_size,3+cells_number+character_number]
        h2_list.append(h2) #h2.shape = [batch_size,cells_number] 

    h1_list = tf.reshape(tf.concat(h1_list,axis=0),[batch_size,T,cells_number])
    h2_list = tf.reshape(tf.concat(h2_list,axis=0),[batch_size,T,cells_number])
    h_list = tf.concat([h1_list,h2_list],axis=2) #shape = [batch_size,T,2*cells_number]
    weights_y = expand_duplicate(weights_y,batch_size,0)#shape = [batch_size,2*cell_numbers,N_out]
    y_hat = tf.nn.xw_plus_b(h_list,weights_y,biais_y) #shape = [batch_size,T,N_out]
    return y_hat


def compute_loss(y_hat,y):

    y_end_of_stroke,y1,y2 = tf.split(y,3,axis=2) #shape = [batch_size,T,1]
    y1 = tf.squeeze(y1,axis=2)
    y2 = tf.squeeze(y2,axis=2)
    y_end_of_stroke = tf.squeeze(y_end_of_stroke,axis=2) #shape = [batch_size,T]
    end_of_stroke = 1 / (1 + tf.exp(y_hat[:,:, 0])) #shape = [batch_size,T]
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[:,:,1:],6,axis=2) #shape = [batch_size,T,M]
    pi = tf.exp(pi_hat) / expand_duplicate(tf.reduce_sum(tf.exp(pi_hat),axis=2),M,2) #shape = [batch_size,T,M]
    sigma1 = tf.exp(sigma1_hat)#shape = [batch_size,T,M]
    sigma2 = tf.exp(sigma2_hat)#shape = [batch_size,T,M]
    mu1 = mu1_hat#shape = [batch_size,T,M]
    mu2 = mu2_hat#shape = [batch_size,T,M]
    rho = tf.tanh(rho_hat) #shape = [batch_size,T,M]
    y1 = expand_duplicate(y1,M,2) #shape = [batch_size,T,M]
    y2 = expand_duplicate(y2,M,2) #shape = [batch_size,T,M]
    Z = tf.square((y1 - mu1) / sigma1) + tf.square((y2 - mu2) / sigma2) - 2 * rho * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2) #shape = [batch_size,T,M]
    gaussian = pi*tf.exp(-Z / (2 * (1 - tf.square(rho)))) / (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))) #shape = [batch_size,T,M]
    eps = 1e-20
    loss_gaussian = -tf.log(tf.reduce_sum(gaussian,axis=2) + eps) #shape = [batch_size,T]
    loss_gaussian = tf.reduce_sum(tf.reduce_sum(loss_gaussian,axis=1))
    loss_bernoulli = tf.reduce_sum(-tf.reduce_sum(tf.log(end_of_stroke + eps)*y_end_of_stroke + tf.log(1-end_of_stroke + eps)*(1 - y_end_of_stroke), axis =1))

    loss = (loss_gaussian+loss_bernoulli)/(T*batch_size)

    return loss




tf.reset_default_graph()

x = tf.placeholder(shape = [None, T, 3], dtype=tf.float32) #shape = [batch_size,T,3]
y = tf.placeholder(shape = [None, T, 3], dtype=tf.float32)
x_list = tf.split(x,T,axis=1) #len(x) = T, x[0] is the batch with all x0 across batch_size examples.
x_list = [tf.squeeze(i, axis=1) for i in x_list] #i.shape = [batch_size,1,3]

y_hat = forward_prop(x_list)
cost = compute_loss(y_hat,y)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9,momentum=0.95,epsilon=0.0001)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5):
        epoch_cost = 0
        num_batches = int(X.shape[0] / batch_size)
        batches = random_batches(X, Y, C, batch_size)
        for batch in batches:
            (batch_X, batch_Y, _) = batch
            _, batch_cost = sess.run([train_op,cost],feed_dict = {x:batch_X, y:batch_Y})
            print("batch cost: " + str(batch_cost))
            epoch_cost += batch_cost / num_batches
        print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
    saver.save(sess, "saved/model.ckpt")
