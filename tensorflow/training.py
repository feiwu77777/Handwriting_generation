import numpy as np
import tensorflow as tf
from utils import *
from prepare_data import *


############### part 2: training a model #########################

def init_params():
    params = {}
    weights_h1_p = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    biais_p = tf.Variable(tf.zeros([3*K]))
    N_out = 1 + 6*M
#    weights_h1_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
#    weights_h2_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
#    weights_h3_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
#    weights_y = tf.concat([weights_h1_y,weights_h2_y,weights_h3_y],axis=0) #shape = [3*cell_numbers,N_out]
    weights_y = tf.Variable(tf.truncated_normal([3*cells_number, N_out], 0.0, 0.075, dtype=tf.float32)) #shape = [3*cell_numbers,N_out]
    biais_y = tf.Variable(tf.zeros([N_out]))
    
    params["weights_h1_p"] = weights_h1_p
    params["biais_p"] = biais_p
    params["weights_y"] = weights_y
    params["biais_y"] = biais_y
    
    return params

#params = init_params()

def init_layers():
    layers = {}
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm3 = tf.nn.rnn_cell.LSTMCell(cells_number)
    init_lstm1 = lstm1.zero_state(batch_size, dtype=tf.float32) 
    init_lstm2 = lstm2.zero_state(batch_size, dtype=tf.float32)
    init_lstm3 = lstm3.zero_state(batch_size, dtype=tf.float32)
    lstm1_state = init_lstm1
    lstm2_state = init_lstm2
    lstm3_state = init_lstm3
    layers["lstm1"] = [lstm1,lstm1_state]
    layers["lstm2"] = [lstm2,lstm2_state]
    layers["lstm3"] = [lstm3,lstm3_state]
    return layers

#layers = init_layers()    

def forward_prop(x_list,c_vec,params,layers):

    lstm1 = layers["lstm1"][0]
    lstm2 = layers["lstm2"][0]
    lstm3 = layers["lstm3"][0]
    lstm1_state = layers["lstm1"][1]
    lstm2_state = layers["lstm2"][1]
    lstm3_state = layers["lstm3"][1]
    
    h1_list = []
    h2_list = []
    h3_list = []

    weights_h1_p = params["weights_h1_p"]
    biais_p = params["biais_p"]
    
    previous_kappa = tf.zeros([batch_size, K, 1])  
    window = tf.zeros([batch_size, character_number])
    
    u = expand_duplicate(expand_duplicate(np.array([i for i in range(1,U+1)], dtype=np.float32),K,0),batch_size,0)  #u.shape = (batch_size,K,U) each line being [i for i in range(1,U+1)] 

    reuse = False
    for t in range(T):
        with tf.variable_scope("lstm1", reuse=reuse):
            h1, lstm1_state = lstm1(tf.concat([x_list[t], window],axis=1), lstm1_state) #[x_list[t] window].shape = [batch_size, 3 + character_number]
        h1_list.append(h1) #h1.shape = [batch_size, cell_numbers]
        k_gaussian = tf.nn.xw_plus_b(h1, weights_h1_p, biais_p) #shape = [batch_size,3*K]
        alpha_hat, beta_hat, kappa_hat = tf.split(k_gaussian,3,axis=1) #shape = [batch_size,K]
        alpha = tf.expand_dims(tf.exp(alpha_hat), 2) #shape = [batch_size,K,1]
        beta = tf.expand_dims(tf.exp(beta_hat), 2) #shape = [batch_size,K,1]
        kappa = previous_kappa + tf.expand_dims(tf.exp(kappa_hat), 2) #shape = [batch_size,K,1]
        previous_kappa = kappa
        phi = tf.reduce_sum(alpha*tf.exp(-beta*tf.square(kappa-u)),axis=1,keepdims=True) #(kappa-u).shape = [batch_size,K,U], phi.shape = [batch_size,1,U] 
        window = tf.squeeze(tf.matmul(phi, c_vec),axis = 1) #shape = [batch_size, character_number] 

        with tf.variable_scope("lstm2", reuse=reuse):
            h2, lstm2_state = lstm2(tf.concat([x_list[t], h1, window],axis=1), lstm2_state) #[x_list[t] h1 window].shape = [batch_size,3+cells_number+character_number]
        h2_list.append(h2) #h2.shape = [batch_size,cells_number] 
        with tf.variable_scope("lstm3", reuse=reuse):
            h3, lstm3_state = lstm3(tf.concat([x_list[t], h2, window],axis=1), lstm3_state) #[x_list[t], h2, window].shape = [batch_size,3+cells_number+character_number]
        h3_list.append(h3) #h3.shape = [batch_size,cells_number]
        reuse = True


    h1_list = tf.reshape(tf.concat(h1_list,axis=0),[batch_size,T,cells_number])
    h2_list = tf.reshape(tf.concat(h2_list,axis=0),[batch_size,T,cells_number])
    h3_list = tf.reshape(tf.concat(h3_list,axis=0),[batch_size,T,cells_number])
    h_list = tf.concat([h1_list,h2_list,h3_list],axis=2) #shape = [batch_size,T,3*cells_number]
    weights_y = params["weights_y"]
    biais_y = params["biais_y"]
    weights_y = expand_duplicate(weights_y,batch_size,0)#shape = [batch_size,3*cell_numbers,N_out]
    y_hat = tf.nn.xw_plus_b(h_list,weights_y,biais_y) #shape = [batch_size,T,N_out]
    return y_hat


def compute_loss(y_hat,y):

    y_end_of_stroke,y1,y2 = tf.split(y,3,axis=2) #shape = [batch_size,T,1]
    y1, y2, y_end_of_stroke = tf.squeeze(y1,axis=2),tf.squeeze(y2,axis=2),tf.squeeze(y_end_of_stroke,axis=2) #shape = [batch_size,T]
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


from tensorflow.python.framework import ops


def train_model(epochs, X_train, Y_train, C_vec, batch_size):
    ops.reset_default_graph()
    
    c_vec = tf.placeholder(shape = [None, U, character_number], dtype=tf.float32)
    x = tf.placeholder(shape = [None, T, 3], dtype=tf.float32) #shape = [batch_size,T,3]
    y = tf.placeholder(shape = [None, T, 3], dtype=tf.float32)
    x_list = tf.split(x,T,axis=1) #len(x) = T, x[0] is the batch with all x0 across batch_size examples.
    x_list = [tf.squeeze(i, axis=1) for i in x_list] #i.shape = [batch_size,1,3]
    
    params = init_params()
    layers = init_layers()
    
    y_hat = forward_prop(x_list,c_vec,params,layers)
    cost = compute_loss(y_hat,y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9,momentum=0.95,epsilon=0.0001)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            epoch_cost = 0
            num_batches = int(X.shape[0] / batch_size)
            batches = random_batches(X_train, Y_train, C_vec, batch_size)
            for batch in batches:
                (batch_X, batch_Y, batch_C) = batch
                _, batch_cost = sess.run([train_op,cost],feed_dict = {x:batch_X, y:batch_Y, c_vec:batch_C})
                print("batch cost: " + str(batch_cost))
                epoch_cost += batch_cost / num_batches
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        parameters = sess.run(params)
        lstm1W = sess.run(layers["lstm1"][0].weights)
        lstm2W = sess.run(layers["lstm2"][0].weights)
        lstm3W = sess.run(layers["lstm3"][0].weights)
        parameters["weights_lstm1"] = lstm1W
        parameters["weights_lstm2"] = lstm2W
        parameters["weights_lstm3"] = lstm3W
    
    return parameters
        
                

#params = train_model(6,X,Y,C,batch_size)


#import pickle
#f = open("model_weights.pkl","wb")
#pickle.dump(params,f)
#f.close()


