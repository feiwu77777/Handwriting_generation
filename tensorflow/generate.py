import numpy as np
import tensorflow as tf
from utils import *
from prepare_data import *


################ Part 3: Generate strokes ######################

import pickle
f = open("model_weights.pkl", 'rb')
params = pickle.load(f)
f.close()

sentence1 = "Example sentence."
sentence2 = C[0]
sentence1 = preprocess_sentence(sentence1)


lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number)
lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number)
lstm3 = tf.nn.rnn_cell.LSTMCell(cells_number)
weights_h1_p = tf.convert_to_tensor(params["weights_h1_p"]) #shape = [cells_number, 3*K]
biais_p = tf.convert_to_tensor(params["biais_p"]) #shape = [3*K]
weights_y = tf.convert_to_tensor(params["weights_y"]) #shape = [3*cell_numbers,N_out]
biais_y = tf.convert_to_tensor(params["biais_y"]) #shape = [N_out]


def init_lstm_weights():
    x = tf.placeholder(shape = [1,3], dtype = tf.float32)
    lstm1_state = lstm1.zero_state(1, dtype=tf.float32) 
    lstm2_state = lstm2.zero_state(1, dtype=tf.float32)
    lstm3_state = lstm3.zero_state(1, dtype=tf.float32)
    window = tf.zeros([1,character_number])
    h1,_ = lstm1(tf.concat([x, window],axis=1), lstm1_state)
    h2,_ = lstm2(tf.concat([x, h1, window],axis=1), lstm2_state)
    h3,_ = lstm3(tf.concat([x, h2, window],axis=1), lstm3_state)
    lstm1.set_weights(params["weights_lstm1"])
    lstm2.set_weights(params["weights_lstm2"])
    lstm3.set_weights(params["weights_lstm3"])
    
init_lstm_weights()


def generate_strokes(sentence, params):
    
    length = sentence.shape[0]
    current_stroke = np.zeros((1,3), dtype = "float32")
    strokes = []
    
    x = tf.placeholder(shape = [1,3], dtype = tf.float32)
    init_lstm1 = lstm1.zero_state(1, dtype=tf.float32) 
    init_lstm2 = lstm2.zero_state(1, dtype=tf.float32)
    init_lstm3 = lstm3.zero_state(1, dtype=tf.float32)
    lstm1_state = init_lstm1
    lstm2_state = init_lstm2
    lstm3_state = init_lstm3
    
    previous_kappa = tf.zeros([K,1])  
    window = tf.zeros([1,character_number])
    
    u = expand_duplicate(np.array([i for i in range(1,length+2)], dtype=np.float32),K,0) #u.shape = [K,length+1] 

    reuse = False
    strokes.append([0,0,0])
    
    while True:
    #for i in range(T-1):
        with tf.variable_scope("lstm1", reuse=reuse):
            h1, lstm1_state = lstm1(tf.concat([x, window],axis=1), lstm1_state) #[x_list[t] window].shape = [3 + character_number]
        output_wl = tf.nn.xw_plus_b(h1, weights_h1_p, biais_p) #shape = [1,3*K]      h1.shape = [1,cells_number]
        alpha_hat, beta_hat, kappa_hat = tf.split(tf.reshape(output_wl,[3*K,1]),3,axis=0) #shape = [K,1]
        alpha = tf.exp(alpha_hat) #shape = [K,1]
        beta = tf.exp(beta_hat) #shape = [K,1]
        kappa = previous_kappa + tf.exp(kappa_hat) #shape = [K,1]
        previous_kappa = kappa
        phi = tf.reduce_sum(alpha*tf.exp(-beta*tf.square(kappa-u)),axis=0,keepdims=True) #phi.shape = [1,length+1] 
        window = tf.matmul(phi[:,:-1], sentence) #shape = [1, character_number] 
        with tf.variable_scope("lstm2", reuse=reuse):
            h2, lstm2_state = lstm2(tf.concat([x, h1, window],axis=1), lstm2_state) #[x_list[t] h1 window].shape = [1,3+cells_number+character_number]
        with tf.variable_scope("lstm3", reuse=reuse):
            h3, lstm3_state = lstm3(tf.concat([x, h2, window],axis=1), lstm3_state) #[x_list[t], h2, window].shape = [1,3+cells_number+character_number]
        h = tf.concat([h1,h2,h3], axis = 1) #shape = [1,3*cells_number]
        y_hat = tf.nn.xw_plus_b(h,weights_y,biais_y) #shape = [1,N_out]
        
        end_of_stroke = 1 / (1 + tf.exp(y_hat[0,0]))
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[0,1:],6,axis=0) #shape = [20,]
        pi = tf.exp(pi_hat) / tf.reduce_sum(tf.exp(pi_hat)) #shape = [20,]
        sigma1 = tf.exp(sigma1_hat)#shape = [M,]
        sigma2 = tf.exp(sigma2_hat)#shape = [M,]
        mu1 = mu1_hat#shape = [M,]
        mu2 = mu2_hat#shape = [M,]
        rho = tf.tanh(rho_hat) #shape = [M,]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            PHI = sess.run(phi, feed_dict = {x: current_stroke})
            if sum(PHI[0,-1] > PHI[0,:-1]) == length:
                print("sentence scan end")
                break
            Pi,Mu1,Mu2,sig1,sig2,Rho = sess.run([pi,mu1,mu2,sigma1,sigma2,rho], feed_dict = {x: current_stroke})
            accuracy = 0
            for m in range(M):
                accuracy += Pi[m]
                if accuracy > 0.5:
                    x1,x2= np.random.multivariate_normal([Mu1[m], Mu2[m]],
                           [[np.square(sig1[m]), Rho[m]*sig1[m] * sig2[m]],[Rho[m]*sig1[m]*sig2[m], np.square(sig2[m])]])
                    break
            proba_end = sess.run(end_of_stroke, feed_dict = {x: current_stroke})
            if proba_end > 0.5:
                x0 = 1
            else:
                x0 = 0
        current_stroke[0,0] = x0
        current_stroke[0,1] = x1
        current_stroke[0,2] = x2
        strokes.append([x0,x1,x2])
        reuse = True
        if len(strokes) >= 2000:
            print("limit")
            break
        
    return np.array(strokes)
    
stro = generate_strokes(sentence2, params)
plot_stroke(stro)


def random_generate_strokes(params):
    
    x = tf.placeholder(shape = [1,3], dtype = tf.float32)
    current_stroke = np.zeros((1,3), dtype = "float32")
    strokes = []
    
    init_lstm1 = lstm1.zero_state(1, dtype=tf.float32) 
    init_lstm2 = lstm2.zero_state(1, dtype=tf.float32)
    init_lstm3 = lstm3.zero_state(1, dtype=tf.float32)
    lstm1_state = init_lstm1
    lstm2_state = init_lstm2
    lstm3_state = init_lstm3
    
    window = tf.zeros([1,character_number])
    
    reuse = False
    strokes.append([0,0,0])
    
    for i in range(T-1):
        with tf.variable_scope("lstm1", reuse=reuse):
            h1, lstm1_state = lstm1(tf.concat([x,window],axis=1), lstm1_state) #[x_list[t] window].shape = [3 + character_number]
        with tf.variable_scope("lstm2", reuse=reuse):
            h2, lstm2_state = lstm2(tf.concat([x, h1,window],axis=1), lstm2_state) #[x_list[t] h1 window].shape = [1,3+cells_number+character_number]
        with tf.variable_scope("lstm3", reuse=reuse):
            h3, lstm3_state = lstm3(tf.concat([x, h2,window],axis=1), lstm3_state) #[x_list[t], h2, window].shape = [1,3+cells_number+character_number]
        h = tf.concat([h1,h2,h3], axis = 1) #shape = [1,3*cells_number]
        y_hat = tf.nn.xw_plus_b(h,weights_y,biais_y) #shape = [1,N_out]
        
        end_of_stroke = 1 / (1 + tf.exp(y_hat[0,0]))
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[0,1:],6,axis=0) #shape = [20,]
        pi = tf.exp(pi_hat) / tf.reduce_sum(tf.exp(pi_hat)) #shape = [20,]
        sigma1 = tf.exp(sigma1_hat)#shape = [M,]
        sigma2 = tf.exp(sigma2_hat)#shape = [M,]
        mu1 = mu1_hat#shape = [M,]
        mu2 = mu2_hat#shape = [M,]
        rho = tf.tanh(rho_hat) #shape = [M,]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Pi,Mu1,Mu2,sig1,sig2,Rho = sess.run([pi,mu1,mu2,sigma1,sigma2,rho], feed_dict = {x: current_stroke})
            accuracy = 0
            for m in range(M):
                accuracy += Pi[m]
                if accuracy > 0.5:
                    x0,x1= np.random.multivariate_normal([Mu1[m], Mu2[m]],
                           [[np.square(sig1[m]), Rho[m]*sig1[m] * sig2[m]],[Rho[m]*sig1[m]*sig2[m], np.square(sig2[m])]])
                    break
            proba_end = sess.run(end_of_stroke, feed_dict = {x: current_stroke})
            if proba_end > 0.5:
                x2 = 1
            else:
                x2 = 0
        current_stroke[0,0] = x0
        current_stroke[0,1] = x1
        current_stroke[0,2] = x2
        strokes.append([x0,x1,x2])
        reuse = True
        
    return np.array(strokes)




