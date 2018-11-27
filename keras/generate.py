from keras.layers import Input, Dense, Concatenate,LSTM,Lambda,Reshape,Add,Multiply,Dot,RepeatVector
from keras.models import Model
import tensorflow as tf
import numpy as np
from prepare_data import *
from utils import *


############ part 3: generate strokes ####################

import pickle
f = open("keras_weights.pkl", 'rb')
params = pickle.load(f)
f.close()

LSTM_cell1 = LSTM(cells_number, return_state = True, name = "lstm1")
LSTM_cell2 = LSTM(cells_number, return_state = True, name = "lstm2")
LSTM_cell3 = LSTM(cells_number, return_state = True, name = "lstm3")
window_dense = Dense(3*K, name = "dense1")
output_dense = Dense(1+6*M, name = "dense2")   


def inference_model(sentence):
    
    U = sentence.shape[0]
    X = Input(shape=(3,),name = "input_x")
    c_vec = Input(shape=(U,character_number),name = "input_cvec")
    init_window = Input((character_number,),name = "input_window")
    init_kappa = Input(shape=(K,1),name = "input_kappa")
    h10 = Input(shape=(cells_number,), name='h10')
    c10 = Input(shape=(cells_number,), name='c10')
    h20 = Input(shape=(cells_number,), name='h20')
    c20 = Input(shape=(cells_number,), name='c20')
    h30 = Input(shape=(cells_number,), name='h30')
    c30 = Input(shape=(cells_number,), name='c30')
    
    u = np.concatenate([np.expand_dims(np.array([i for i in range(1,U+2)], dtype=np.float32),0) for _ in range(K)], axis = 0) #shape = [K,U]
    
    conc1 = Concatenate(axis=1)([X,init_window])
    conc1 = Reshape((1,3+character_number))(conc1)
    h1, _,c1  = LSTM_cell1(conc1, initial_state = [h10, c10])
    
    output_wl = window_dense(h1)
    alpha_hat, beta_hat, kappa_hat = Lambda(lambda x: [x[:,:K],x[:,K:2*K],x[:,2*K:3*K]])(output_wl)
    alpha = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2))(alpha_hat)
    beta = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2))(beta_hat)
    kappa = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2))(kappa_hat)
    kappa = Add()([kappa,init_kappa])
    un = Lambda(lambda x: tf.square(x-u))(kappa)
    deux = Multiply(name = "mult1-%i" % t)([beta,un])
    trois = Lambda(lambda x: tf.exp(-x))(deux)
    quatro = Multiply()([alpha,trois])
    phi = Lambda(lambda x: tf.reduce_sum(x,axis = 1))(quatro)
    product = Lambda(lambda x: x[:,:-1])(phi)
    window = Dot(axes = [1,1])([product,c_vec]) 
    
    conc2 = Concatenate(axis = 1)([X,h1,window])
    conc2 = Reshape((1,3+character_number+400))(conc2)
    h2,_,c2 = LSTM_cell2(conc2, initial_state = [h20, c20])

    
    conc3 = Concatenate(axis = 1)([X,h2,window])
    conc3 = Reshape((1,3+character_number+400))(conc2)
    h3,_,c3 = LSTM_cell3(conc3, initial_state = [h30, c30])
    
    h = Concatenate(axis=1)([h1,h2,h3])
    y_hat = output_dense(h)
    
    model = Model(inputs = [X,h10,c10,h20,c20,h30,c30,c_vec,init_window,init_kappa], outputs = [y_hat,h1,c1,h2,c2,h3,c3,window,kappa,phi])
    
    return model




def generate_strokes(sentence):

    U = sentence.shape[0]
    inference = inference_model(sentence)
  
    LSTM_cell1.set_weights(params["weights_lstm1"])
    LSTM_cell2.set_weights(params["weights_lstm2"])
    LSTM_cell3.set_weights(params["weights_lstm3"])
    window_dense.set_weights(params["window_weights"])
    output_dense.set_weights(params["output_weights"])
    
    h1 = np.zeros((1,cells_number))
    c1 = np.zeros((1,cells_number))
    h2 = np.zeros((1,cells_number))
    c2 = np.zeros((1,cells_number))
    h3 = np.zeros((1,cells_number))
    c3 = np.zeros((1,cells_number))
    window = np.zeros((1,character_number))
    kappa = np.zeros((1,K,1))
    x = np.zeros((1,3))    
    strokes = []
    
    while True:
        L = [x,h1,c1,h2,c2,h3,c3,sentence,window,kappa]
        y_hat,h1,c1,h2,c2,h3,c3,window,kappa,phi = inference.predict(L)
        if sum(phi[U] > phi[:U-1]) == U:
            break
        x = sample(y_hat)
        strokes.append(x)
    return np.array(strokes)

sentence = "example sentence"
sentence = preprocess_sentence(sentence)
