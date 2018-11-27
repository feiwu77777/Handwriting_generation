from keras.layers import Input, Dense, Concatenate,LSTM,Lambda,Reshape,Add,Multiply,Dot
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam


################## part1: Preparing data ##########################

strokes = np.load('data/strokes.npy',encoding = 'latin1')
with open('data/sentences.txt') as f:
    texts = f.readlines()
    
T = 10

X = np.zeros((100,10,3))
for i in range(X.shape[0]):
    X[i,:,:] = strokes[i][:10,:]
    
Y = np.zeros((10,100,3))
for i in range(X.shape[0]):
    Y[:T-1,i,:] = X[i,1:T,:]
    
    
def preprocess_text(texts):
    for i in range(len(texts)):
        texts[i] = texts[i][:len(texts[i])-1]
    unique = []   
    for i in range(len(texts)):
        unique.append(list(set(texts[i])))   
    unique = np.concatenate(unique,axis = 0)
    unique = list(set(unique))
    cut = ["!",".","0","8","2","4","+","#",")",'"',";",":","3","/",",","(","9","5","1","2","6","7","-","'","?"]    
    unique = [o for o in unique if o not in cut]            
    texts = np.array(texts)
    dictionnary = {character:index for index, character in enumerate(unique)}
    maxs = len(texts[0])
    for i in range(1,len(texts)):
        if len(texts[i]) > maxs:
            maxs = len(texts[i])
    C_vec = np.zeros((texts.shape[0],maxs,len(unique)+1),dtype = "float32")
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            if texts[i][j] in cut:
                C_vec[i,j,len(unique)] = 1
            else:
                C_vec[i,j,dictionnary[texts[i][j]]] = 1       
    return C_vec, dictionnary, cut,texts
C,d,cut,texts = preprocess_text(texts)
C = C[:100,:,:]

character_number = C.shape[2]
K = 10
U = C.shape[1]
batch_size = 10
M = 20



################## Defining model structure ###################################


LSTM_cell1 = LSTM(400, return_state = True, name = "lstm1")
LSTM_cell2 = LSTM(400, return_state = True, name = "lstm2")
LSTM_cell3 = LSTM(400, return_state = True, name = "lstm3")
window_dense = Dense(3*K, name = "dense1")
output_dense = Dense(121, name = "dense2")


def model_structure():
    
    X = Input(shape=(T,3),name = "input_x")
    c_vec = Input(shape=(U,character_number),name = "input_cvec")
    init_window = Input((character_number,),name = "input_window")
    init_kappa = Input(shape=(K,1),name = "input_kappa")
    h10 = Input(shape=(400,), name='h10')
    c10 = Input(shape=(400,), name='c10')
    h20 = Input(shape=(400,), name='h20')
    c20 = Input(shape=(400,), name='c20')
    h30 = Input(shape=(400,), name='h30')
    c30 = Input(shape=(400,), name='c30')
    
    window = init_window
    kappa_prev = init_kappa
    h1 = h10
    c1 = c10
    h2 = h10
    c2 = c10
    h3 = h10
    c3 = c10
    outputs = []
    
    u = np.concatenate([np.expand_dims(np.array([i for i in range(1,U+1)], dtype=np.float32),0) for _ in range(K)], axis = 0)

    
    for t in range(T):
        x = Lambda(lambda x: x[:,t,:], name = "lamb1-%i" % t)(X)
        conc1 = Concatenate(axis=1, name ="conc1-%i" % t)([x,window])
        conc1 = Reshape((1,3+character_number), name = "reshape1-%i" % t)(conc1)
        h1, _,c1  = LSTM_cell1(conc1, initial_state = [h1, c1])
        
        output_wl = window_dense(h1)
        alpha_hat, beta_hat, kappa_hat = Lambda(lambda x: [x[:,:K],x[:,K:2*K],x[:,2*K:3*K]],name="lamb2-%i" % t)(output_wl)
        alpha = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb3-%i" % t)(alpha_hat)
        beta = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb4-%i" % t)(beta_hat)
        kappa = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb5-%i" % t)(kappa_hat)
        kappa = Add(name = "add1-%i" % t)([kappa,kappa_prev])
        kappa_prev = kappa
        un = Lambda(lambda x: tf.square(x-u), name = "lamb6-%i" % t)(kappa)
        deux = Multiply(name = "mult1-%i" % t)([beta,un])
        trois = Lambda(lambda x: tf.exp(-x), name = "lamb7-%i" % t)(deux)
        quatro = Multiply(name = "mult2-%i" % t)([alpha,trois])
        phi = Lambda(lambda x: tf.reduce_sum(x,axis = 1), name = "lamb8-%i" % t)(quatro)
        window = Dot(axes = [1,1], name = "dot1-%i" % t)([phi,c_vec]) 
        
        conc2 = Concatenate(axis = 1, name = "conc2-%i" % t)([x,h1,window])
        conc2 = Reshape((1,3+character_number+400), name = "reshape2-%i" % t)(conc2)
        h2,_,c2 = LSTM_cell2(conc2, initial_state = [h2, c2])
    
        
        conc3 = Concatenate(axis = 1, name = "conc3-%i" % t)([x,h2,window])
        conc3 = Reshape((1,3+character_number+400), name = "reshape3-%i" % t)(conc2)
        h3,_,c3 = LSTM_cell3(conc3, initial_state = [h3, c3])
        
        h = Concatenate(axis=1, name = "conc4-%i" % t)([h1,h2,h3])
        y_hat = output_dense(h)
        outputs.append(y_hat)
    
    model = Model(inputs = [X,h10,c10,h20,c20,h30,c30,c_vec,init_window,init_kappa], outputs = outputs)
    
    return model


model= model_structure()



###################### Defining custom loss function ###############################

def expand_duplicate(x, N, dim):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)],axis =dim)


def compute_loss(y,y_hat):

    y_end_of_stroke, y1, y2 = tf.split(y,3,axis=1) #shape = [batch_size,1]
    y1 = tf.squeeze(y1,axis=1) 
    y2 = tf.squeeze(y2,axis=1) 
    y_end_of_stroke = tf.squeeze(y_end_of_stroke,axis=1) 
    end_of_stroke = 1 / (1 + tf.exp(y_hat[:, 0])) #shape = [batch_size]
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[:,1:],6,axis=1) #shape = [batch_size,M]
    pi = tf.exp(pi_hat) / expand_duplicate(tf.reduce_sum(tf.exp(pi_hat),axis=1),M,1) #shape = [batch_size,M]
    sigma1 = tf.exp(sigma1_hat)#shape = [batch_size,M] 
    sigma2 = tf.exp(sigma2_hat)#shape = [batch_size,M] 
    mu1 = mu1_hat#shape = [batch_size,M] 
    mu2 = mu2_hat#shape = [batch_size,M]
    rho = tf.tanh(rho_hat) #shape = [batch_size,M]
    y1 = expand_duplicate(y1,M,1) #shape = [batch_size,M]
    y2 = expand_duplicate(y2,M,1) #shape = [batch_size,M]
    Z = tf.square((y1 - mu1) / sigma1) + tf.square((y2 - mu2) / sigma2) - 2 * rho * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2) #shape = [batch_size,M]
    gaussian = pi*tf.exp(-Z / (2 * (1 - tf.square(rho)))) / (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))) #shape = [batch_size,M]
    eps = 1e-20
    loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(gaussian,axis=1) + eps))
    loss_bernoulli = -tf.reduce_sum(tf.log(end_of_stroke + eps)*y_end_of_stroke + tf.log(1-end_of_stroke + eps)*(1 - y_end_of_stroke))

    loss = (loss_gaussian+loss_bernoulli)/batch_size

    return loss



######################## Training the model #############################

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01, clipnorm = 5.0)

model.compile(optimizer=opt, loss = compute_loss)

h10 = np.zeros((100,400))
c10 = np.zeros((100,400))
h20 = np.zeros((100,400))
c20 = np.zeros((100,400))
h30 = np.zeros((100,400))
c30 = np.zeros((100,400))
init_window = np.zeros((100,character_number))
init_kappa = np.zeros((100,K,1))
L = [X,h10,c10,h20,c20,h30,c30,C,init_window,init_kappa]

model.fit(L,list(Y),batch_size = 10,epochs = 5)
    

##################### Saving layers parameters #############################

params = {"weights_lstm1": LSTM_cell1.get_weights(),
          "weights_lstm2": LSTM_cell2.get_weights(),
          "weights_lstm3": LSTM_cell3.get_weights(),
          "window_weights": window_dense.get_weights(),
          "output_dense": output_dense.get_weights(),
          }

import pickle
f = open("keras_weights.pkl","wb")
pickle.dump(params,f)
f.close()


###################### Loading saved weights of layer #########################

f = open("keras_weights.pkl", 'rb')
params = pickle.load(f)
f.close()

LSTM_cell1 = LSTM(400, return_state = True, name = "lstm1")
LSTM_cell2 = LSTM(400, return_state = True, name = "lstm2")
LSTM_cell3 = LSTM(400, return_state = True, name = "lstm3")
window_dense = Dense(3*K, name = "dense1")
output_dense = Dense(121, name = "dense2")

def calculate_pi(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x))
    


def inference_model():
    
    X = Input(shape=(3,),name = "input_x")
    c_vec = Input(shape=(U,character_number),name = "input_cvec")
    init_window = Input((character_number,),name = "input_window")
    init_kappa = Input(shape=(K,1),name = "input_kappa")
    h10 = Input(shape=(400,), name='h10')
    c10 = Input(shape=(400,), name='c10')
    h20 = Input(shape=(400,), name='h20')
    c20 = Input(shape=(400,), name='c20')
    h30 = Input(shape=(400,), name='h30')
    c30 = Input(shape=(400,), name='c30')
    
    u = np.concatenate([np.expand_dims(np.array([i for i in range(1,U+2)], dtype=np.float32),0) for _ in range(K)], axis = 0) #shape = [K,U]
    
    t = 0    
    conc1 = Concatenate(axis=1, name ="conc1-%i" % t)([X,init_window])
    conc1 = Reshape((1,3+character_number), name = "reshape1-%i" % t)(conc1)
    h1, _,c1  = LSTM_cell1(conc1, initial_state = [h10, c10])
    
    output_wl = window_dense(h1)
    alpha_hat, beta_hat, kappa_hat = Lambda(lambda x: [x[:,:K],x[:,K:2*K],x[:,2*K:3*K]],name="lamb2-%i" % t)(output_wl)
    alpha = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb3-%i" % t)(alpha_hat)
    beta = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb4-%i" % t)(beta_hat)
    kappa = Lambda(lambda x: tf.expand_dims(tf.exp(x),axis=2),name="lamb5-%i" % t)(kappa_hat)
    kappa = Add(name = "add1-%i" % t)([kappa,init_kappa])
    un = Lambda(lambda x: tf.square(x-u), name = "lamb6-%i" % t)(kappa)
    deux = Multiply(name = "mult1-%i" % t)([beta,un])
    trois = Lambda(lambda x: tf.exp(-x), name = "lamb7-%i" % t)(deux)
    quatro = Multiply(name = "mult2-%i" % t)([alpha,trois])
    phi = Lambda(lambda x: tf.reduce_sum(x,axis = 1), name = "lamb8-%i" % t)(quatro)
    product = Lambda(lambda x: x[:,:-1])(phi)
    window = Dot(axes = [1,1], name = "dot1-%i" % t)([product,c_vec]) 
    
    conc2 = Concatenate(axis = 1, name = "conc2-%i" % t)([X,h1,window])
    conc2 = Reshape((1,3+character_number+400), name = "reshape2-%i" % t)(conc2)
    h2,_,c2 = LSTM_cell2(conc2, initial_state = [h20, c20])

    
    conc3 = Concatenate(axis = 1, name = "conc3-%i" % t)([X,h2,window])
    conc3 = Reshape((1,3+character_number+400), name = "reshape3-%i" % t)(conc2)
    h3,_,c3 = LSTM_cell3(conc3, initial_state = [h30, c30])
    
    h = Concatenate(axis=1, name = "conc4-%i" % t)([h1,h2,h3])
    y_hat = output_dense(h)
    
    model = Model(inputs = [X,h10,c10,h20,c20,h30,c30,c_vec,init_window,init_kappa], outputs = [y_hat,h1,c1,h2,c2,h3,c3,window,kappa,phi])
    
    return model


infe = inference_model()
  
LSTM_cell1.set_weights(params["weights_lstm1"])
LSTM_cell2.set_weights(params["weights_lstm2"])
LSTM_cell3.set_weights(params["weights_lstm3"])
window_dense.set_weights(params["window_weights"])
output_dense.set_weights(params["output_weights"])



h1 = np.zeros((1,400))
c1 = np.zeros((1,400))
h2 = np.zeros((1,400))
c2 = np.zeros((1,400))
h3 = np.zeros((1,400))
c3 = np.zeros((1,400))
window = np.zeros((1,character_number))
kappa = np.zeros((1,K,1))
x = np.zeros((1,3))
L = [x,h10,c10,h20,c20,h30,c30,C,window,kappa]


strokes = [[0,0,0]]
while True:
    
    y_hat,h1,c1,h2,c2,h3,c3,window,kappa,phi = infe.predict(L)
    if phi(U+1) > phi(u):
        break
    x = sample(y_hat)
    strokes.append(x)

def sample(y_hat):
    end_of_stroke = 1 / (1 + np.exp(y_hat[0,0]))
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = np.split(y_hat[0,1:],6,axis=0) #shape = [20,]
    pi = np.exp(pi_hat) / np.sum(np.exp(pi_hat)) #shape = [20,]
    sigma1 = np.exp(sigma1_hat)#shape = [M,]
    sigma2 = np.exp(sigma2_hat)#shape = [M,]
    mu1 = mu1_hat#shape = [M,]
    mu2 = mu2_hat#shape = [M,]
    rho = np.tanh(rho_hat) #shape = [M,]
    accuracy = 0
    for m in range(M):
        accuracy += pi[m]
        if accuracy > 0.5:
            x1,x2= np.random.multivariate_normal([mu1[m], mu2[m]],
                   [[np.square(sigma1[m]), rho[m]*sigma1[m] * sigma2[m]],[rho[m]*sigma1[m]*sigma2[m], np.square(sigma2[m])]])
            break
    if end_of_stroke > 0.5:
        x0 = 1
    else:
        x0 = 0
    x = [x0,x1,x2]
    return np.array(x)



   ################### Work in progress ################################
