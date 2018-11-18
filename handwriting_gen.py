import numpy as np
import tensorflow as tf


T = 400

def cut_same(strokes):  
    L = []
    for i in range(strokes.shape[0]):
        stroke = strokes[i]
        array = []
        array.append(stroke[0,:])
        for j in range(1,stroke.shape[0]):
            if sum(stroke[j,:] == np.zeros((3,))) == 3:
                pass
            else:
                array.append(stroke[j,:])
        array = np.array(array)
        L.append(array)
    return L


def filter_data(strokes,texts):
    indices = []
    for i in range(len(strokes)):
        if strokes[i].shape[0] <= T:
            indices.append(i)
    texts = np.array(texts)[indices]
    strokes = np.array(strokes)[indices]
    return strokes,texts


def preprocess_strokes(strokes):
    m = np.zeros((strokes.shape[0],T,3),dtype = "float32")
    for i in range(len(strokes)):
        m[i,:strokes[i].shape[0],:] = strokes[i]
    return m




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
            
    return C_vec


def load_data():
    strokes = np.load('data/strokes.npy',encoding = 'latin1')
    with open('data/sentences.txt') as f:
        texts = f.readlines()
    strokes = cut_same(strokes)
    strokes,texts = filter_data(strokes,texts)
    strokes = preprocess_strokes(strokes)
    Y = np.zeros(strokes.shape,dtype ="float32")
    for i in range(Y.shape[0]):
        Y[i,:T-1,:] = strokes[i,1:T,:]
    C_vec = preprocess_text(texts)
    return strokes,Y,C_vec


X,Y,C = load_data()    


           
           
def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)],axis =dim)


def random_batches(X, Y, C, batch_size):
    m = X.shape[0]              
    batches = []
 
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:]
    shuffled_Y = Y[permutation,:,:]
    shuffled_C = C[permutation,:,:]

    num_complete_batches = int(np.floor(m/batch_size))
    for k in range(num_complete_batches):
        batch_X = shuffled_X[k*batch_size : k*batch_size + batch_size,:,:]
        batch_Y = shuffled_Y[k*batch_size : k*batch_size + batch_size,:,:]
        batch_C = shuffled_C[k*batch_size : k*batch_size + batch_size,:,:]
        batch = (batch_X, batch_Y, batch_C)
        batches.append(batch)

#    if m%batch_size != 0:
#        batch_X = shuffled_X[num_complete_batches*batch_size : m, :,:]
#        batch_Y = shuffled_Y[num_complete_batches*batch_size : m, :,:]
#        batch_C = shuffled_C[num_complete_batches*batch_size : m,:,:]
#        batch = (batch_X, batch_Y, batch_C)
#        batches.append(batch)
    
    return batches


U = C.shape[1]
character_number = C.shape[2]
cells_number = 400 
batch_size = 16
K = 10
M = 20

def forward_prop(cells_number,batch_size,K,T,M,x_list,c_vec):
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm3 = tf.nn.rnn_cell.LSTMCell(cells_number)
    init_lstm1 = lstm1.zero_state(batch_size, dtype=tf.float32) 
    init_lstm2 = lstm2.zero_state(batch_size, dtype=tf.float32)
    init_lstm3 = lstm2.zero_state(batch_size, dtype=tf.float32)
    lstm1_state = init_lstm1
    lstm2_state = init_lstm2
    lstm3_state = init_lstm3
    h1_list = []
    h2_list = []
    h3_list = []
    weights_h1_p = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    biais_p = tf.zeros([3*K])
    init_kappa = tf.zeros([batch_size, K, 1])  
    init_window = tf.zeros([batch_size, character_number])
    window = init_window
    kappa_prev = init_kappa
    u = expand(expand(np.array([i for i in range(1,U+1)], dtype=np.float32), 0, K), 0, batch_size)  #u.shape = (batch_size,K,U) each line being [i for i in range(1,U+1)] 

    reuse = False
    for t in range(T):
        with tf.variable_scope("lstm1", reuse=reuse):
            h1, lstm1_state = lstm1(tf.concat([x_list[t], window],axis=1), lstm1_state) #[x_list[t] window].shape = [batch_size, 3 + character_number]
        h1_list.append(h1) #h1.shape = [batch_size, cell_numbers]
        k_gaussian = tf.nn.xw_plus_b(h1, weights_h1_p, biais_p) #shape = [batch_size,3*K]
        alpha_hat, beta_hat, kappa_hat = tf.split(k_gaussian,3,axis=1) #shape = [batch_size,K]
        alpha = tf.expand_dims(tf.exp(alpha_hat), 2) #shape = [batch_size,K,1]
        beta = tf.expand_dims(tf.exp(beta_hat), 2) #shape = [batch_size,K,1]
        kappa = kappa_prev + tf.expand_dims(tf.exp(kappa_hat), 2) #shape = [batch_size,K,1]
        kappa_prev = kappa
        phi = tf.reduce_sum(alpha*tf.exp(-beta*tf.square(kappa-u)),axis=1,keepdims=True) #(kappa-u).shape = [batch_size,K,U], phi.shape = [batch_size,1,U] 
        w_list = []
        for batch in range(batch_size):
            w_list.append(tf.matmul(phi[batch,:,:], c_vec[batch,:,:]))
        window = tf.concat(w_list,axis=0) #shape = [batch_size,character_number]

        with tf.variable_scope("lstm2", reuse=reuse):
            h2, lstm2_state = lstm2(tf.concat([x_list[t], h1, window],axis=1), lstm2_state) #[x_list[t] h1 window].shape = [batch_size,3+cells_number+character_number]
        h2_list.append(h2) #h2.shape = [batch_size,cells_number] 
        with tf.variable_scope("lstm3", reuse=reuse):
            h3, lstm3_state = lstm3(tf.concat([x_list[t], h2, window],axis=1), lstm2_state) #[x_list[t], h2, window].shape = [batch_size,3+cells_number+character_number]
        h3_list.append(h3) #h3.shape = [batch_size,cells_number]
        reuse = True


    h1_list = tf.reshape(tf.concat(h1_list,axis=0),[batch_size,T,cells_number])
    h2_list = tf.reshape(tf.concat(h2_list,axis=0),[batch_size,T,cells_number])
    h3_list = tf.reshape(tf.concat(h3_list,axis=0),[batch_size,T,cells_number])
    h_list = tf.concat([h1_list,h2_list,h3_list],axis=2) #shape = [batch_size,T,3*cells_number]
    N_out = 1 + 6*M
    weights_h1_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
    weights_h2_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
    weights_h3_y = tf.Variable(tf.truncated_normal([cells_number, N_out], 0.0, 0.075, dtype=tf.float32))
    biais_y = tf.zeros([N_out])
    weights_y = tf.concat([weights_h1_y,weights_h2_y,weights_h3_y],axis=0) #shape = [3*cell_numbers,N_out]
    weights_y = expand(weights_y,0,batch_size)#shape = [batch_size,3*cell_numbers,N_out]
    y_hat = tf.nn.xw_plus_b(h_list,weights_y,biais_y) #shape = [batch_size,T,N_out]
    return y_hat


def compute_loss(cells_number,batch_size,y_hat,y):

    y1, y2, y_end_of_stroke = tf.split(y,3,axis=2) #shape = [batch_size,T,1]
    y1, y2, y_end_of_stroke = tf.squeeze(y1,axis=2),tf.squeeze(y2,axis=2),tf.squeeze(y_end_of_stroke,axis=2) #shape = [batch_size,T]
    end_of_stroke = 1 / (1 + tf.exp(y_hat[:,:, 0])) #shape = [batch_size,T]
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[:,:,1:],6,axis=2) #shape = [batch_size,T,M]
    pi = tf.exp(pi_hat) / expand(tf.reduce_sum(tf.exp(pi_hat),axis=2), 2, M) #shape = [batch_size,T,M]
    sigma1 = tf.exp(sigma1_hat)#shape = [batch_size,T,M]
    sigma2 = tf.exp(sigma2_hat)#shape = [batch_size,T,M]
    mu1 = mu1_hat#shape = [batch_size,T,M]
    mu2 = mu2_hat#shape = [batch_size,T,M]
    rho = tf.tanh(rho_hat) #shape = [batch_size,T,M]
    y1 = expand(y1, 2, M) #shape = [batch_size,T,M]
    y2 = expand(y2, 2, M) #shape = [batch_size,T,M]
    Z = tf.square((y1 - mu1) / sigma1) + tf.square((y2 - mu2) / sigma2) - 2 * rho * (y1 - mu1) * (y2 - mu2) / (sigma1 * sigma2) #shape = [batch_size,T,M]
    gaussian = pi*tf.exp(-Z / (2 * (1 - tf.square(rho)))) / (2 * np.pi * sigma1 * sigma2 * tf.sqrt(1 - tf.square(rho))) #shape = [batch_size,T,M]
    eps = 1e-20
    loss_gaussian = -tf.log(tf.reduce_sum(gaussian,axis=2) + eps) #shape = [batch_size,T]
    loss_gaussian = tf.reduce_sum(tf.reduce_sum(loss_gaussian,axis=1))
    loss_bernoulli = tf.reduce_sum(-tf.reduce_sum(tf.log(end_of_stroke + eps)*y_end_of_stroke + tf.log(1-end_of_stroke + eps)*(1 - y_end_of_stroke), axis =1))

    loss = (loss_gaussian+loss_bernoulli)/(T*batch_size)

    return loss


from tensorflow.python.framework import ops


def model(epochs, X_train, Y_train, C_vec, batch_size):
    ops.reset_default_graph()
    costs = []

    c_vec = tf.placeholder(shape = [None, U, character_number], dtype=tf.float32)
    x = tf.placeholder(shape = [None, T, 3], dtype=tf.float32) #shape = [batch_size,T,3]
    y = tf.placeholder(shape = [None, T, 3], dtype=tf.float32)
    x_list = tf.split(x,T,axis=1) #len(x) = T, x[0] is the batch with all x0 across batch_size examples.
    x_list = [tf.squeeze(x_i, axis=1) for x_i in x_list] #x_i.shape = [batch_size,1,3]

    y_hat = forward_prop(cells_number,batch_size,K,T,M,x_list,c_vec)
    cost = compute_loss(cells_number,batch_size,y_hat,y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9,momentum=0.95,epsilon=0.0001)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
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
            costs.append(epoch_cost)
        saver.save(sess, 'my_test_model',global_step=1000)
#
#model(20,X,Y,C,batch_size)
#

ops.reset_default_graph()

c_vec = tf.placeholder(shape = [None, U, character_number], dtype=tf.float32)
x = tf.placeholder(shape = [None, T, 3], dtype=tf.float32) #shape = [batch_size,T,3]
y = tf.placeholder(shape = [None, T, 3], dtype=tf.float32)
x_list = tf.split(x,T,axis=1) #len(x) = T, x[0] is the batch with all x0 across batch_size examples.
x_list = [tf.squeeze(x_i, axis=1) for x_i in x_list] #x_i.shape = [batch_size,1,3]

y_hat = forward_prop(cells_number,batch_size,K,T,M,x_list,c_vec)
cost = compute_loss(cells_number,batch_size,y_hat,y)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9,momentum=0.95,epsilon=0.0001)
gvs = optimizer.compute_gradients(cost)
capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())
for epoch in range(10):
    epoch_cost = 0
    num_batches = int(X.shape[0] / batch_size)
    batches = random_batches(X, Y, C, batch_size)
    for batch in batches:
        (batch_X, batch_Y, batch_C) = batch
        _, batch_cost = sess.run([train_op,cost],feed_dict = {x:batch_X, y:batch_Y, c_vec:batch_C})
        print("batch cost: %g" % batch_cost)
        epoch_cost += batch_cost / num_batches
    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
saver.save(sess, 'my_test_model/model.tfmodel')
sess.close()
    
    
