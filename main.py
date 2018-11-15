U = 
character_number =
cells_number = 400 
batch_size = 32
K = 10
T = 
M = 20
biais = 0
learning_rate = 


def forward_prop(cells_number,batch_size,K,T,x_list,c_vec):
    lstm1 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm2 = tf.nn.rnn_cell.LSTMCell(cells_number)
    lstm3 = tf.nn.rnn_cell.LSTMCell(cells_number)
    init_lstm1 = lstm1.zero_state(batch_size, dtype=tf.float32) #shape = 
    init_lstm2 = lstm2.zero_state(batch_size, dtype=tf.float32)
    init_lstm3 = lstm2.zero_state(batch_size, dtype=tf.float32)
    lstm1_state = init_lstm1
    lstm2_state = init_lstm2
    lstm3_state = init_lstm3
    h1_list = []
    h2_list = []
    h3_list = []
    weights_h1_p = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    bias_p = tf.zeros([3*K])
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
        phi = tf.reduce_sum(alpha*tf.exp(-beta*tf.square(kappa-u),axis=1,keep_dims=True)) #(kappa-u).shape = [batch_size,K,U], phi.shape = [batch_size,1,U] 
        w_list = []
        for batch in range(batch_size):
            w_list.append(tf.matmul(phi[batch,:,:], c_vec[batch,:,:]))
        window = tf.concat(w_list,axis=0) #shape = [batch_size,character_number]

        with tf.variable_scope("lstm2", reuse=share):
            h2, lstm2_state = lstm2(tf.concat([x_list[t], h1, window],axis=1), lstm2_state) #[x_list[t] h1 window].shape = [batch_size,3+cells_number+character_number]
        h2_list.append(h2) #h2.shape = [batch_size,cells_number] 
        with tf.variable_scope("lstm3", reuse=share):
            h3, lstm3_state = lstm3(tf.concat([x_list[t], h2, window],axis=1), lstm2_state) #[x_list[t], h2, window].shape = [batch_size,3+cells_number+character_number]
        h3_list.append(h3) #h3.shape = [batch_size,cells_number]
        reuse = True


    h1_list = tf.reshape(tf.concat(h1_list,axis=0),[batch_size,T,cells_number])
    h2_list = tf.reshape(tf.concat(h2_list,axis=0),[batch_size,T,cells_number])
    h3_list = tf.reshape(tf.concat(h3_list,axis=0),[batch_size,T,cells_number])
    h_list = tf.concat([h1_list,h2_list,h3_list],axis=2) #shape = [batch_size,T,3*cells_number]
    return h_list



def compute_loss(cells_number,N_out,batch_size,h_list,y):
    weights_h1_y = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    weights_h2_y = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    weights_h3_y = tf.Variable(tf.truncated_normal([cells_number, 3*K], 0.0, 0.075, dtype=tf.float32))
    biais_y = tf.zeros([N_out])
    weights_y = tf.concat([weights_h1_y,weights_h2_y,weights_h3_y],axis=0) #shape = [3*cell_numbers,N_out]

    loss=0
    for batch in batch_size:
        y_hat = tf.nn.xw_plus_b(h_list[batch,:,:],weights_y,biais_y) #shape = [T,N_out]
        y1, y2, y_end_of_stroke = tf.split(y[batch,:,:],3,axis=1) #shape = [T,1]

        end_of_stroke = 1 / (1 + tf.exp(y_hat[:, 0])) #shape = [T,1]
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = tf.split(y_hat[:,1:],6,axis=1) #shape = [T,M]
        pi = tf.exp(pi_hat) / expand(tf.reduce_sum(tf.exp(pi_hat),axis=1), 1, M)
        sigma1 = tf.exp(sigma1_hat)
        sigma2 = tf.exp(sigma2_hat)
        rho = tf.tanh(rho_hat) #shape = [T,M]

        gaussian = pi * bivariate_gaussian(expand(y1, 1, M), expand(y2, 1, M),mu1, mu2, sigma1, sigma2, rho) #shape= [T,M]
        loss_gaussian = tf.reduce_sum(-tf.log(tf.reduce_sum(gaussian,axis=1)))
        loss_bernoulli = -tf.reduce_sum(tf.log(end_of_stroke)*y_end_of_stroke + tf.log(1-end_of_stroke)*(1 - y_end_of_stroke))

        loss += (loss_gaussian + loss_bernoulli)

    loss = loss/batch_size
    return loss 



def model(X_train, Y_train, C_vec,epochs = 10, batch_size = 32, print_cost = True):
    ops.reset_default_graph()
    cost = []

    c_vec = tf.placeholder([None, U, character_number], dtype=tf.float32)
    x = tf.placeholder([None, T, 3], dtype=tf.float32) #shape = [batch_size,T,3]
    y = tf.placeholder([None, T, 3], dtype=tf.float32)
    x = tf.split(x,T,axis=1) #len(x) = T, x[0] is the batch with all x0 across batch_size examples.
    x_list = [tf.squeeze(x_i, axis=1) for x_i in x] #x_i.shape = [batch_size,1,3]

    h_list = forward_prop(cells_number,batch_size,K,T,x_list,c_vec)
    cost = compute_loss(cells_number,N_out,batch_size,h_list，y)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001,decay=0.9,momentum=0.95,epsilon=0.0001)
    gvs = optimizer.compute_gradients(cost)
    capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
    opti = optimizer.apply_gradients(capped_gvs)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess,run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_batches = int(m / batch_size)
            batches = random_batches(X_train, Y_train, C_vec, batch_size)
            for batch in batches:
                (batch_X, batch_Y, batch_C) = batch
                _ , batch_cost = sess.run([opti,cost],feed_dict = {x:batch_X, y:batch_Y, c_vec:batch_C})
                epoch_cost += batch_cost / num_batches
        if print_cost == True and epoch % 100 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(epoch_cost)


def random_batches(X, Y, C, batch_size = 32, seed = 0):
    m = X.shape[0]              
    batches = []
    np.random.seed(seed)
 
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    shuffled_C = C[permutation]

    num_complete_batches = np.floor(m/batch_size)
    for k in range(0, num_complete_batches):
        batch_X = shuffled_X[k*batch_size : k*batch_size + batch_size,:]
        batch_Y = shuffled_Y[k*batch_size : k*batch_size + batch_size,:]
        batch_C = shuffled_C[k*batch_size : k*batch_size + batch_size]
        batch = (batch_X, batch_Y, batch_C)
        batches.append(batch)

    if m%batch_size != 0:
        batch_X = shuffled_X[num_complete_batches*batch_size : m, :]
        batch_Y = shuffled_Y[num_complete_batches*batch_size : m, :]
        batch_C = shuffled_C[k*batch_size : k*batch_size + batch_size]
        batch = (batch_X, batch_Y, batch_C)
        batches.append(batch)
    
    return batches
