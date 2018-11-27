from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    plt.close()
    
    
    
def expand_duplicate(x, N, dim):
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