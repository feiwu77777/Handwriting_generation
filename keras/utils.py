import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def sample(y_hat):
    end_of_stroke = 1 / (1 + np.exp(y_hat[0,0]))
    pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = np.split(y_hat[0,1:],6,axis=0) #shape = [20,]
    pi = np.exp(pi_hat) / np.sum(np.exp(pi_hat)) #shape = [20,]
    sigma1 = np.exp(sigma1_hat)#shape = [M,]
    sigma2 = np.exp(sigma2_hat)#shape = [M,]
    mu1 = mu1_hat#shape = [M,]
    mu2 = mu2_hat#shape = [M,]
    rho = np.tanh(rho_hat) #shape = [M,]
    g = np.random.choice(np.arange(M), p = Pi)
    x1,x2= np.random.multivariate_normal([Mu1[g], Mu2[g]],
           [[np.square(sig1[g]), Rho[g]*sig1[g] * sig2[g]],[Rho[g]*sig1[g]*sig2[g], np.square(sig2[g])]])
    if end_of_stroke > 0.5:
        x0 = 1
    else:
        x0 = 0
    x = np.array([x0,x1,x2])
    x = x.reshape((1,3))
    return x


def calculate_pi(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x))



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
