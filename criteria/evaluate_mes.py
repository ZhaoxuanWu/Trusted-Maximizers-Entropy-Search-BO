import tensorflow as tf 
import numpy as np 
import time 


import utils 



"""
provide:
    GP hypers: l, sigma, sigma0
               Xsamples, ysamples
    samples of ymax
    criterion: select next x
"""

def mes(x, xdim, n_max, n_hyp, Xsamples, Ysamples, ls, sigmas, sigma0s, ymaxs, invKs, dtype=tf.float32):
    """
    X: n x xdim
    Y: n x 1
    ls: nh x xdim
    sigmas: nh x 1 signal variances
    sigma0s: nh x 1 noise variances
        where nh is the number of hyperparameters
    invKs: nh x n x n
    ymaxs: nh x n_maxs
    """

    mes = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        
        invK = invKs[i,...]

        f_mean, f_var = utils.compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, invK, dtype=dtype)
        f_std = tf.sqrt(f_var)

        noise_mean = 0.0
        noise_std = tf.sqrt(sigma0)
    
        ent_f = utils.evaluate_norm_entropy(f_std, dtype=dtype)

        ent_tnorm_f = tf.zeros(shape=tf.shape(x)[0], dtype=dtype)

        for j in range(n_max):
            ent_tnorm_f = ent_tnorm_f + utils.evaluate_tnorm_entropy(f_mean, f_std, ymaxs[i,j], dtype=dtype) / tf.constant(n_max, dtype=dtype)

        mes = mes + tf.squeeze(ent_f - ent_tnorm_f) / tf.constant(n_hyp, dtype=dtype)

    return mes 

