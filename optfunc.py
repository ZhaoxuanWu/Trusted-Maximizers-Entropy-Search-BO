"""
sample hyperparameters
sample maximum values by gumble, random features
all in python

optimization the criterion
    1 hyperparameter
        feed: Xsamples, ysamples, l, sigma, sigma0, initialxs
    multiple hyperparameters
        feed: Xsamples, ysamples, ls, sigmas, sigma0s, initialxs
    return 
        train operation
        x to query

    !covmatrix computed once and cached for each optimization


GPflow
    optimize for the correct hyperparameters
    samples of hyperparameters

Implement:
    feed: Xsamples, ysamples
    return 
        samples of maximum values
            by gumble
            by random features
                optimize for maximum from Xsamples, and some random initialization
    x to query

    optimize a function with multiple initializations
        list of x variables (initialized with initializations)
        list of optimizers for each x variables
            feed: initial values
                  Xsamples, ysamples
                  l, sigma, sigma0
"""

import tensorflow as tf 
import numpy as np 
import scipy as sp 
import time 
import scipy.stats as spst
import sys 
import utils

import matplotlib.pyplot as plt 



# draw random features, and their weights
def draw_random_init_weights_features(
            xdim, n_funcs, n_features, 
            
            xx, # (nobs, xdim)
            yy, # (nobs, 1)
            
            l, sigma, sigma0, 
            # (1,xdim), (), ()
            
            dtype=tf.float32, 
            name='random_features'):
    """
    sigma, sigma0: scalars
    l: 1 x xdim
    xx: n x xdim
    yy: n x 1
    n_features: a scalar
    different from draw_random_weights_features,
        this function set W, b, noise as Variable that is initialized randomly
        rather than sample W, b, noise from random function
    """

    n = tf.shape(xx)[0]

    xx = tf.tile( tf.expand_dims(xx, axis=0), multiples=(n_funcs,1,1) )
    yy = tf.tile( tf.expand_dims(yy, axis=0), multiples=(n_funcs,1,1) )
    idn = tf.tile(tf.expand_dims(tf.eye(n, dtype=dtype), axis=0), multiples=(n_funcs,1,1))

    # draw weights for the random features.
    W = tf.get_variable(name="{}_W".format(name), 
                shape=(n_funcs, n_features,xdim), 
                dtype=dtype, 
                initializer=tf.random_normal_initializer()) \
        * tf.tile( tf.expand_dims(tf.sqrt(l), axis=0), 
                   multiples=(n_funcs,n_features,1) )
    # W = tf.random.normal(shape=(n_funcs, n_features,xdim), dtype=dtype) * tf.tile( tf.expand_dims(tf.sqrt(l), axis=0), multiples=(n_funcs,n_features,1) )
    # n_funcs x n_features x xdim

    b = 2.0 * np.pi \
        * tf.get_variable(
            name="{}_b".format(name), 
            shape=(n_funcs,n_features,1), 
            dtype=dtype, 
            initializer=tf.random_uniform_initializer(minval=0., maxval=1.))
    # b = 2.0 * np.pi * tf.random.uniform(shape=(n_funcs,n_features,1), dtype=dtype)
    # n_funcs x n_features x 1

    # compute the features for xx.
    Z = tf.cast(tf.sqrt(2.0 * sigma / n_features), dtype=dtype)\
        * tf.cos( tf.matmul(W, xx, transpose_b=True)
                + tf.tile(b, multiples=(1,1,n) ))
    # n_funcs x n_features x n

    # draw the coefficient theta.
    noise = tf.get_variable(
                name="{}_noise".format(name), 
                shape=(n_funcs,n_features,1), 
                dtype=dtype, 
                initializer=tf.random_normal_initializer())
    # noise = tf.random.normal(shape=(n_funcs,n_features,1))
    # n_funcs x n_features x 1


    def true_clause():
        Sigma = tf.matmul(Z, Z, transpose_a=True) + sigma0 * idn
        # n_funcs x n x n of rank n or n_features

        mu = tf.matmul(tf.matmul(Z, utils.multichol2inv(Sigma, n_funcs, dtype=dtype)), yy)
        # n_funcs x n_features x 1

        # tf.linalg.eigh returns None sometimes!!!
        e, v = tf.linalg.eigh(Sigma)

        # e = tf.linalg.eigvalsh(Sigma)
        e = tf.expand_dims(e, axis=-1)
        # n_funcs x n x 1

        r = tf.reciprocal(tf.sqrt(e) * (tf.sqrt(e) + tf.sqrt(sigma0)))
        # n_funcs x n x 1

        theta = noise \
            - tf.matmul(Z, 
                        tf.matmul(v, 
                                r * tf.matmul(v, 
                                                tf.matmul(Z, noise, transpose_a=True), 
                                                transpose_a=True))) \
            + mu
        # n_funcs x n_features x 1

        return theta 


    def false_clause():
        Sigma = utils.multichol2inv( tf.matmul(Z, Z, transpose_b=True) / sigma0 
                        + tf.tile(tf.expand_dims(tf.eye(n_features, dtype=dtype), axis=0), multiples=(n_funcs,1,1)), 
                        n_funcs, dtype=dtype)

        mu = tf.matmul(tf.matmul(Sigma, Z), yy) / sigma0

        theta = mu + tf.matmul(tf.cholesky(Sigma), noise)
        return theta


    # theta = tf.cond(
    #             pred=tf.less(n, n_features),
    #             true_fn=true_clause,
    #             false_fn=false_clause
    #         )
    print("Need to debug the sampling of theta, W, b in optfunc.py:draw_random_init_weights_features")
    theta = false_clause()

    return theta, W, b


def make_function_sample(x, n_features, sigma, theta, W, b, dtype=tf.float32):
    fval = tf.squeeze( tf.sqrt(2.0 * sigma / n_features) \
                * tf.matmul(theta,
                            tf.cos( tf.matmul(W, 
                                              x, 
                                              transpose_b=True) 
                                    + b ), 
                            transpose_a=True) )
    # x must be a 2d tensor
    # return: n_funcs x tf.shape(x)[0]
    #      or (tf.shape(x)[0],) if n_funcs = 1

    return fval


def duplicate_function_with_multiple_inputs(f, n_inits, xmin=-np.infty, xmax=np.infty, dtype=tf.float32):

    xs_list = [None] * n_inits
    fvals = [None] * n_inits

    for i in range(n_inits):
        xs_list[i] = tf.get_variable(shape=(1,xdim), dtype=dtype, name='{}_{}'.format(name, i),
                                constraint=lambda x: tf.clip_by_value(x, xmin, xmax))
        fvals[i] = f(xs_list[i])

    fvals = tf.squeeze(tf.stack(fvals))
    xs = tf.stack(xs_list)
    return xs, xs_list, fvals


# find maximum of a function with multiple initializers
# a function is a tensor, so this function can be used in the above function
def find_maximum_with_multiple_init_tensor(xs_list, fvals, n_inits, xdim, optimizer, dtype=tf.float32):
    """
    # xmin=-np.infty, xmax=np.infty,
    xs: list of size n_inits of (1,xdim)
    fvals: (n_inits,): function value with inputs are xs tensor
    initializers: n_inits x xdim
    """
    # initializers: n x d
    # func: a tensor function 
    #     input:  tensor n x d 
    #     output: tensor n x 1
    # n_inits: scalar (not a tensor)
    """
    returns:
        vals: shape = (n_inits,)
        invars: shape = (n_inits,xdim)
        maxval: scalar
        maxinvar: shape= (xdim,)
    """

    trains = [None] * n_inits

    for i in range(n_inits):
        trains[i] = optimizer.minimize(-fvals[i], var_list=[xs_list[i]])

    max_idx = tf.argmax(fvals)
    return trains, max_idx


def find_maximum_list_of_funcs(xdim, n_inits, n_funcs, xs, xs_list, fvals, optimizer, dtype=tf.float32):
    """
    xs: shape=(n_funcs, n_inits, xdim)
    xs_list: list of n_funcs lists of size n_inits of tensor (1,xdim)
    fvals: tensor of shape (n_funcs, n_inits)
    #initializers: (n_funcs, n_inits, xdim)
    """
    train_all = []
    max_val_all = [None] * n_funcs
    max_input_all = [None] * n_funcs
    max_idx_all = []

    for i in range(n_funcs):
        trains, max_idx = find_maximum_with_multiple_init_tensor(xs_list[i], fvals[i,...], n_inits, xdim, dtype=dtype, optimizer=optimizer)

        train_all.extend(trains)
        max_idx_all.append(max_idx)

        max_input_all[i] = xs[i,max_idx,...]
        max_val_all[i] = fvals[i,max_idx]

    max_val_arr = tf.reshape(tf.stack(max_val_all), shape=(n_funcs,))
    max_input_arr = tf.reshape(tf.stack(max_input_all), shape=(n_funcs,xdim))
    max_idx_arr = tf.reshape(tf.stack(max_idx_all), shape=(n_funcs,))

    return train_all, max_val_arr, max_input_arr, max_idx_arr


def gen_fval_xs(funcs, n_inits, xdim, xmin, xmax, dtype=tf.float32, name='test'):
    """
    if funcs is a list of functions
        return xs: nfuncs x n_inits x xdim
               xs_list: list of nfuncs lists of n_inits tensors of size (1,xdim)
               fvals: nfuncs x n_inits
    else:
        return xs: n_inits x xdim
               xs_list: list of n_inits tensors of size (1,xdim)
               fvals: n_inits,
    """
    if isinstance(funcs, list):
        print("List of functions")

        n_funcs = len(funcs)
        xs_list = [[tf.get_variable(shape=(1,xdim), dtype=dtype, name='{}_{}_{}'.format(name, i, j),
                                    constraint=lambda x: tf.clip_by_value(x, xmin, xmax)) for i in range(n_inits)] for j in range(n_funcs)]

        xs = []
        for i in range(n_funcs):
            xs.append( tf.stack(xs_list[i]) )
        xs = tf.stack(xs)

        fvals = []
        for i in range(n_funcs):
            fvals_i = []
            for j in range(n_inits):
                fvals_i.append( tf.squeeze(funcs[i](xs_list[i][j])) )

            fvals.append( tf.squeeze(tf.stack(fvals_i)) )

        fvals = tf.stack(fvals)

    else: # funcs is a function
        print("A function")
        xs_list = [tf.get_variable(shape=(1,xdim), dtype=dtype, name='test_func_mul_init_{}'.format(i),
                                    constraint=lambda x: tf.clip_by_value(x, xmin, xmax)) for i in range(n_inits)]

        fvals = [funcs(x) for x in xs_list]

        xs = tf.reshape(tf.concat(xs_list, axis=0), shape=(n_inits, xdim))
        fvals = tf.squeeze(tf.concat(fvals, axis=0))

    return xs, xs_list, fvals



# draw random features, and their weights
def draw_random_init_weights_features_np(
            xdim, n_funcs, n_features, 
            
            xx, # (nobs, xdim)
            yy, # (nobs, 1)
            
            l, sigma, sigma0):
            # (1,xdim), (), ()
    """
    sigma, sigma0: scalars
    l: 1 x xdim
    xx: n x xdim
    yy: n x 1
    n_features: a scalar
    different from draw_random_weights_features,
        this function set W, b, noise as Variable that is initialized randomly
        rather than sample W, b, noise from random function
    """
    n = xx.shape[0]
    l = l.reshape(1,xdim)
    
    xx = np.tile( xx.reshape(1,n,xdim), reps=(n_funcs,1,1) )
    yy = np.tile( yy.reshape(1,n,1), reps=(n_funcs,1,1) )
    idn = np.tile( np.eye(n).reshape(1,n,n), reps=(n_funcs,1,1) )

    # draw weights for the random features.
    W = np.random.randn(n_funcs, n_features, xdim) \
        * np.tile(np.sqrt(l).reshape(1,1,xdim), 
                  reps=(n_funcs, n_features, 1))
    # n_funcs x n_features x xdim

    b = 2.0 * np.pi * np.random.rand(n_funcs, n_features, 1)
    # n_funcs x n_features x 1

    # compute the features for xx.
    Z = np.sqrt(2.0 * sigma / n_features) \
        * np.cos( np.matmul(W, np.transpose(xx, (0,2,1)))
                  + np.tile(b, reps=(1,1,n)) )
    # n_funcs x n_features x n

    # draw the coefficient theta.
    noise = np.random.randn(n_funcs, n_features, 1)
    # n_funcs x n_features x 1

    if n < n_features:
        Sigma = np.matmul(np.transpose(Z, (0,2,1)), Z) \
                + sigma0 * idn
        # n_funcs x n x n

        mu = np.matmul( np.matmul(Z, np.linalg.inv(Sigma) ), yy)
        # n_funcs x n_features x 1

        e, v = np.linalg.eig(Sigma)
        # n_funcs, n
        # n_funcs, n, n

        e = e.reshape(n_funcs, n, 1)
        # n_funcs x n x 1

        r = 1.0 / (np.sqrt(e) * (np.sqrt(e) + np.sqrt(sigma0)))
        # n_funcs x n x 1

        theta = noise \
                - np.matmul(Z, 
                            np.matmul(v, 
                                      r * np.matmul(np.transpose(v, (0,2,1)), 
                                                    np.matmul(np.transpose(Z,(0,2,1)), 
                                                              noise) 
                                                    )
                                     )
                            ) \
                + mu
        # n_funcs x n_features x 1
    else:
        Sigma = np.linalg.inv(
            np.matmul(Z, np.transpose(Z,(0,2,1))) / sigma0
            + np.tile( np.eye(n_features).reshape(1,n_features,n_features), reps=(n_funcs,1,1) )
        )
        mu = np.matmul( np.matmul(Sigma,Z), yy ) / sigma0

        theta = mu + np.matmul(np.linalg.cholesky(Sigma), noise)

    return theta, W, b


# for testing draw_random_init_weights_features_np
def make_function_sample_np(x, n_features, sigma, theta, W, b):
    fval = np.squeeze(
        np.sqrt(2.0 * sigma / n_features)
        * np.matmul(theta.T,
                    np.cos(
                        np.matmul(W, x.T)
                        + b
                        )
                    ) )
    # x must be a 2d tensor
    # return: n_funcs x tf.shape(x)[0]
    #      or (tf.shape(x)[0],) if n_funcs = 1

    return fval

########################## TEST FUNCTIONS ##########################

def test_find_maximum_with_multiple_init_tensor(ntrain=10, n_inits=5, dtype = tf.float32):
    """
    Adam with 1000 iterations
    """
    tf.reset_default_graph()

    xdim = 2
    xmin = -10.
    xmax = 10.
    
    func = lambda x: tf.sin( tf.matmul(x,x,transpose_b=True) )

    initializers = tf.random.uniform(shape=(n_inits,xdim), dtype=dtype) * (xmax - xmin) + xmin

    xs, xs_list, fvals = gen_fval_xs(func, n_inits, xdim, xmin, xmax, dtype=dtype, name='test_max_f_mulinit')

    assign_inits = []
    for i in range(n_inits):
        assign_inits.append( tf.assign(xs_list[i], tf.reshape(initializers[i,:], shape=(1,xdim))) )

    optimizer = tf.train.AdamOptimizer()

    trains, max_idx = find_maximum_with_multiple_init_tensor(xs_list, fvals, n_inits, xdim, dtype=dtype, name='find_maximum_multiple_inputs', optimizer=optimizer)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(assign_inits)

        xs_val, xs_list_val, fvals_val = sess.run([xs, xs_list, fvals])
        print('')
        print('input = ', xs_val[0,...])
        print(xs_list_val[0])
        print('output = ', fvals_val[0])

        for i in range(ntrain):
            _, max_idx_val, xs_val, xs_list_val, fvals_val = sess.run([trains, max_idx, xs, xs_list, fvals])

        print('')
        print('input = ', xs_val[0,...])
        print(xs_list_val[0])
        print('output = ', fvals_val[0])


def test_find_maximum_list_of_funcs(ntrain, n_inits=5, dtype = tf.float32):
    """
    Adam with 1000 iterations
    """
    tf.reset_default_graph()

    xdim = 2
    xmin = -10.
    xmax = 10.
    
    funcs = [lambda x: tf.sin( tf.matmul(x,x,transpose_b=True) ),
             lambda x: tf.cos( tf.matmul(x,x,transpose_b=True)) + 2.0 ]
    n_funcs = len(funcs)

    initializers = tf.random.uniform(shape=(n_funcs, n_inits, xdim), dtype=dtype) * (xmax - xmin) + xmin

    xs, xs_list, fvals = gen_fval_xs(funcs, n_inits, xdim, xmin, xmax, dtype=dtype, name='test_max_listf')

    assign_inits = []
    for i in range(n_funcs):
        for j in range(n_inits):
            assign_inits.append( tf.assign(xs_list[i][j], tf.reshape(initializers[i,j,:], shape=(1,xdim))) )

    optimizer = tf.train.AdamOptimizer()

    trains, max_vals, max_inputs = find_maximum_list_of_funcs(xdim, n_inits, n_funcs, xs, xs_list, fvals, optimizer=optimizer, dtype=dtype, name="opt_list_funcs")


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(assign_inits)


        for i in range(ntrain):
            _, max_vals_val, max_inputs_val, xs_val, fvals_val = sess.run([trains, max_vals, max_inputs, xs, fvals])

            if i == ntrain - 1 or i == 1:
                print('')
                print('max input = ', max_inputs_val)
                print('max output = ', max_vals_val)
                print('xs = ', xs_val)
                print('fvals = ', fvals_val)


def test_draw_random_weights_features(n_funcs=10, n_features=500, dtype=tf.float32, randomize_funcs=False, func_param_plc=False, plot=True):
    tf.reset_default_graph()

    xdim = 1

    xx = tf.placeholder(shape=(None, xdim), dtype=dtype, name='xx')
    yy = tf.placeholder(shape=(None, 1), dtype=dtype, name='yy')
    l = tf.get_variable(shape=(1,xdim), dtype=dtype, name='l')
    sigma = tf.get_variable(shape=(), dtype=dtype, name='sigma')
    sigma0 = tf.get_variable(shape=(), dtype=dtype, name='sigma0')

    x = tf.placeholder(shape=(None, xdim), dtype=dtype, name='x')


    thetas, Ws, bs = draw_random_init_weights_features(xdim, n_funcs, n_features, xx, yy, l, sigma, sigma0, dtype=dtype, name='random_features')

    if func_param_plc:
        thetas_plc = tf.placeholder(shape=(n_funcs, n_features, 1), dtype=dtype, name='theta')
        Ws_plc = tf.placeholder(shape=(n_funcs, n_features, xdim), dtype=dtype, name='W')
        bs_plc = tf.placeholder(shape=(n_funcs, n_features, 1), dtype=dtype, name='b')        

        fvals = []
        for i in range(n_funcs):
            fvals.append( make_function_sample(x, n_features, sigma, thetas_plc[i,...], Ws_plc[i,...], bs_plc[i,...], dtype=dtype) )
        fvals = tf.stack(fvals)

    else:
        fvals = []
        for i in range(n_funcs):
            fvals.append( make_function_sample(x, n_features, sigma, thetas[i,...], Ws[i,...], bs[i,...], dtype=dtype) )
        fvals = tf.stack(fvals)



    xx_val = np.array([[0.], [1.], [4.], [5.]])
    yy_val = np.array([[-5.], [0.5], [3.0], [0.3]])
    l_val = np.array([[10.0]])
    # xx_val = np.array([[0., 1.], [1.,1.], [4.,1.], [5.,1.1]])
    # yy_val = np.array([[-5.], [0.5], [3.0], [0.3]])
    # l_val = np.array([[10.0, 5.0]])

    sigma_val = 2.0
    sigma0_val = 1e-3

    x_val = np.linspace(0., 5., 100).reshape(-1,1)
    # x_val = np.array(list(zip( np.linspace(0., 5., 50), np.ones(50) )))

    n_plot_funcs = 3
    func_vals = []
    fixed_func_vals = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        l.load(l_val, sess)
        sigma.load(sigma_val, sess)
        sigma0.load(sigma0_val, sess)

        start = time.time()

        feed_dict = {xx: xx_val, yy: yy_val, x: x_val}

        if func_param_plc and not randomize_funcs:
            thetas_val, Ws_val, bs_val = sess.run([thetas, Ws, bs], feed_dict=feed_dict)

        for i in range(n_plot_funcs):
            if randomize_funcs:
                sess.run(tf.global_variables_initializer())

                if func_param_plc:
                    thetas_val, Ws_val, bs_val = sess.run([thetas, Ws, bs], feed_dict=feed_dict)
            
            if func_param_plc:
                func_val = sess.run(fvals, feed_dict=feed_dict)
            else:                    
                func_val = sess.run(fvals, feed_dict=feed_dict)

            func_vals.append(func_val)
        print("End in {:.4f}s".format(time.time() - start))

    if plot:
        fig, axs = plt.subplots(n_plot_funcs,2)
        for j in range(n_plot_funcs):
            for i in range(n_funcs):
                axs[j,0].plot(np.squeeze(x_val), np.squeeze(func_vals[j][i,...]))

            axs[j,0].scatter(xx_val, yy_val)

        plt.show()


def test_maximize_random_weights_features(n_funcs=10, ntrain=100, n_inits=5, n_features=500, dtype=tf.float32, plot=True):
    # TODO: not working, only optimize for 1 functions...
    tf.reset_default_graph()

    xdim = 1
    xmin = 0.
    xmax = 5.

    xx = tf.placeholder(shape=(None, xdim), dtype=dtype, name='xx')
    yy = tf.placeholder(shape=(None, 1), dtype=dtype, name='yy')
    l = tf.get_variable(shape=(1,xdim), dtype=dtype, name='l')
    sigma = tf.get_variable(shape=(), dtype=dtype, name='sigma')
    sigma0 = tf.get_variable(shape=(), dtype=dtype, name='sigma0')

    x = tf.placeholder(shape=(None, xdim), dtype=dtype, name='x')


    thetas, Ws, bs = draw_random_init_weights_features(xdim, n_funcs, n_features, xx, yy, l, sigma, sigma0, dtype=dtype, name='random_features')

    thetas_plc = tf.get_variable(shape=(n_funcs, n_features, 1), dtype=dtype, name='theta')
    Ws_plc = tf.get_variable(shape=(n_funcs, n_features, xdim), dtype=dtype, name='W')
    bs_plc = tf.get_variable(shape=(n_funcs, n_features, 1), dtype=dtype, name='b')        


    # funcs = [lambda x: tf.sin( x ),
    #          lambda x: tf.cos(x) + 2.0 ]
    # funcs_np = [lambda x: np.sin( x ),
    #             lambda x: np.cos(x) + 2.0 ]
    # n_funcs = len(funcs)
    


    # optimizing function samples
    initializers = tf.random.uniform(shape=(n_funcs, n_inits, xdim), dtype=dtype) * (xmax - xmin) + xmin

    funcs = []
    for i in range(n_funcs):
        funcs.append( (lambda x: make_function_sample(x, n_features, sigma, thetas_plc[i,...], Ws_plc[i,...], bs_plc[i,...], dtype=dtype)) )

    fvals = []
    for i in range(n_funcs):
        fvals.append( funcs[i](x) )
    fvals = tf.stack(fvals)

    print("# of funcs: ", len(funcs))
    sys.stdout.flush()

    # xs, xs_list, opt_fvals = gen_fval_xs(funcs, n_inits, xdim, xmin, xmax, dtype=dtype, name='test_max_listf')
    print("IMPORTANT: cannot use gen_fval_xs, only the last function would be used")
    name = 'test_max_listf'
    xs_list = [[tf.get_variable(shape=(1,xdim), dtype=dtype, name='{}_{}_{}'.format(name, i, j),
                                constraint=lambda x: tf.clip_by_value(x, xmin, xmax)) for i in range(n_inits)] for j in range(n_funcs)]

    xs = []
    for i in range(n_funcs):
        xs.append( tf.stack(xs_list[i]) )
    xs = tf.stack(xs)

    opt_fvals = []
    for i in range(n_funcs):
        fvals_i = []
        for j in range(n_inits):
            fval = tf.squeeze( tf.sqrt(2.0 * sigma / n_features) \
                            * tf.matmul(thetas_plc[i,...],
                                        tf.cos( tf.matmul(Ws_plc[i,...], 
                                                          xs_list[i][j], 
                                                          transpose_b=True) 
                                                + bs_plc[i,...] ), 
                                        transpose_a=True) )
            fvals_i.append( fval )


        opt_fvals.append( tf.squeeze(tf.stack(fvals_i)) )

    opt_fvals = tf.stack(opt_fvals)




    assign_inits = []
    for i in range(n_funcs):
        for j in range(n_inits):
            assign_inits.append( tf.assign(xs_list[i][j], tf.reshape(initializers[i,j,:], shape=(1,xdim))) )

    optimizer = tf.train.AdamOptimizer()

    trains, max_vals, max_inputs, max_idx_arr = find_maximum_list_of_funcs(xdim, n_inits, n_funcs, xs, xs_list, opt_fvals, optimizer=optimizer, dtype=dtype, name="opt_list_funcs")


    xx_val = np.array([[0.], [1.], [4.], [5.]])
    yy_val = np.array([[-5.], [0.5], [3.0], [0.3]])
    l_val = np.array([[10.0]])

    sigma_val = 2.0
    sigma0_val = 1e-3

    x_val = np.linspace(xmin, xmax, 100).reshape(-1,1)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start = time.time()

        l.load(l_val, sess)
        sigma.load(sigma_val, sess)
        sigma0.load(sigma0_val, sess)

        thetas_val, Ws_val, bs_val = sess.run([thetas, Ws, bs], feed_dict={xx: xx_val,
                             yy: yy_val,
                             x: x_val})
        
        thetas_plc.load(thetas_val, sess)
        Ws_plc.load(Ws_val, sess)
        bs_plc.load(bs_val, sess)

        func_val = sess.run(fvals, feed_dict={x: x_val})

        print("End evaluating functions in {:.4f}s".format(time.time() - start))
        sys.stdout.flush()

        start = time.time()
        sess.run(assign_inits)


        xs_val, opt_fvals_val = sess.run([xs, opt_fvals])


        for i in range(ntrain):
            _, max_vals_val, max_inputs_val, max_idx_arr_val = sess.run([trains, max_vals, max_inputs, max_idx_arr])

            if i == ntrain - 1 or i == 1:
                print('')
                print('max input = ', max_inputs_val)
                print('max output = ', max_vals_val)
                print('max_idx_arr = ', max_idx_arr_val)
                # for j in range(n_funcs):
                #     print('              ', xs_val[j,max_idx_arr_val[j],...])
                #     print('            f:', opt_fvals_val[j,max_idx_arr_val[j],...])

        print("End optimizing in {:.4f}s".format(time.time() - start))
        sys.stdout.flush()

    if plot:
        fig, axs = plt.subplots()

        for i in range(n_funcs):
            axs.plot(np.squeeze(x_val), np.squeeze(func_val[i,...]), zorder=0)

        axs.scatter(xx_val, yy_val, zorder=3)
        axs.scatter(np.squeeze(max_inputs_val), max_vals_val, zorder=5, c='r')

        plt.show()

