import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import time 


import utils 


# Using MI+

def get_pNK_test_obs(
            ls, sigmas, sigma0s,

            nhyp,

            X, # (nobs, xdim)
            test_xs, # (ntest, xdim)
            dtype=tf.float32
            ):
    """
    Returns 
        The covariance between (x_test, x_obs) vs. (x_test, x_obs)
        The noise is only for x_obs part of K

        pNKs (nhyp, ntest+nobs, ntest+nobs)
        invpNKs (nhyp, ntest+nobs, ntest+nobs)
    """
    nobs = tf.shape(X)[0]
    ntest = tf.shape(test_xs)[0]
    pNKs = [None] * nhyp
    invpNKs = [None] * nhyp 

    for i in range(nhyp):

        Xtest_obs = tf.concat([test_xs, X], axis=0)

        noiselessK = utils.computeKmm(Xtest_obs, ls[i,...], sigmas[i,...], dtype=dtype)

        noisemat = tf.eye(nobs, dtype=dtype) * sigma0s[i,...]

        noisemat = tf.pad(noisemat, [[ntest,0], [ntest,0]], "CONSTANT")

        pNK = noiselessK + noisemat

        pNKs[i] = pNK
        invpNKs[i] = tf.linalg.inv(pNK)
    
    pNKs = tf.stack(pNKs)
    invpNKs = tf.stack(invpNKs)
    # (nhyp, ntest + nobs, ntest + nobs)

    return pNKs, invpNKs


def get_queried_f_stat_given_data_maxidx(
                    xs, # (nx,xdim)
                    l, sigma, sigma0,

                    ntest, nobs, 

                    X, # (nobs, xdim)
                    Y, # (nobs,1)
                    test_xs, # (ntest, xdim)

                    invpNK, # (ntest + nobs, ntest + nobs)

                    # statistics of f-values at test inputs
                    # given different maximum candidates
                    post_mean_test_fs, # nmax, ntest
                    post_cov_test_fs, # nmax, ntest, ntest            
                    dtype=tf.float32
                    ):
    nx = tf.shape(xs)[0]

    Xtest_obs = tf.concat([test_xs, X], axis=0)
    # (ntest+nobs,xdim)
    k_x_xto = utils.computeKnm(xs, Xtest_obs, l, sigma, dtype=dtype) # K_{xs, Xtest_obs}
    # (nx, ntest + nobs)

    # NOTE: only compute diagonal elements!!
    k_x = sigma * tf.ones(shape=(nx,), dtype=dtype)
    # (nx,)

    Kq = k_x - tf.reduce_sum( (k_x_xto @ invpNK) * k_x_xto, axis=1 )
    # (nx,)

    M = k_x_xto @ invpNK[:,:ntest]
    # (nx,ntest)
    b = k_x_xto @ invpNK[:,ntest:] @ tf.reshape(Y, shape=(nobs,1))
    # (nx,1)

    query_mean = M @ tf.transpose(post_mean_test_fs) + b
    # (nx,nmax)
    query_mean = tf.transpose(query_mean)
    # (nmax,nx)

    nmax = tf.shape(post_cov_test_fs)[0]

    M = tf.expand_dims(M, axis=0)
    # (1,nx,ntest)
    Kq = tf.reshape(Kq, shape=(1,nx))
    query_var = Kq + tf.reduce_sum( (
                tf.tile(M, multiples=(nmax,1,1)) 
                @ post_cov_test_fs) 
                * M, 
            axis=2)
    # (nmax,nx)

    query_var = tf.reshape(query_var, shape=(nmax,nx))

    return query_mean, query_var
    # (nmax, nx)


def mp(x, # nx, xdim
        ls, sigmas, sigma0s,
        X, Y, # (nobs,xdim), (nobs,1)

        xdim, nx, nobs, nhyp, 

        test_xs, # ntest, xdim (same for all hyp)
        max_probs_all, # nhyp, nmax

        # statistics of f-values at test inputs
        # given different maximum candidates
        post_mean_test_fs_all, # nhyp, nmax, ntest
        post_cov_test_fs_all, # nhyp, nmax, ntest, ntest

        # K_test_max needs to be precomputed
        # and its inverse
        # need naming convension for noisy
        # vs. noiseless K
        # and partial noisy-noiseless?
        invpNK_all, # nhyp, ntest+nobs, ntest+nobs

        dtype=tf.float32):
    """
    ntest: # of test inputs
    nmax: # of maximum candidate in test_xs
    """
    ntest = tf.shape(post_mean_test_fs_all)[2]
    avg_mp = tf.zeros(shape=(nx,), dtype=dtype)

    for i in range(nhyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        # sigma, sigma0: scalar
        # l: (1,xdim)

        max_probs = max_probs_all[i,...] # nmax,

        # statistics of f-values at test inputs
        # given different maximum candidates
        post_mean_test_fs = post_mean_test_fs_all[i,...] # nmax, ntest
        post_cov_test_fs = post_cov_test_fs_all[i,...] # nmax, ntest, ntest

        non_zero_prob_idxs = tf.squeeze(tf.where(max_probs > 0.))
        nmax = tf.shape(non_zero_prob_idxs)[0]
        post_mean_test_fs = tf.gather(post_mean_test_fs, non_zero_prob_idxs, axis=0)
        post_cov_test_fs = tf.gather(post_cov_test_fs, non_zero_prob_idxs, axis=0)
        max_probs = tf.gather(max_probs, non_zero_prob_idxs, axis=0)
        
        invpNK = invpNK_all[i,...] # ntest+nobs,ntest+nobs

        queried_meanf_given_data_maxidx, queried_varf_given_data_maxidx = \
            get_queried_f_stat_given_data_maxidx(
                    x,
                    l, sigma, sigma0,

                    ntest, nobs, 

                    X, # (nobs, xdim)
                    Y, # (nobs,1)
                    test_xs, # (ntest, xdim)

                    invpNK, # (ntest + nobs, ntest + nobs)

                    # statistics of f-values at test inputs
                    # given different maximum candidates
                    post_mean_test_fs, # nmax, ntest
                    post_cov_test_fs, # nmax, ntest, ntest
                    dtype=dtype)
        # (nmax,nx), (nmax,nx)

        queried_meany_given_data_maxidx = queried_meanf_given_data_maxidx
        queried_vary_given_data_maxidx = queried_varf_given_data_maxidx + sigma0

        queried_meany_given_data_maxidx0 = tf.tile( 
                tf.expand_dims(queried_meany_given_data_maxidx, axis=0),
                multiples=(nmax,1,1) )

        queried_meany_given_data_maxidx1 = tf.tile( 
                tf.expand_dims(queried_meany_given_data_maxidx, axis=1),
                multiples=(1,nmax,1) )

        queried_vary_given_data_maxidx0 = tf.tile( 
                tf.expand_dims(queried_vary_given_data_maxidx, axis=0),
                multiples=(nmax,1,1) )

        queried_vary_given_data_maxidx1 = tf.tile( 
                tf.expand_dims(queried_vary_given_data_maxidx, axis=1),
                multiples=(1,nmax,1) )

        diff_meany = queried_meany_given_data_maxidx0 - queried_meany_given_data_maxidx1
        
        mips = (0.5 * tf.log( queried_vary_given_data_maxidx1 \
                                 / queried_vary_given_data_maxidx0 )
                + 0.5 * (queried_vary_given_data_maxidx0 \
                        + diff_meany * diff_meany) \
                        / queried_vary_given_data_maxidx1
                - 0.5)
        # (nmax, nmax, nx)

        max_probs0 = tf.tile(tf.reshape(max_probs, shape=(1,nmax,1)), multiples=(nmax,1,1))
        max_probs1 = tf.tile(tf.reshape(max_probs, shape=(nmax,1,1)), multiples=(1,nmax,1))
        # (nmax, nmax, 1)

        mp_val = tf.reduce_sum(tf.reduce_sum(max_probs0 * max_probs1 * mips, axis=0), axis=0)
        # (nmax, nmax, nx)

        avg_mp = avg_mp + mp_val / tf.constant(nhyp, dtype=dtype)
    
    return avg_mp
