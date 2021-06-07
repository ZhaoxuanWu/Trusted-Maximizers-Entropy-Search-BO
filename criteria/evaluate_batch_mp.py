import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import sys


import utils

# Using MI


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

        should build this covariance incrementally!
            -> add noise to x_test only!
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


def get_batch_queried_f_stat_given_data_maxidx(
                    xs, # (nx, batchsize, xdim)
                    l, sigma, sigma0,

                    ntest, nobs, xdim,

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
    batchsize = tf.shape(xs)[1]
    nmax = tf.shape(post_cov_test_fs)[0]

    Xtest_obs = tf.concat([test_xs, X], axis=0)
    # (ntest+nobs,xdim)

    # K_{xs, Xtest_obs}
    k_x_xto = utils.computeKnm(
                    tf.reshape(xs, shape=(-1,xdim)),
                    Xtest_obs,
                    l, sigma, dtype=dtype)
    # (nx * batchsize, ntest + nobs)
    k_x_xto = tf.reshape(k_x_xto, shape=(nx, batchsize, -1))
    # (nx, batchsize, ntest+nobs)

    k_x = utils.computeKmm(xs, l, sigma, nd=3, dtype=dtype)
    # (nx, batchsize, batchsize)

    invpNK = tf.tile(tf.expand_dims(invpNK, axis=0),
                     multiples=(nx,1,1))
    # (nx, ntest+nobs, ntest+nobs)

    Kq = k_x - (k_x_xto @ invpNK) @ tf.transpose(k_x_xto, perm=(0,2,1))
    # (nx,batchsize,batchsize)


    M = k_x_xto @ invpNK[:,:,:ntest]
    # (nx,batchsize,ntest)

    repY = tf.tile( tf.reshape(Y, shape=(1,nobs,1)), multiples=(nx,1,1))
    # (nx, nobs, 1)
    b = k_x_xto @ (invpNK[:,:,ntest:] @ repY)
    # (nx,batchsize,1)

    post_mean_test_fs_T = tf.transpose(post_mean_test_fs)
    # ntest, nmax
    post_mean_test_fs_T = tf.tile(tf.reshape(post_mean_test_fs_T, shape=(1, ntest, nmax)), multiples=(nx,1,1))
    # nx, ntest, nmax

    query_mean = M @ post_mean_test_fs_T + b
    # (nx,batchsize,nmax)

    query_mean = tf.transpose(query_mean, perm=[2,0,1])
    # (nmax,nx,batchsize)

    M = tf.tile(tf.expand_dims(M, axis=0),
        multiples=(nmax,1,1,1))
    # (nmax,nx,batchsize,ntest)
    Kq = tf.tile(tf.expand_dims(Kq, axis=0),
        multiples=(nmax,1,1,1))
    # (nmax,nx,batchsize,batchsize)

    post_cov_test_fs = tf.expand_dims(post_cov_test_fs, axis=1)
    # (nmax, 1, ntest, ntest)
    post_cov_test_fs = tf.tile(post_cov_test_fs,
                        multiples=(1,nx,1,1))
    # (nmax, nx, ntest, ntest)

    query_var = Kq + M @ post_cov_test_fs @ tf.transpose(M, perm=[0,1,3,2])
    # (nmax, nx, batchsize, batchsize)

    return query_mean, query_var
    # (nmax, nx, batchsize)
    # (nmax, nx, batchsize, batchsize)


def mp(x, # nx, batchsize, xdim
        ls, sigmas, sigma0s,
        X, Y, # (nobs,xdim), (nobs,1)

        xdim, nobs, nhyp,
        nysample,

        test_xs, # ntest, xdim (same for all hyp)
        max_probs_all, # nhyp, nmax

        # statistics of f-values at test inputs
        # given different maximum candidates
        post_mean_test_fs_all, # nhyp, nmax, ntest
        post_cov_test_fs_all, # nhyp, nmax, ntest, ntest

        invKobs_all, # nhyp, nobs, nobs

        # K_test_max needs to be precomputed
        # and its inverse
        # need naming convension for noisy
        # vs. noiseless K
        # and partial noisy-noiseless?
        invpNK_all, # nhyp, ntest+nobs, ntest+nobs

        dtype=tf.float32,

        niteration=10,
        use_loop=True,
        parallel_iterations=1):
    """
    ntest: # of test inputs
    nmax: # of maximum candidate in test_xs
    """
    nx = tf.shape(x)[0]
    ntest = tf.shape(post_mean_test_fs_all)[2]
    batchsize = tf.shape(x)[1]

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

        invKobs = invKobs_all[i,...] # nobs, nobs
        invpNK = invpNK_all[i,...] # ntest+nobs,ntest+nobs

        queried_meanf_given_data_maxidx, queried_covf_given_data_maxidx = \
            get_batch_queried_f_stat_given_data_maxidx(
                    x,
                    l, sigma, sigma0,

                    ntest, nobs, xdim,

                    X, # (nobs, xdim)
                    Y, # (nobs,1)
                    test_xs, # (ntest, xdim)

                    invpNK, # (ntest + nobs, ntest + nobs)

                    # statistics of f-values at test inputs
                    # given different maximum candidates
                    post_mean_test_fs, # nmax, ntest
                    post_cov_test_fs, # nmax, ntest, ntest
                    dtype=dtype)
        # (nmax, nx, batchsize)
        # (nmax, nx, batchsize, batchsize)

        noise_mat = tf.eye(batchsize, dtype=dtype) * sigma0
        noise_mat = tf.expand_dims(tf.expand_dims(noise_mat, 0), 0)
        # (1,1,batchsize,batchsize)

        queried_covy_given_data_maxidx = queried_covf_given_data_maxidx + noise_mat
        # (nmax, nx, batchsize, batchsize)

        # transform_covy_given_data_maxidx
        eigvalues, eigvects = tf.linalg.eigh(queried_covy_given_data_maxidx)
        # eigvalues: (nmax, nx, batchsize)
        # eigvects: (nmax, nx, batchsize, batchsize)

        eigvalues = tf.clip_by_value(eigvalues, clip_value_min=0., clip_value_max=np.infty)
        transform_covy_given_data_maxidx = eigvects @ tf.matrix_diag(tf.sqrt(eigvalues))
        # (nmax, nx, batchsize, batchsize)

        body = lambda j, sum_mp: [j+1, \
            sum_mp + mp_each_batch_y_sample(
                nx, batchsize, nmax, nysample,

                max_probs,

                queried_meanf_given_data_maxidx, # (nmax, nx, batchsize)
                queried_covy_given_data_maxidx, # (nmax, nx, batchsize, batchsize)
                transform_covy_given_data_maxidx, # (nmax, nx, batchsize, batchsize)
                dtype
            )]

        _, sum_mp = tf.while_loop(
            lambda j, sum_mp: j < niteration,
            body,
            (tf.constant(0), tf.zeros(shape=(nx,), dtype=dtype)),
            parallel_iterations=parallel_iterations
        )

        mp_val = sum_mp / tf.constant(niteration, dtype=dtype)

        avg_mp = avg_mp + mp_val / tf.constant(nhyp, dtype=dtype)

    return avg_mp



def mp_each_batch_y_sample(
        nx, batchsize,
        nmax,
        nysample, # only used when stochastic == True

        max_probs, # nmax,

        queried_meany_given_data_maxidx, # (nmax, nx, batchsize)
        queried_covy_given_data_maxidx, # (nmax, nx, batchsize, batchsize)
        transform_covy_given_data_maxidx, # (nmax, nx, batchsize, batchsize)
        # using eigendecomposition
        # transform_covy_given_data_maxidx @ transform_covy_given_data_maxidx.T = queried_covy_given_data_maxidx
        dtype=tf.float32
        ):

    standard_normals = tfp.distributions.MultivariateNormalDiag(
        loc = tf.zeros_like(queried_meany_given_data_maxidx),
        scale_diag = tf.ones_like(queried_meany_given_data_maxidx)
    )
    # (nmax, nx, batchsize)

    # sampling y given posterior | max_idx, data
    # shape (nysample, nmax, nx)
    standard_normal = standard_normals.sample(nysample)
    # (nysample, nmax, nx, batchsize)

    ysample = tf.reduce_mean(transform_covy_given_data_maxidx @ tf.expand_dims(standard_normal, axis=-1), axis=-1) \
            + queried_meany_given_data_maxidx
    # (nysample, nmax, nx, batchsize)

    # (1) H[y|max_idx]
    log_prob = standard_normals.log_prob(ysample)
    # (nysample, nmax, nx)
    weighted_log_prob = log_prob * tf.reshape(max_probs, shape=(1,nmax,1))

    cond_ent_y = -tf.reduce_mean( tf.reduce_sum(weighted_log_prob, axis=1), axis=0 )
    # (nx,)

    # (2) H[y]
    marginal_ysample = tf.expand_dims(ysample, 2)
    # (nysample, nmax, 1, nx, batchsize)

    # marginal_ysample = tf.tile(marginal_ysample, multiples=(1, 1, nmax, 1, 1))
    # # (nysample, nmax, nmax, nx, batchsize)

    # log_marginal_prob = normal_dists.log_prob(marginal_ysample)
    log_marginal_prob = tf.expand_dims(log_prob, 2)
    # (nysample, nmax, 1, nx)
    log_marginal_prob = tf.tile(log_marginal_prob, multiples=(1, 1, nmax, 1))
    # (nysample, nmax, nmax, nx)
    weighted_log_marginal_prob = log_marginal_prob + tf.log(tf.reshape(max_probs, shape=(1,1,nmax,1)))
    log_marginal_prob = tf.reduce_logsumexp(weighted_log_marginal_prob, axis=2)
    # (nysample,nmax,nx)

    weighted_log_marginal_prob = log_marginal_prob * tf.reshape(max_probs, shape=(1,nmax,1))
    # (nysample,nmax,nx)
    ent_y = -tf.reduce_mean(tf.reduce_sum(weighted_log_marginal_prob, axis=1), axis=0)
    # (nx,)

    mp_val = tf.reshape(ent_y - cond_ent_y, shape=(nx,))
    return mp_val
