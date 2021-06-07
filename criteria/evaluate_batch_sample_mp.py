import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import time 


import utils 


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


def get_batch_queried_f_stat_given_test_samples(
                    x, # (nx, batchsize, xdim)
                    l, sigma, sigma0,

                    xdim, batchsize, ntest, nobs, 

                    X, # (nobs, xdim)
                    Y, # (nobs,1)
                    test_xs, # (ntest, xdim)

                    # invpNK, # (ntest + nobs, ntest + nobs)
                    invpNK_test, # (ntest+nobs, ntest)
                    invpNK_obs, # (ntest+nobs, nobs)
                    # samples of f-value given different maximum candidates
                    post_test_samples, # nmax, ntest, nsample
                    dtype=tf.float32):

    nx = tf.shape(x)[0]
    nmax = tf.shape(post_test_samples)[0]
    ntest = tf.shape(post_test_samples)[1]
    nobs = tf.shape(invpNK_obs)[1]
    nsample = tf.shape(post_test_samples)[2]

    Xtest_obs = tf.concat([test_xs, X], axis=0)
    # (ntest+nobs,xdim)

    # K_{x, Xtest_obs}
    k_x_xto = utils.computeKnm(
                tf.reshape(x, shape=(-1,xdim)), 
                Xtest_obs, 
                l, sigma, 
                dtype=dtype) 
    # (nx * batchsize, ntest+nobs)
    k_x_xto = tf.reshape(k_x_xto, shape=(nx, batchsize, -1))
    # (nx, batchsize, ntest+nobs)

    k_x = utils.computeKmm(x, l, sigma, nd=3, dtype=dtype)
    # (nx,batchsize,batchsize)

    """
    query_mean = k_x_xto @ invpNK @ [post_test_samples, Y]
    k_x_xto: nx,bs,nt+no
    invpNK: nt+no,nt+no
    post_test_samples: nmax,ntest,nsample
    Y: no,1

    from observation:
        Y -> nx,no,1
        invpNK_obs -> (nx,nt+no,no)
        k_x_xto @ invpNK_obs @ Y
        nx,bs,1
    """
    Y = tf.tile( tf.expand_dims(Y, axis=0),
            multiples=(nx,1,1) )
    # (nx,nobs,1)
    invpNK_obs = tf.tile( tf.expand_dims(invpNK_obs, axis=0),
                        multiples=(nx,1,1))
    # (nx,ntest+nobs,nobs)
    query_mean_obs = k_x_xto @ invpNK_obs @ Y 
    # (nx,batchsize,1)

    """
    from test samples:
        invpNK -> nx,nt+no,nt+no
        k_x_xto ->nmax,nx,bs,nt+no
        invpNK_test -> nmax,nx,nt+no,nt
            _obs  -> nmax,nx,nt+no,no
        post_test_samples -> nmax,nx,ntest,nsample
        
        k_x_xto @ invpNK_test @ post_test_samples
        nmax,nx,bs,nsample
    """
    invpNK_test = tf.tile( tf.expand_dims(invpNK_test, axis=0),
                        multiples=(nx,1,1))
    # (nx,ntest+nobs,ntest)
    query_mean_test = tf.tile(
                        tf.expand_dims(k_x_xto, axis=0),
                        multiples=(nmax,1,1,1)) \
                    @ tf.tile(
                        tf.expand_dims(invpNK_test, axis=0),
                        multiples=(nmax,1,1,1)) \
                    @ tf.tile(
                        tf.expand_dims(post_test_samples, axis=1),
                        multiples=(1,nx,1,1))
    # (nmax,nx,batchsize,nsample)

    query_mean = tf.expand_dims(query_mean_obs, axis=0) \
                 + query_mean_test
    # (nmax,nx,batchsize,nsample)

    """
    query_cov = k_x - k_x_xto @ invpNK @ k_x_xto.T
    k_x: nx,bs,bs
    k_x_xto: nx,bs,nt+no
    invpNK: nt+no,nt+no -> (nx,nt+no,nt+no) (invpNK_test and invpNK_obs should be converted before)
    """
    invpNK = tf.concat([invpNK_test, invpNK_obs], axis=2)
    # (nx,ntest+nobs,ntest+nobs)
    query_cov = k_x - k_x_xto @ invpNK @ tf.transpose(k_x_xto, perm=(0,2,1))
    # (nx,batchsize,batchsize)

    return query_mean, query_cov
    # (nmax, nx, batchsize, nsample)
    # (nx,batchsize,batchsize)


def mp(x, # nx, batchsize, xdim
        ls, sigmas, sigma0s,
        X, Y, # (nobs,xdim), (nobs,1)

        xdim, nobs, nhyp, 
        nysample,

        test_xs, # ntest, xdim (same for all hyp)
        max_probs_all, # nhyp, nmax

        # samples of f-values 
        # given different maximum candidates
        post_test_samples_all, # nhyp, nmax, ntest, nsample
        post_test_mask_all, # nhyp, nmax, nsample, dtype: tf.bool
        # as the numbers of samples for different nmax are different
        # mask is to indicate which samples are used

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
    batchsize = tf.shape(x)[1]
    ntest = tf.shape(post_test_samples_all)[2]
    avg_mp = tf.zeros(shape=(nx,), dtype=dtype)

    for i in range(nhyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        # sigma, sigma0: scalar
        # l: (1,xdim)

        max_probs = max_probs_all[i,...] # nmax,

        # samples of f-value given different maximum candidates
        post_test_samples = post_test_samples_all[i,...] # nmax, ntest, nsample
        post_test_masks = post_test_mask_all[i,...] # nmax, nsample
        
        non_zero_prob_idxs = tf.squeeze(tf.where(max_probs > 0.))
        nmax = tf.shape(non_zero_prob_idxs)[0]

        post_test_samples = tf.gather(post_test_samples, non_zero_prob_idxs, axis=0)

        # post_test_masks == 0.0 if the sample is invalid,
        #                    1.0 if the sample if valid
        post_test_masks = tf.gather(post_test_masks, non_zero_prob_idxs, axis=0)
        # (nmax,nsample)

        max_probs = tf.gather(max_probs, non_zero_prob_idxs, axis=0)
        
        invpNK = invpNK_all[i,...] # ntest+nobs,ntest+nobs

        invpNK_test = tf.gather(invpNK, indices=tf.range(ntest, dtype=tf.int32), axis=1)
        invpNK_obs = tf.gather(invpNK, indices=tf.range(ntest, ntest + nobs, dtype=tf.int32), axis=1)

        query_meanf_given_test_samples, query_covf_given_test_samples = \
            get_batch_queried_f_stat_given_test_samples(
                            x,
                            l, sigma, sigma0,

                            xdim, batchsize, ntest, nobs, 

                            X, # (nobs, xdim)
                            Y, # (nobs,1)
                            test_xs, # (ntest, xdim)

                            invpNK_test, # (ntest + nobs, ntest)
                            invpNK_obs, # (ntest+nobs, nobs)
                            # samples of f-value given different maximum candidates
                            post_test_samples, # nmax, ntest, nsample
                            dtype=dtype)
        # (nmax, nx, batchsize, nsample)
        # (nx, batchsize, batchsize)
        
        noise_mat = tf.eye(batchsize, dtype=dtype) * sigma0
        noise_mat = tf.expand_dims(noise_mat, 0)
        # (1,batchsize,batchsize)

        query_covy_given_test_samples = query_covf_given_test_samples + noise_mat
        # (nx, batchsize, batchsize)
    
        query_meanf_given_test_samples = tf.transpose(
            query_meanf_given_test_samples,
            perm = (0, 1, 3, 2)
        )
        # (nmax, nx, nsample, batchsize)

        query_covy_given_test_samples = tf.expand_dims(
            tf.expand_dims(query_covy_given_test_samples, 1),
            axis = 0
        )
        # (1, nx, 1, batchsize, batchsize)

        body = lambda j, sum_mp: [j+1, \
            sum_mp + mp_each_batch_y_samplemp_each_batch_y_sample(
                x,
                nx, batchsize, nmax, nysample,

                max_probs,

                query_meanf_given_test_samples,# (nmax, nx, nsample, batchsize)
                query_covy_given_test_samples, # (1, nx, 1, batchsize, batchsize)
                post_test_masks, # (nmax, nsample)
                dtype=dtype
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



def mp_each_batch_y_samplemp_each_batch_y_sample(
                x,
                nx, batchsize, nmax, nysample,

                max_probs, # (nmax)

                query_meany_given_test_samples,# (nmax, nx, nsample, batchsize)
                query_covy_given_test_samples, # (1, nx, 1, batchsize, batchsize)
                post_test_masks, # (nmax, nsample)
                dtype=tf.float32
            ):
    nsample = tf.shape(post_test_masks)[-1]
    normal_dists \
        = tfp.distributions.MultivariateNormalFullCovariance(
            loc = query_meany_given_test_samples,
            covariance_matrix = query_covy_given_test_samples
        )
    # (nmax, nx, nsample, batchsize)

    # sampling y given posterior | max_idx, data
    # shape (nysample, nmax, nx)
    ysample = normal_dists.sample(nysample)
    # (nysample, nmax, nx, nsample, batchsize)

    # (1) H[y|max_idx]
    log_prob = normal_dists.log_prob(ysample)
    # (nysample, nmax, nx, nsample)

    ext_post_test_masks = tf.reshape(post_test_masks, shape=(1,nmax,1,nsample))
    ext_post_test_masks = tf.tile(ext_post_test_masks, multiples=(nysample,1,1,1))
    ext_post_test_masks = tf.tile(ext_post_test_masks, multiples=(1,1,nx,1))
    # (nysample, nmax, nx, nsample)
    
    log_prob = tf.where(ext_post_test_masks, 
                        log_prob, 
                        tf.ones_like(log_prob, dtype=dtype) 
                            * tf.constant(-np.infty, dtype=dtype))
    log_mixture_prob = tf.reduce_logsumexp(log_prob, axis=3)
    # (nysample, nmax, nx)

    weighted_log_mixure_prob = log_mixture_prob * tf.reshape(max_probs, shape=(1,nmax,1))
    # (nysample, nmax, nx)

    # print("evaluate_mp: the line below is incorrectly implemented for other stochastic criteria!")
    cond_ent_y = -tf.reduce_mean( tf.reduce_sum(weighted_log_mixure_prob, axis=1), axis=0 )
    # (nx,)


    # (2) H[y]
    print("sample from different max_idx should have different weight!, \
        this is incorrectly implemented for evaluate_mp.py, \
        this could be incorrect for evaluate_emes.py too! CHECK")
    # ysample.shape = (nysample, nmax, nx, nsample, batchsize)
    marginal_ysample = tf.tile(
            tf.expand_dims(ysample, axis=2),
            multiples=(1,1,nmax,1,1, 1))
    # (nysample, nmax, nmax, nx, nsample, batchsize)
    # ____marginal___

    # Marginalizing over nsample
    log_prob = normal_dists.log_prob(marginal_ysample)
    # (nysample, nmax, nmax, nx, nsample)

    ext_post_test_masks = tf.expand_dims(ext_post_test_masks, axis=2)
    ext_post_test_masks = tf.tile(ext_post_test_masks, multiples=(1,1,nmax,1,1))
    # (nysample, nmax, nmax, nx, nsample)

    log_prob = tf.where(ext_post_test_masks, 
                        log_prob, 
                        tf.ones_like(log_prob, dtype=dtype) 
                            * tf.constant(-np.infty, dtype=dtype))
    log_marginal_mixture_prob = tf.reduce_logsumexp(log_prob, axis=4)
    # (nysample, nmax, nmax, nx)

    # Marginalizing over nmax as p(y) mixture of nmax Gaussians
    weighted_log_marginal_mixture_prob = log_marginal_mixture_prob + tf.log( tf.reshape(max_probs, shape=(1, 1, nmax, 1)) )
    # (nysample, nmax, nmax, nx)
    log_marginal_prob = tf.reduce_logsumexp(weighted_log_marginal_mixture_prob, axis=2)
    # (nysample, nmax, nx)

    # Weighted average
    weighted_log_marginal_prob = log_marginal_prob * tf.reshape(max_probs, shape=(1,nmax,1))
    # (nysample, nmax, nx)

    ent_y = - tf.reduce_mean( tf.reduce_sum(weighted_log_marginal_prob, axis=1), axis=0)
    # (nx,)

    mp_val = tf.reshape(ent_y - cond_ent_y, shape=(nx,))
    return mp_val
