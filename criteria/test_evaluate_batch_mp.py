import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import scipy as sp 
import scipy.stats as spst

import matplotlib

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt 
plt.style.use('seaborn')


import seaborn as sns
colors = sns.color_palette()

import sys

sys.path.insert(0, './..')


import utils
import evaluate_batch_mp
import evaluate_batch_sample_mp
import ep
import empirical_approximation



def get_maxidxs_stats(nhyp, nmax, ntest, 
                max_idxs, # (nhyp, nmax)
                test_means, # (nhyp, ntest)
                test_covs, # (nhyp, ntest, ntest)
                use_EP=False): 
    """
    stats = {probabilities, means, covs}
    """
    max_probs_np = np.zeros([nhyp,nmax])
    for i in range(nhyp):
        max_probs_np[i,:] = utils.compute_post_maxidxs_np(
            test_means[i,...],
            test_covs[i,...],
            max_idxs[i,...])

    # EP approximates distribution of f-value at x_test
    post_mean_tests = np.zeros([nhyp, nmax, ntest])
    post_cov_tests = np.zeros([nhyp, nmax, ntest, ntest])

    for i in range(nhyp):
        for j in range(nmax):
            if use_EP:
                post_mean_test, post_cov_test =  ep.approximate_EP_np( 
                        max_idxs[i,j],
                        test_means[i,...].reshape(-1,1),
                        test_covs[i,...],
                        resolution=1e-9,
                        max_niter=200)
            else:
                post_mean_test, post_cov_test = empirical_approximation.get_empirical_stat(test_means[i,...], test_covs[i,...], max_idxs[i,j], nsample=100000, ntrial=100000, importance_sampling=False)

            post_mean_tests[i,j,:] = post_mean_test.squeeze()
            post_cov_tests[i,j,...] = post_cov_test

    # some post_mean, post_cov might be nan from EP
    # set their probability to 0.0
    is_nan_exist = False
    for i in range(nhyp):
        for j in range(nmax):
            if np.any(np.isnan(post_mean_tests[i,j,:])) or np.any(np.isnan(post_cov_tests[i,j,...])):
                print("EP returns nan: suppress a maxidx {} with probability {:.3f} to 0.0".format(max_idxs[i,j], max_probs_np[i,j]))

                is_nan_exist = True 

                max_probs_np[i,j] = 0.0
                
                post_mean_tests[i,j,:] = 0.0
                post_cov_tests[i,j,...] = np.eye(ntest)

        # re-normalize max probs
        max_probs_np[i,:] = max_probs_np[i,:] / np.sum(max_probs_np[i,:])

    return max_probs_np, post_mean_tests, post_cov_tests



def test_evaluate_mp_multiple_x(seed=None):

    np.random.seed(1)
    tf.reset_default_graph()

    if seed is not None:
        tf.set_random.seed(seed)

    dtype = tf.float64
    xdim = 1
    nhyp = 1

    IS_LOAD = True
    SMALL_LS = True
    LEGEND = False

    if SMALL_LS:
        folder = "small_ls"
        ls_val = np.array([[5e-1]]) # lengthscale
        sigmas_val = np.array([[1.0]]) # signal variance
        sigma0s_val = np.array([[2e-1]]) # noise variance
    else:
        folder = "large_ls"
        ls_val = np.array([[10.]]) # lengthscale
        sigmas_val = np.array([[1.0]]) # signal variance
        sigma0s_val = np.array([[2e-1]]) # noise variance

    """
    small_ls: 8e-1, 1, 2e-1
    large_ls: 5., 1., 2e-1
    """

    batchsize = 2
    n = 26
    x1d_val = np.linspace(0., 4., n).reshape(-1,1)
    xplot_val = x1d_val.copy()
    xx_val, xy_val = np.meshgrid(x1d_val, x1d_val)
    x_val = np.concatenate([xx_val.reshape(-1,1), xy_val.reshape(-1,1)], axis=1).reshape(n*n,2,1)

    true_func = lambda x: np.sin(x)

    Xs_val = np.array([[0.], [0.5], [1.12], [3.]])
    Ys_val = true_func(Xs_val) + np.random.randn(Xs_val.shape[0], Xs_val.shape[1]) * np.sqrt(sigma0s_val[0,0])

    # test_xs_val = np.array([0., 1., 2., 5.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
    test_xs_val = np.array([1.12, 1.44, 1.92]).reshape(1,-1,1)
    ntest = test_xs_val.shape[1]
    # max_idxs = np.array([[0, 1, 2, 3]]) # (nhyp, nmax)
    max_idxs = np.array([[0,1,2]])
    nmax = max_idxs.shape[1]

    xplot = tf.placeholder(dtype=dtype, shape=(None, xdim), name='xplot')

    x = tf.placeholder(dtype=dtype, shape=(None, batchsize, xdim), name='x')

    ls = tf.placeholder(dtype=dtype, shape=(nhyp,xdim), name='ls')
    sigmas = tf.placeholder(dtype=dtype, shape=(nhyp,1), name='sigmas') 
    sigma0s = tf.placeholder(dtype=dtype, shape=(nhyp,1), name='sigma0s')

    Xsamples = tf.placeholder(dtype=dtype, shape=(None, xdim), name='Xsamples')
    Ysamples = tf.placeholder(dtype=dtype, shape=(None, 1), name='Ysamples')
    test_xs = tf.placeholder(dtype=dtype, shape=(nhyp, None,xdim), name='test_xs')

    # xdim
    nx = x_val.shape[0]
    nobs = Xs_val.shape[0]



    # stat of f values at maxlocs
    test_mean_all = [] # (nhyp, ntest)
    test_cov_all = [] # (nhyp, ntest, ntest)

    for i in range(nhyp):
        test_mean_i, test_cov_i = utils.compute_mean_var_f(test_xs[i,...], 
                                    Xsamples, Ysamples, 
                                    ls[i,...], sigmas[i,...], sigma0s[i,...],
                                    fullcov=True, 
                                    dtype=dtype)

        test_mean_all.append(tf.squeeze(test_mean_i))
        test_cov_all.append(test_cov_i)

    test_mean_all = tf.stack(test_mean_all)
    # (nhyp, ntest)
    test_cov_all = tf.stack(test_cov_all)
    # shape = (nhyp, ntest, ntest)


    # mean,var of f values at x
    fs_mean, fs_var = utils.compute_mean_var_f(xplot, 
                                Xsamples, Ysamples, 
                                ls[0,...], sigmas[0,...], sigma0s[0,...],
                                fullcov=False, 
                                dtype=dtype)


    # invpNK_all
    _, invpNK_all = evaluate_batch_mp.get_pNK_test_obs(ls, sigmas, sigma0s, 
                nhyp,
                Xsamples,
                test_xs[0,...],
                dtype=dtype)
    # (nhyp, ntest+nobs, ntest+nobs)


    # invKobs_all
    invNKobs_all = []
    for i in range(nhyp):
        NKobs = utils.computeNKmm(Xsamples, ls[i,...], sigmas[i,...], sigma0s[i,...], dtype=dtype)
        invNKobs = utils.chol2inv(NKobs, dtype=dtype)
        invNKobs_all.append(invNKobs)
    invNKobs_all = tf.stack(invNKobs_all)

    # max_probs_all
    max_probs = tf.placeholder(dtype=dtype, shape=(nhyp, nmax), name='max_probs')

    # statistics of f-values at test inputs
    # given different maximum candidates
    post_mean_test_fs_all = tf.placeholder(dtype=dtype, shape=(nhyp, nmax, ntest), name='post_mean_test_fs_all')
    post_cov_test_fs_all = tf.placeholder(dtype=dtype, shape=(nhyp, nmax, ntest, ntest), name='post_cov_test_fs_all')
    
    
    nysamples = 10
    niteration = 100

    mp_sto = evaluate_batch_mp.mp(x, 
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,
            
            xdim, nobs, nhyp,
            nysamples,
            
            test_xs[0,...],
            max_probs,
            
            post_mean_test_fs_all,
            post_cov_test_fs_all,

            invNKobs_all,
            invpNK_all,

            dtype=dtype,

            niteration=niteration,
            use_loop=True,
            parallel_iterations=1)


    post_test_samples_all_plc = tf.placeholder(dtype=dtype, 
                            shape=(nhyp, None, None, None), 
                            name='post_test_samples_all_plc')
    post_test_mask_all_plc = tf.placeholder(dtype=tf.bool, 
                            shape=(nhyp, None, None),
                            name='post_test_mask_all_plc')

    sample_mp = evaluate_batch_sample_mp.mp(x, # nx, xdim
        ls, sigmas, sigma0s,
        Xsamples, Ysamples, # (nobs,xdim), (nobs,1)

        xdim, nobs, nhyp, 
        nysamples, # nysample

        test_xs[0,...], # ntest, xdim (same for all hyp)
        max_probs, # nhyp, nmax

        # samples of f-values 
        # given different maximum candidates
        post_test_samples_all_plc, # nhyp, nmax, ntest, nsample
        post_test_mask_all_plc, # nhyp, nmax, nsample, dtype: tf.bool
        # as the numbers of samples for different nmax are different
        # mask is to indicate which samples are used

        # K_test_max needs to be precomputed
        # and its inverse
        # need naming convension for noisy
        # vs. noiseless K
        # and partial noisy-noiseless?
        invpNK_all, # nhyp, ntest+nobs, ntest+nobs

        dtype=dtype,

        niteration=niteration,
        use_loop=True,
        parallel_iterations=1)



    if not IS_LOAD:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            fs_mean_val, fs_var_val = sess.run(
                [fs_mean, fs_var],
                feed_dict = {
                        xplot: xplot_val,
                        ls: ls_val,
                        sigmas: sigmas_val,
                        sigma0s: sigma0s_val,
                        Xsamples: Xs_val,
                        Ysamples: Ys_val
                }
            )

            np.savez('{}/test_evaluate_batch_mp_fs_mean.npz'.format(folder), fs_mean_val)
            np.savez('{}/test_evaluate_batch_mp_fs_var.npz'.format(folder), fs_var_val)

            test_mean_all_val, test_cov_all_val = sess.run(
                    [test_mean_all, test_cov_all],
                    feed_dict = {
                        test_xs: test_xs_val,
                        ls: ls_val,
                        sigmas: sigmas_val,
                        sigma0s: sigma0s_val,
                        Xsamples: Xs_val,
                        Ysamples: Ys_val
                    })

            max_probs_np, post_mean_test_fs_all_np, post_cov_test_fs_all_np = get_maxidxs_stats(nhyp, nmax, ntest, 
                    max_idxs, # (nhyp, nmax)
                    test_mean_all_val, # (nhyp, ntest)
                    test_cov_all_val)

            invpNK_all_np, invNKobs_all_np  = sess.run([invpNK_all, invNKobs_all], 
                    feed_dict = {
                        test_xs: test_xs_val,
                        ls: ls_val,
                        sigmas: sigmas_val,
                        sigma0s: sigma0s_val,
                        Xsamples: Xs_val,
                        Ysamples: Ys_val
                    })

            mps_np = sess.run(mp_sto,
                feed_dict={
                        x: x_val,
                        test_xs: test_xs_val,
                        ls: ls_val,
                        sigmas: sigmas_val,
                        sigma0s: sigma0s_val,
                        Xsamples: Xs_val,
                        Ysamples: Ys_val,
                        max_probs: max_probs_np,
                        post_mean_test_fs_all: post_mean_test_fs_all_np,
                        post_cov_test_fs_all: post_cov_test_fs_all_np,
                        invNKobs_all: invNKobs_all_np,
                        invpNK_all: invpNK_all_np
                })

            np.savez('{}/test_evaluate_batch_mp_mps_mp.npz'.format(folder), mps_np)


            # for sample mp for each hyp!
            sample_mp_nmax, unique_maxidxs, sample_mp_nsample, sample_mp_samples, sample_mp_masks, max_samples_size \
                = utils.sample_multivariate_normal_maxidx_np(test_mean_all_val[0,...], test_cov_all_val[0,...], 
                nsample=1000, n_min_sample=1)
            
            # construct max_probs, post_test_samples_all, post_test_mask_all
            max_probs_np = np.tile( max_samples_size.reshape(1,sample_mp_nmax), reps=(nhyp,1) ).astype(float)
            max_probs_np /= np.sum(max_probs_np, axis=1, keepdims=True)

            post_test_samples_np = sample_mp_samples.reshape(1, sample_mp_nmax, ntest, sample_mp_nsample)
            post_test_masks_np = sample_mp_masks.reshape(1, sample_mp_nmax, sample_mp_nsample)
            
            sample_mp_val = sess.run(sample_mp,
                feed_dict = {
                        x: x_val,
                        test_xs: test_xs_val,
                        ls: ls_val,
                        sigmas: sigmas_val,
                        sigma0s: sigma0s_val,
                        Xsamples: Xs_val,
                        Ysamples: Ys_val,
                        max_probs: max_probs_np,
                        post_test_samples_all_plc: post_test_samples_np,
                        post_test_mask_all_plc: post_test_masks_np,
                        invpNK_all: invpNK_all_np
                })

            np.savez('{}/test_evaluate_batch_mp_sample_mp.npz'.format(folder), sample_mp_val)
    else:
        fs_mean_val = np.load('{}/test_evaluate_batch_mp_fs_mean.npz'.format(folder))['arr_0']
        fs_var_val = np.load('{}/test_evaluate_batch_mp_fs_var.npz'.format(folder))['arr_0']

        mps_np = np.load('{}/test_evaluate_batch_mp_mps_mp.npz'.format(folder))['arr_0']
        sample_mp_val = np.load('{}/test_evaluate_batch_mp_sample_mp.npz'.format(folder))['arr_0']
        

    idx1dto2d = lambda i: ( int(i/n), i % n )
    mps_max_idx = np.argmax(mps_np)
    mps_max_idx_x, mps_max_idx_y = idx1dto2d(mps_max_idx)

    smp_max_idx = np.argmax(sample_mp_val)
    smp_max_idx_x, smp_max_idx_y = idx1dto2d(smp_max_idx)

    print("mps maximized at {},{}".format(x1d_val[mps_max_idx_x], x1d_val[mps_max_idx_y]))
    print("sample_mp maximized at {},{}".format(x1d_val[smp_max_idx_x], x1d_val[smp_max_idx_y]))
    
    # print("mps idx: ", mps_max_idx, idx2d_to_idx1d(mps_max_idx))
    # print("sample_mp idx: ", smp_max_idx, idx2d_to_idx1d(smp_max_idx))

    figsize = (2.1,2.1)
    fig, ax = plt.subplots(figsize=figsize)

    ax.contour(x_val[:,0].reshape(n,n), x_val[:,1].reshape(n,n), sample_mp_val.reshape(n,n), levels=10)
    ax.set_ylabel(r'$\rm x_{0}$', fontsize=13)
    ax.set_xlabel(r'$\rm x_{1}$', fontsize=13)

    sample_cidx = 2
    max_idx = np.argmax(sample_mp_val)
    max_idx0 = max_idx % n
    max_idx1 = int(np.floor(max_idx / n))
    sample_maxx0 = xplot_val[max_idx0,0]
    sample_maxx1 = xplot_val[max_idx1,0]
    ax.scatter([sample_maxx0], [sample_maxx1], color=colors[sample_cidx], marker=7, zorder=100)
    # ax.scatter([sample_maxx0, sample_maxx1], [sample_maxx1, sample_maxx0], color=colors[sample_cidx], marker=7, zorder=100)

    ax.set_xticks([0,1,2,3,4])
    plt.tight_layout()


    fig, ax = plt.subplots(figsize=figsize)

    ax.contour(x_val[:,0].reshape(n,n), x_val[:,1].reshape(n,n), mps_np.reshape(n,n), levels=10)
    # axs[2].tricontourf(x_val[:,0].squeeze(), x_val[:,1].squeeze(), mps_np, cmap="ocean")
    ax.set_ylabel(r'$\rm x_{0}$', fontsize=13)
    ax.set_xlabel(r'$\rm x_{1}$', fontsize=13)

    ep_cidx = 4
    max_idx = np.argmax(mps_np)
    max_idx0 = max_idx % n
    max_idx1 = int(np.floor(max_idx / n))
    ep_maxx0 = xplot_val[max_idx0,0]
    ep_maxx1 = xplot_val[max_idx1,0]
    ax.scatter([ep_maxx0], [ep_maxx1], color=colors[ep_cidx], marker=6, zorder=100)
    # ax.scatter([ep_maxx0, ep_maxx1], [ep_maxx1, ep_maxx0], color=colors[ep_cidx], marker=6, zorder=100)
    ax.set_xticks([0,1,2,3,4])

    plt.tight_layout()


    fig, ax = plt.subplots(figsize=[figsize[0]*1.2, figsize[0]])

    ax.scatter(Xs_val, Ys_val, zorder=100)
    # fs_mean_val, fs_var_val
    ax.plot(xplot_val.squeeze(), fs_mean_val)
    line = ax.plot(xplot_val.squeeze(), fs_mean_val + 2 * np.sqrt(fs_var_val), '--')
    ax.plot(xplot_val.squeeze(), fs_mean_val - 2 * np.sqrt(fs_var_val), '--', color=line[0].get_color())
    ax.set_xlabel(r'$\rm x$', fontsize=13)
    ax.set_ylabel(r'$\rm f(x)$', fontsize=13)

    ylim = ax.get_ylim()
    line = ax.plot([test_xs_val[0,0], test_xs_val[0,0]], ylim, ':', alpha=0.7, zorder=1)
    test_x_color = line[0].get_color()

    for test_x in test_xs_val.squeeze()[1:]:
        line = ax.plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)

    sample_max_xs = np.array([sample_maxx0, sample_maxx1]).squeeze()
    ax.scatter(sample_max_xs, true_func(sample_max_xs), marker=7, color=colors[sample_cidx], label=r'M-SES$\rm _{sp}$', zorder=200)
    # axs[0].plot([sample_maxx0, sample_maxx0], ylim, '--', color=colors[sample_cidx])
    # axs[0].plot([sample_maxx1, sample_maxx1], ylim, '--', color=colors[sample_cidx])

    ep_max_xs = np.array([ep_maxx0, ep_maxx1]).squeeze()
    ax.scatter(ep_max_xs, true_func(ep_max_xs), marker=6, color=colors[ep_cidx], label=r'M-SES$\rm _{ep}$', zorder=200)
    # axs[0].plot([ep_maxx0, ep_maxx0], ylim, '-.', color=colors[ep_cidx])
    # axs[0].plot([ep_maxx1, ep_maxx1], ylim, '-.', color=colors[ep_cidx])
    if LEGEND:
        ax.legend()
    ax.set_xticks([0,1,2,3,4])

    plt.tight_layout()
    plt.show()




    # fig, axs = plt.subplots(3, figsize=(3,9))

    # axs[1].contour(x_val[:,0].reshape(n,n), x_val[:,1].reshape(n,n), sample_mp_val.reshape(n,n), levels=10)
    # axs[1].set_ylabel(r'$x_{t,0}$')
    # axs[1].set_xlabel(r'$x_{t,1}$')

    # sample_cidx = 2
    # max_idx = np.argmax(sample_mp_val)
    # max_idx0 = max_idx % n
    # max_idx1 = int(np.floor(max_idx / n))
    # sample_maxx0 = xplot_val[max_idx0,0]
    # sample_maxx1 = xplot_val[max_idx1,0]
    # axs[1].scatter([sample_maxx0, sample_maxx1], [sample_maxx1, sample_maxx0], color=colors[sample_cidx], marker=7, zorder=100)

    # axs[2].contour(x_val[:,0].reshape(n,n), x_val[:,1].reshape(n,n), mps_np.reshape(n,n), levels=10)
    # # axs[2].tricontourf(x_val[:,0].squeeze(), x_val[:,1].squeeze(), mps_np, cmap="ocean")
    # axs[2].set_ylabel(r'$x_{t,0}$')
    # axs[2].set_xlabel(r'$x_{t,1}$')

    # ep_cidx = 3
    # max_idx = np.argmax(mps_np)
    # max_idx0 = max_idx % n
    # max_idx1 = int(np.floor(max_idx / n))
    # ep_maxx0 = xplot_val[max_idx0,0]
    # ep_maxx1 = xplot_val[max_idx1,0]
    # axs[2].scatter([ep_maxx0, ep_maxx1], [ep_maxx1, ep_maxx0], color=colors[ep_cidx], marker=6, zorder=100)


    # axs[0].scatter(Xs_val, Ys_val, zorder=100)
    # # fs_mean_val, fs_var_val
    # axs[0].plot(xplot_val.squeeze(), fs_mean_val)
    # line = axs[0].plot(xplot_val.squeeze(), fs_mean_val + 2 * np.sqrt(fs_var_val), '--')
    # axs[0].plot(xplot_val.squeeze(), fs_mean_val - 2 * np.sqrt(fs_var_val), '--', color=line[0].get_color())
    # axs[0].set_xlabel(r'x')
    # axs[0].set_ylabel(r'f(x)')

    # ylim = axs[0].get_ylim()
    # line = axs[0].plot([test_xs_val[0,0], test_xs_val[0,0]], ylim, ':', alpha=0.7, zorder=1)
    # test_x_color = line[0].get_color()

    # for test_x in test_xs_val.squeeze()[1:]:
    #     line = axs[0].plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)

    # sample_max_xs = np.array([sample_maxx0, sample_maxx1]).squeeze()
    # axs[0].scatter(sample_max_xs, true_func(sample_max_xs), marker=7, color=colors[sample_cidx])
    # # axs[0].plot([sample_maxx0, sample_maxx0], ylim, '--', color=colors[sample_cidx])
    # # axs[0].plot([sample_maxx1, sample_maxx1], ylim, '--', color=colors[sample_cidx])

    # ep_max_xs = np.array([ep_maxx0, ep_maxx1]).squeeze()
    # axs[0].scatter(ep_max_xs, true_func(ep_max_xs), marker=6, color=colors[ep_cidx])
    # # axs[0].plot([ep_maxx0, ep_maxx0], ylim, '-.', color=colors[ep_cidx])
    # # axs[0].plot([ep_maxx1, ep_maxx1], ylim, '-.', color=colors[ep_cidx])

    # plt.tight_layout()
    # plt.show()

    return mps_np, sample_mp_val, x1d_val
