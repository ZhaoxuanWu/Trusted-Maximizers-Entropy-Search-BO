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
import evaluate_mp
import evaluate_mp_lite
import evaluate_sample_mp
import ep
import empirical_approximation
import evaluate_pes



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

    # l: 5., sigma: 3., sigma0: 5.
    # l: 2., sigma: 3., sigma0: 5.
    # small length-scale: 1.
    # large length-scale: 5.
    ls_val = np.array([[25.]]) # lengthscale
    sigmas_val = np.array([[3.0]]) # signal variance
    sigma0s_val = np.array([[5.0]]) # noise variance

    """
    uncorr: 
        ls: 25, sigma: 3, sigma0: 5.
        Xs_val = np.array([[0.], [1.], [2.], [3.]])
        Ys_val = np.array([2.5, 2.7, 2.8, 0.3]).reshape(4,1)
        test_xs_val = np.array([1., 2.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
    corr:
        ls: 1., sigma 3, sigma0: 5.
        Xs_val = np.array([[0.], [1.], [2.], [3.]])
        Ys_val = np.array([2.5, 2.7, 2.8, 0.3]).reshape(4,1)
        test_xs_val = np.array([1., 2.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
    exploit example
        ls: 4., sigma 3, sigma0: 5
        Xs_val = np.array([[0.], [1.], [2.], [3.]])
        Ys_val = np.array([3.5, 2.0, 1.2, 0.3]).reshape(4,1)
        test_xs_val = np.array([0., 1., 2.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
    """
 
    x_val = np.linspace(0., 4., 100).reshape(-1,1)

    Xs_val = np.array([[0.], [1.], [2.], [3.]])
    Ys_val = np.array([2.5, 2.7, 2.8, 0.3]).reshape(4,1)
    test_xs_val = np.array([1., 2.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
        
    # Xs_val = np.array([[0.], [1.], [2.], [3.]])
    # Ys_val = np.array([3.5, 2.0, 1.2, 0.3]).reshape(4,1)
    # # Ys_val = np.sin(Xs_val) + np.random.randn(Xs_val.shape[0], Xs_val.shape[1]) * np.sqrt(sigma0s_val[0,0])
    # # print(Ys_val)

    # test_xs_val = np.array([0., 1., 2.]).reshape(1,-1,1) # (nhyp, ntest, xdim)
    # # test_xs_val = np.array([0.9, 1.1]).reshape(1,-1,1)
    ntest = test_xs_val.shape[1]
    max_idxs = np.array([[0, 1]]) # (nhyp, nmax)
    # max_idxs = np.array([[0, 1]]) # (nhyp, nmax)
    nmax = max_idxs.shape[1]

    x = tf.placeholder(dtype=dtype, shape=(None, xdim), name='x')

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
    fs_mean_all = [] # (nhyp, ntest)
    fs_var_all = [] # (nhyp, ntest, ntest)

    for i in range(nhyp):
        fs_mean_i, fs_var_i = utils.compute_mean_var_f(test_xs[i,...], 
                                    Xsamples, Ysamples, 
                                    ls[i,...], sigmas[i,...], sigma0s[i,...],
                                    fullcov=False, 
                                    dtype=dtype)

        fs_mean_all.append(tf.squeeze(fs_mean_i))
        fs_var_all.append(fs_var_i)
    fs_mean_all = tf.stack(fs_mean_all)
    # (nhyp, ntest)
    fs_var_all = tf.stack(fs_var_all)
    # shape = (nhyp, ntest)    


    # invpNK_all
    _, invpNK_all = evaluate_mp.get_pNK_test_obs(ls, sigmas, sigma0s, 
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
    
    
    nysamples = 1000

    mp_det = evaluate_mp.mp(x, 
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,
            
            xdim, nx, nobs, nhyp,
            nysamples,
            
            test_xs[0,...],
            max_probs,
            
            post_mean_test_fs_all,
            post_cov_test_fs_all,

            invNKobs_all,
            invpNK_all,

            stochastic=False,
            dtype=dtype,

            niteration=50,
            use_loop=True,
            parallel_iterations=1)

    mp_sto = evaluate_mp.mp(x, 
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,
            
            xdim, nx, nobs, nhyp,
            nysamples,
            
            test_xs[0,...],
            max_probs,
            
            post_mean_test_fs_all,
            post_cov_test_fs_all,

            invNKobs_all,
            invpNK_all,

            stochastic=True,
            dtype=dtype,

            niteration=500,
            use_loop=True,
            parallel_iterations=1)

    mp_lite = evaluate_mp_lite.mp(x, 
            ls, sigmas, sigma0s,
            Xsamples, Ysamples,
            
            xdim, nx, nobs, nhyp,
            
            test_xs[0,...],
            max_probs,
            
            post_mean_test_fs_all,
            post_cov_test_fs_all,

            invpNK_all,

            dtype=dtype)


    post_test_samples_all_plc = tf.placeholder(dtype=dtype, 
                            shape=(nhyp, None, None, None), 
                            name='post_test_samples_all_plc')
    post_test_mask_all_plc = tf.placeholder(dtype=tf.bool, 
                            shape=(nhyp, None, None),
                            name='post_test_mask_all_plc')

    sample_mp = evaluate_sample_mp.mp(x, # nx, xdim
        ls, sigmas, sigma0s,
        Xsamples, Ysamples, # (nobs,xdim), (nobs,1)

        xdim, nx, nobs, nhyp, 
        100, # nysample

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

        niteration=10,
        use_loop=True,
        parallel_iterations=1)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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


        fs_mean_all_val, fs_var_all_val = sess.run(
                [fs_mean_all, fs_var_all],
                feed_dict = {
                    test_xs: x_val.reshape(1,-1,1),
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

        mpd_np, mps_np, mp_lite_np = sess.run([mp_det, mp_sto, mp_lite],
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



    fig, axs = plt.subplots(4, sharex=True, figsize=(3,4))

    from matplotlib.ticker import FormatStrFormatter
    for ax in axs:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

    axs[0].scatter(Xs_val, Ys_val, zorder=100)
    # fs_mean_all_val, fs_var_all_val
    axs[0].plot(x_val, fs_mean_all_val[0,:])
    line = axs[0].plot(x_val, fs_mean_all_val[0,:] + 2.*np.sqrt(fs_var_all_val[0,:]), '--')
    axs[0].plot(x_val, fs_mean_all_val[0,:] - 2.*np.sqrt(fs_var_all_val[0,:]), '--', color=line[0].get_color())
    axs[0].set_ylabel(r'$\rm f(x)$', fontsize=13)
    
    ylim = axs[0].get_ylim()
    line = axs[0].plot([test_xs_val[0,0], test_xs_val[0,0]], ylim, ':', alpha=0.7, zorder=1)
    test_x_color = line[0].get_color()

    for test_x in test_xs_val.squeeze()[1:]:
        line = axs[0].plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)


    # axs[1].scatter(test_xs_val.squeeze(), np.max(sample_mp_val) * np.ones_like(test_xs_val.squeeze()))
    max_idx = np.argmax(sample_mp_val)
    axs[1].scatter([x_val[max_idx]], [sample_mp_val[max_idx]], color=colors[2], marker='X', zorder=100)
    axs[1].plot(x_val, sample_mp_val)#, '-.', label='sample MP')
    mprange = np.max(sample_mp_val) - np.min(sample_mp_val)
    axs[1].set_ylim(np.min(sample_mp_val) - mprange/5, np.max(sample_mp_val) + mprange/5)
    axs[1].set_ylabel(r'M-SES$\rm _{sp}$', fontsize=13)
    # axs[1].legend()

    ylim = axs[1].get_ylim()
    for test_x in test_xs_val.squeeze():
        line = axs[1].plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)


    # axs[2].scatter(test_xs_val.squeeze(), np.max(mps_np) * np.ones_like(test_xs_val.squeeze()))
    max_idx = np.argmax(mps_np)
    axs[2].scatter([x_val[max_idx]], [mps_np[max_idx]], color=colors[2], marker='X', zorder=100)
    axs[2].plot(x_val, mps_np)#, label='stochastic MP')
    mprange = np.max(mps_np) - np.min(mps_np)
    axs[2].set_ylim(np.min(mps_np) - mprange/5, np.max(mps_np) + mprange/5)
    axs[2].set_ylabel(r'M-SES$\rm _{ep}$', fontsize=13)
    # axs[2].legend()

    ylim = axs[2].get_ylim()
    for test_x in test_xs_val.squeeze():
        line = axs[2].plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)


    # axs[3].scatter(test_xs_val.squeeze(), np.max(mpd_np) * np.ones_like(test_xs_val.squeeze()))
    max_idx = np.argmax(mpd_np)
    axs[3].scatter([x_val[max_idx]], [mpd_np[max_idx]], color=colors[2], marker='X', zorder=100)
    axs[3].plot(x_val, mpd_np)#, '--', label='deterministic MP')
    mprange = np.max(mpd_np) - np.min(mpd_np)
    axs[3].set_ylim(np.min(mpd_np) - mprange/5, np.max(mpd_np) + mprange/5)
    axs[3].set_ylabel(r'M-SES$\rm _{mm}$', fontsize=13)
    # axs[3].legend()

    ylim = axs[3].get_ylim()
    for test_x in test_xs_val.squeeze():
        line = axs[3].plot([test_x, test_x], ylim, ':', color=test_x_color, alpha=0.7, zorder=1)

    axs[3].set_xlabel(r'$\rm x$', fontsize=13)
    axs[3].set_xticks([0, 1, 2, 3, 4])

    # axs[3].scatter(test_xs_val.squeeze(), np.max(mp_lite_np) * np.ones_like(test_xs_val.squeeze()))
    # axs[3].plot(x_val, mp_lite_np, '-.', label='MI+ MP')
    # mprange = np.max(mp_lite_np) - np.min(mp_lite_np)
    # axs[3].set_ylim(np.min(mp_lite_np) - mprange/5, np.max(mp_lite_np) + mprange/5)
    # axs[3].legend()

    plt.tight_layout()
    plt.show()

    return mps_np, mpd_np, mp_lite_np






def test_optimize_scpes():

    np.random.seed(1)
    tf.reset_default_graph()

    dtype = tf.float64
    xdim = 1
    nhyp = 1

    ls_val = np.array([[1.]]) # lengthscale
    sigmas_val = np.array([[5.0]]) # signal variance
    sigma0s_val = np.array([[1.0]]) # noise variance

    x_val = np.linspace(0., 6., 100).reshape(-1,1)
    
    Xs_val = np.array([[0.], [2.], [5.], [6.]])
    Ys_val = np.sin(Xs_val) + np.random.randn(Xs_val.shape[0], Xs_val.shape[1]) * np.sqrt(sigma0s_val[0,0])

    # maxlocs_val = np.array([[0.], [1.], [1.5], [2.], [2.5], [3.], [3.5]])
    # maxlocs_val = np.array([[0.2], [0.5], [0.8]])
    maxlocs_val = np.array([[0.], [2.], [5.]])

    nmaxloc = maxlocs_val.shape[0]



    x = tf.placeholder(dtype=dtype, shape=(None, xdim))
    x_to_opt = tf.get_variable(shape = (1,xdim), 
                            dtype = dtype, 
                            name  = "x_to_opt", 
                            constraint = lambda x: tf.clip_by_value(x, 
                                            np.min(x_val), np.max(x_val)),
                            initializer=tf.initializers.random_uniform(
                                            minval=np.min(x_val), 
                                            maxval=2.0,#np.max(x_val), 
                                            dtype=dtype))

    ls = tf.placeholder(dtype=dtype, shape=(nhyp,xdim))
    sigmas = tf.placeholder(dtype=dtype, shape=(nhyp,1)) 
    sigma0s = tf.placeholder(dtype=dtype, shape=(nhyp,1))

    Xsamples = tf.placeholder(dtype=dtype, shape=(None, xdim))
    Ysamples = tf.placeholder(dtype=dtype, shape=(None, 1))
    maxlocs = tf.placeholder(dtype=dtype, shape=(None, xdim))

    ntrain = 500

    # xdim
    nx = x_val.shape[0]
    nx_to_opt = 1
    nobs = Xs_val.shape[0]
    # nhyp
    nysample = 10
    niteration = 5

    nmaxfd = 2
    nmaxf = 2


    # stat of f values at maxlocs
    mean_maxf_all = [] # (nhyp, nmaxloc)
    std_maxf_all = [] # (nhyp, nmaxloc)

    for i in range(nhyp):
        mean_maxf_i, var_maxf_i = utils.compute_mean_var_f(maxlocs, 
                                    Xsamples, Ysamples, 
                                    ls[i,...], sigmas[i,...], sigma0s[i,...], 
                                    dtype=dtype)
        
        mean_maxf_all.append(tf.squeeze(mean_maxf_i))
        std_maxf_all.append(tf.sqrt(tf.squeeze(var_maxf_i)))

    mean_maxf_all = tf.stack(mean_maxf_all)
    std_maxf_all = tf.stack(std_maxf_all)
    # shape = (nhyp, nmaxloc)


    # stat of f values for observation
    post_data_f_means_all = []
    sqrt_post_data_f_covs_all = []

    for i in range(nhyp):
        post_data_f_mean, post_data_f_cov = utils.compute_mean_var_f(
                    Xsamples, 
                    Xsamples, Ysamples, 
                    ls[i,...], sigmas[i,...], sigma0s[i,...], 
                    fullcov=True, dtype=dtype)

        sqrt_post_data_f_cov = utils.sqrtm(post_data_f_cov)

        post_data_f_means_all.append(post_data_f_mean)
        sqrt_post_data_f_covs_all.append(sqrt_post_data_f_cov)

    post_data_f_means_all = tf.stack(post_data_f_means_all)
    sqrt_post_data_f_covs_all = tf.stack(sqrt_post_data_f_covs_all)
    # mean.shape = (nhyp, nobs)
    # cov.shape = (nhyp, nobs, nobs)


    maxlocs_duplicate_hyp = tf.tile(tf.expand_dims(maxlocs, 0), multiples=(nhyp,1,1))

    # invKmaxs
    invKmaxs = []

    for i in range(nhyp):
        invKmax = utils.computeNKmm_multiple_data(nmaxloc, Xsamples, maxlocs, 
                                ls[i,...], sigmas[i,...], sigma0s[i,...], 
                                dtype=dtype, inverted=True)
        invKmaxs.append(invKmax)

    invKmaxs = tf.stack(invKmaxs)


    cov_maxf_all = []
    for i in range(nhyp):
        _, cov_maxf = utils.compute_mean_var_f(
                    maxlocs, 
                    Xsamples, Ysamples, 
                    ls[i,...], sigmas[i,...], sigma0s[i,...], 
                    fullcov=True, dtype=dtype)
        cov_maxf_all.append(cov_maxf)

    cov_maxf_all = tf.stack(cov_maxf_all)


    maxlocs_prob = utils.estimate_posterior_maxx_fullcov_tf(nhyp, nmaxloc, 
                        mean_maxf_all, # nhyp, nmaxloc
                        cov_maxf_all, # nhyp, nmaxloc, nmaxloc
                        dtype=dtype)
    # nhyp, nmaxloc
    maxlocs_prob = tf.tile(
                        tf.expand_dims(maxlocs_prob, 
                                       axis=1),
                        multiples=(1,nmaxfd,1))
    # nhyp, nmaxfd, nmaxloc

    # print("temporarily try uniform distribution for maxlocs_prob!")
    # maxlocs_prob = tf.ones(shape=(nhyp, nmaxfd, nmaxloc), dtype=dtype) / tf.constant(nmaxloc, dtype=dtype)
    # # ###


    scpes = evaluate_scpes.scpes(x,
                ls, sigmas, sigma0s,
                Xsamples, Ysamples,

                xdim, nx, nobs, nhyp, nmaxloc,
                nmaxfd, nmaxf,
                nysample,
                niteration,

                maxlocs_duplicate_hyp, # (nhyp, nmaxloc, xdim)
                maxlocs_prob, # (nhyp, nmaxfd, nmaxloc)

                mean_maxf_all, # (nhyp, nmaxloc)
                std_maxf_all, # (nhyp, nmaxloc)

                post_data_f_means_all, # (nhyp, nobs,) or (nhyp, nobs,1)
                sqrt_post_data_f_covs_all, # (nhyp, nobs, nobs)

                # invKs, # (nhyp, nobs, nobs)
                invKmaxs, # (nhyp, nmaxloc, nobs+1, nobs+1)

                use_loop=True,
                parallel_iterations=1,
                dtype=dtype,
                debug=False)


    scpes_to_opt = evaluate_scpes.scpes(x_to_opt,
                ls, sigmas, sigma0s,
                Xsamples, Ysamples,

                xdim, nx_to_opt, nobs, nhyp, nmaxloc,
                nmaxfd, nmaxf,
                nysample,
                niteration,

                maxlocs_duplicate_hyp, # (nhyp, nmaxloc, xdim)
                maxlocs_prob, # (nhyp, nmaxfd, nmaxloc)

                mean_maxf_all, # (nhyp, nmaxloc)
                std_maxf_all, # (nhyp, nmaxloc)

                post_data_f_means_all, # (nhyp, nobs,) or (nhyp, nobs,1)
                sqrt_post_data_f_covs_all, # (nhyp, nobs, nobs)

                # invKs, # (nhyp, nobs, nobs)
                invKmaxs, # (nhyp, nmaxloc, nobs+1, nobs+1)

                use_loop=True,
                parallel_iterations=30,
                dtype=dtype,
                debug=False)

    train = tf.train.AdamOptimizer().minimize(-scpes_to_opt, var_list=[x_to_opt])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        maxlocs_prob_val = sess.run(maxlocs_prob,
                feed_dict = {
                    x: x_val,
                    ls: ls_val,
                    sigmas: sigmas_val,
                    sigma0s: sigma0s_val,
                    Xsamples: Xs_val,
                    Ysamples: Ys_val,
                    maxlocs: maxlocs_val
                })

        print("Maxlocs prob: {}".format(maxlocs_prob_val))


        scpes_val = sess.run(
                scpes,
                feed_dict = {
                    x: x_val,
                    ls: ls_val,
                    sigmas: sigmas_val,
                    sigma0s: sigma0s_val,
                    Xsamples: Xs_val,
                    Ysamples: Ys_val,
                    maxlocs: maxlocs_val
                })

        x_to_opt_init_val = sess.run(x_to_opt)


        for i in range(ntrain):
            if i % 100 == 0:
                print("{}/{}".format(i,ntrain))
                sys.stdout.flush()

            sess.run(train,
                feed_dict = {
                    x: x_val,
                    ls: ls_val,
                    sigmas: sigmas_val,
                    sigma0s: sigma0s_val,
                    Xsamples: Xs_val,
                    Ysamples: Ys_val,
                    maxlocs: maxlocs_val
                })
        
        x_to_opt_final_val = sess.run(x_to_opt)

    print("Optimized x: {}".format(x_to_opt_final_val))
    
    fig, axs = plt.subplots(2)

    axs[0].plot(x_val, scpes_val)
    axs[0].plot([x_to_opt_init_val.squeeze(), x_to_opt_init_val.squeeze()],
                [np.min(scpes_val), np.max(scpes_val)], 
                linestyle='--', color='b', label='init')
    axs[0].plot([x_to_opt_final_val.squeeze(), x_to_opt_final_val.squeeze()],
                [np.min(scpes_val), np.max(scpes_val)], 
                linestyle='-', color='r', label='final')
    axs[0].legend()

    axs[1].scatter(Xs_val, Ys_val)
    axs[1].scatter(maxlocs_val, np.max(scpes_val) * np.ones_like(maxlocs_val))
    plt.show()

