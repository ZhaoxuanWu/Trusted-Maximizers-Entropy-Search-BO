import sys
sys.path.insert(0, './criteria')

import os
import argparse

"""
acquistion functions: 
    'mes', 
    'ei', 
    'ucb', 
    'pes',
    'ftl'
    'sftl': sample_mp
functions:
    'func_1d_8modes',
    'func_1d_4modes',
    'negative_hartmann3d',
    'negative_hartmann4d',
    'negative_Branin'
"""

DEBUG = False


parser = argparse.ArgumentParser(description='BO methods for discrete domain of x.')
parser.add_argument('-g', '--gpu', help='gpu device index for tensorflow',
                    required=False,
                    type=str,
                    default='0')
parser.add_argument('-p', '--folder', help='folder to store the result of different BO methods',
                    required=False,
                    type=str,
                    default='.')
parser.add_argument('-c', '--criterion', help='BO acquisition function',
                    required=False,
                    type=str,
                    default='sftl')
parser.add_argument('-e', '--mode', help='mode: empirical, sample, ep',
                    required=False,
                    type=str,
                    default='sample')
# parser.add_argument('-n', '--noisevar', help='noise variance',
#                     required=False,
#                     type=float,
#                     default=0.0001)
# parser.add_argument('-l', '--lengthscale', help='lengthscale of the GP',
#                     required=False,
#                     type=float,
#                     default=10.0)
parser.add_argument('-q', '--numqueries', help='number/budget of queries',
                    required=False,
                    type=int,
                    default=50)
parser.add_argument('-r', '--numruns', help='number of random experiments',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-d', '--deterministic', help='1 use deterministic version of acquisition, 0 use stochastic version',
                    required=False,
                    type=int,
                    default=0)
parser.add_argument('-s', '--numhyps', help='number of sampled hyperparameters',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-m', '--nmax', help='number of function samples',
                    required=False,
                    type=int,
                    default=5)
parser.add_argument('-a', '--nparal', help='number of parallel iterations',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-o', '--nsto', help='number of stochastic evaluation of acquisition',
                    required=False,
                    type=int,
                    default=50)
parser.add_argument('-y', '--nysample', help='number of y samples to evaluate acquisition',
                    required=False,
                    type=int,
                    default=10)
parser.add_argument('--ninit', help='number of initial observations',
                    required=False,
                    type=int,
                    default=2)
parser.add_argument('--function', help='function to optimize: \
                                        func_1d_8modes, \
                                        func_1d_4modes, \
                                        negative_hartmann3d, \
                                        negative_hartmann4d, \
                                        negative_Branin',
                    required=False,
                    type=str,
                    default='func_1d_4modes')
parser.add_argument('-t', '--dtype', help='type of float: float32 or float64',
                    required=False,
                    type=str,
                    default='float64')




args = parser.parse_args()

# print all arguments
print('================================')
for arg in vars(args):
    print(arg, getattr(args, arg))
print('================================')

gpu_device_id = args.gpu

criterion = args.criterion


folder_prefix = args.folder
folder = '{}/{}'.format(folder_prefix, args.criterion)
if not os.path.exists(folder):
    os.makedirs(folder)

nquery = args.numqueries
nrun = args.numruns
nhyp = args.numhyps

deterministic = args.deterministic
mode = args.mode

nmax = args.nmax
nstoiter = args.nsto
nysample = args.nysample
parallel_iterations = args.nparal
n_initial_training_x = args.ninit
func_name = args.function

print("nrun: {}".format(nrun))
print("nquery: {}".format(nquery))
print("nhyp: {}".format(nhyp))
print("nmax: {}".format(nmax))
print("n_initial_training_x: {}".format(n_initial_training_x))
print("Function: {}".format(func_name))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device_id


import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp
import scipy as sp 
import time 
import scipy.stats as spst


import matplotlib.pyplot as plt 


import utils 
import optfunc
import functions

import ep 
import empirical_approximation

import evaluate_mes
import evaluate_pes
import evaluate_mp
import evaluate_mp_lite
import evaluate_sample_mp
import evaluate_ei
import evaluate_ucb


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True 


if args.dtype == 'float32':
    dtype = tf.float32
    nptype = np.float32
elif args.dtype == 'float64':
    dtype = tf.float64
    nptype = np.float64
else:
    raise Exception("Unknown dtype: {}".format(args.dtype))


duplicate_resolution=1e-3
print("duplicate_resolution = {}".format(duplicate_resolution))

nsample_to_est_performance = 100
print("nsample to estimate performance: {}".format(nsample_to_est_performance))


def optimize_mean_f(xdim, nhyp, 
                    Xsamples_plc, Ysamples_plc, 
                    invKs, 
                    xs, 
                    ls, sigmas, sigma0s, 
                    dtype=tf.float32):
    """
    sess.run: 
        assign
        train for n times
        max_x
    """
    mean_f = utils.compute_mean_f(xs, xdim, nhyp, 
                    Xsamples_plc, Ysamples_plc, 
                    ls, sigmas, sigma0s, 
                    invKs, dtype=dtype)
    # shape = (nx,)

    idx = tf.argmax(mean_f)
    return xs[idx,:], mean_f[idx], idx


def get_required_placeholders(criterion, crit_params, 
                            dtype, is_debug_mode=False):

    nhyp = crit_params['nhyp']
    xdim = crit_params['xdim']
    nx_to_optimize = crit_params['nx_to_optimize']
    batchsize = crit_params['batchsize'] # batchsize to evaluate/optimize criterion
    # nmax = crit_params['nmax']

    invKs_plc = tf.placeholder(shape=(nhyp,None,None), dtype=dtype, name='invKs_plc')
    # (nhyp, nobs, nobs)
    sample_max_fs_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None), name='sample_max_xs_plc')
    # (nhyp,nmax)
    sample_max_xs_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None, xdim), name='sample_max_fs_plc')
    # (nhyp,nmax,xdim)
    
    test_xs_plc = tf.placeholder(dtype=dtype, 
                shape=(None, crit_params['xdim']), 
                name='test_xs')
    # (ntest,xdim)s
    max_probs_plc = tf.placeholder(dtype=dtype, shape=( crit_params['nhyp'], None), name='max_probs_plc')
    # (nhyp, nmax)
    post_mean_tests_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None), 
            name='post_mean_tests_plc')
    # (nhyp, nmax, ntest)
    post_cov_tests_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None, None), 
            name='post_cov_tests_plc')
    # (nhyp, nmax, ntest, ntest)

    invpNKs_plc = tf.placeholder(dtype=dtype, 
            shape=( crit_params['nhyp'], None, None), 
            name='invpNKs_plc')
    # (nhyp, nobs+ntest, nobs+ntest)

    X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name='X_plc')
    Y_plc = tf.placeholder(dtype=dtype, shape=(None, 1), name='Y_plc')
    
    xs_to_optimize = tf.placeholder(dtype=dtype, 
            shape=(None, xdim), 
            name='xs_to_optimize')
    xs_batch = tf.placeholder(dtype=dtype,
            shape=(None, xdim),
            name='xs_batch')

    max_observed_y_plc = tf.placeholder(dtype=dtype, shape=(), 
        name = 'max_observed_y_plc')
    
    required_placeholders = {
        'X': X_plc,
        'Y': Y_plc,
        'xs_to_optimize': xs_to_optimize,
        'xs_batch': xs_batch
    }

    if criterion == 'mes':
        required_placeholders['sample_max_fs'] = sample_max_fs_plc
        required_placeholders['invKs'] = invKs_plc

    elif criterion == 'ei':
        required_placeholders['invKs'] = invKs_plc
        required_placeholders['max_observed_y'] = max_observed_y_plc
    
    elif criterion == 'ucb':
        required_placeholders['invKs'] = invKs_plc
        
    elif criterion == 'pes':
        invKmaxsams_plc = tf.placeholder(dtype=dtype, 
                shape=(nhyp, None, None, None), # nhyp, nmax, nobs+1, nobs+1
                name='invKmaxsams_plc')

        required_placeholders['invKs'] = invKs_plc
        required_placeholders['sample_max_xs'] = sample_max_xs_plc
        required_placeholders['invKmaxsams'] = invKmaxsams_plc
        required_placeholders['max_observed_y'] = max_observed_y_plc

    elif criterion == 'ftl':
        required_placeholders['test_xs'] = test_xs_plc
        required_placeholders['max_probs'] = max_probs_plc
        required_placeholders['post_mean_tests'] = post_mean_tests_plc
        required_placeholders['post_cov_tests'] = post_cov_tests_plc
        required_placeholders['invKs'] = invKs_plc
        required_placeholders['invpNKs'] = invpNKs_plc
    
    elif criterion == 'sftl':
        post_test_samples_plc = tf.placeholder(dtype=dtype, 
                                shape=(nhyp, None, None, None), 
                                name='post_test_samples_plc')
        post_test_masks_plc = tf.placeholder(dtype=tf.bool,
                                shape=(nhyp, None, None), 
                                name='post_test_masks_plc')

        required_placeholders['test_xs'] = test_xs_plc
        required_placeholders['max_probs'] = max_probs_plc
        required_placeholders['post_test_samples'] = post_test_samples_plc
        required_placeholders['post_test_masks'] = post_test_masks_plc
        required_placeholders['invKs'] = invKs_plc
        required_placeholders['invpNKs'] = invpNKs_plc
    
    if is_debug_mode:
        required_placeholders['xs_to_debug'] = tf.placeholder(
                    shape=(None,xdim), dtype=dtype, name='xs_plot')

    return required_placeholders


def get_intermediate_tensors(criterion, crit_params, 
        required_placeholders, 
        ls, sigmas, sigma0s, # tf.variable to load
        dtype,
        is_debug_mode=False):

    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    nmax = crit_params['nmax']
    batchsize = crit_params['batchsize'] # batchsize to evaluate/optimize criterion

    xs_to_optimize = required_placeholders['xs_to_optimize']
    xs_batch = required_placeholders['xs_batch']
    Xsamples_plc = required_placeholders['X']
    Ysamples_plc = required_placeholders['Y']

    invKs = utils.precomputeInvKs(xdim, nhyp, 
                ls, sigmas, sigma0s, 
                Xsamples_plc, dtype)
    # nhyp x nobs x nobs


    # Optimize mean function
    max_meanf_x, max_meanf_f, max_meanf_idx  \
                = optimize_mean_f(
                        xdim, nhyp, 
                        Xsamples_plc, Ysamples_plc, 
                        required_placeholders['invKs'], 
                        xs_to_optimize, 
                        ls, sigmas, sigma0s, 
                        dtype=dtype)


    # Sampling functions from the GP posterior
    obs_standard_norm = tfp.distributions.MultivariateNormalDiag(
                                loc        = tf.zeros(nx, dtype=dtype),
                                scale_diag = tf.ones(nx, dtype=dtype) )
    obs_samples = obs_standard_norm.sample(sample_shape=nmax)
    # (nmax, nx)

    sample_max_xs = []
    sample_max_fs = []

    x_to_opt_meanf_all = []
    x_to_opt_covf_all = []

    for i in range(nhyp):
        x_to_opt_meanf, x_to_opt_covf = utils.compute_mean_var_f(
                            xs_to_optimize, 
                            Xsamples_plc, Ysamples_plc, 
                            ls[i,...], sigmas[i,...], sigma0s[i,...], 
                            fullcov=True, 
                            dtype=dtype)
        
        x_to_opt_meanf_all.append(x_to_opt_meanf)
        x_to_opt_covf_all.append(x_to_opt_covf)

        x_to_opt_sqrtcovf = utils.sqrtm(x_to_opt_covf)

        x_to_opt_fsamples = x_to_opt_sqrtcovf @ tf.transpose(obs_samples)\
                + tf.reshape(x_to_opt_meanf, shape=(nx,1))
        # shape = (nx, nmax)

        max_idx = tf.argmax(x_to_opt_fsamples, axis=0)
        sample_max_x = tf.gather(xs_to_optimize, max_idx, axis=0)
        # nmax, xdim
        sample_max_f = tf.reduce_max(x_to_opt_fsamples, axis=0)
        # nmax,

        sample_max_xs.append( sample_max_x )
        sample_max_fs.append( sample_max_f )

    sample_max_xs = tf.stack(sample_max_xs)
    # nhyp, nmax, xdim
    sample_max_fs = tf.stack(sample_max_fs)
    # nhyp, nmax

    x_to_opt_meanf_all = tf.stack(x_to_opt_meanf_all)
    x_to_opt_covf_all = tf.stack(x_to_opt_covf_all)

    max_observed_y = tf.reduce_max(Ysamples_plc)

    max_crit_x, max_crit, max_crit_idx = optimize_criterion(
                            xs_batch, batchsize,
                            criterion, crit_params,
                            ls, sigmas, sigma0s,
                            required_placeholders,
                            dtype=dtype)

    intermediate_tensors = {
        'max_meanf_x': max_meanf_x,
        'max_meanf_f': max_meanf_f,
        'max_meanf_idx': max_meanf_idx,
        'max_crit_x': max_crit_x,
        'max_crit': max_crit,
        'max_crit_idx': max_crit_idx,
        'invKs': invKs,

        # for distributional performance measure
        'x_to_opt_meanf': x_to_opt_meanf_all,
        'x_to_opt_covf': x_to_opt_covf_all,
    }

    if criterion == 'mes':
        intermediate_tensors['sample_max_xs'] = sample_max_xs
        intermediate_tensors['sample_max_fs'] = sample_max_fs

    elif criterion == 'ei':
        intermediate_tensors['max_observed_y'] = max_observed_y

    elif criterion == 'ucb':
        pass
    
    elif criterion == 'pes':
        invKmaxsams = utils.eval_invKmaxsams(xdim, nhyp, nmax, 
                            ls, sigmas, sigma0s, 
                            required_placeholders['X'], 
                            required_placeholders['sample_max_xs'], 
                            dtype)

        intermediate_tensors['invKmaxsams'] = invKmaxsams         
        intermediate_tensors['sample_max_xs'] = sample_max_xs
        intermediate_tensors['sample_max_fs'] = sample_max_fs
        intermediate_tensors['max_observed_y'] = max_observed_y

    elif criterion in ['ftl', 'sftl']:
        mean_tests_given_data = [] # nhyp, ntest
        cov_tests_given_data = [] # nhyp, ntest, ntest

        for i in range(crit_params['nhyp']):
            mean_test_i, cov_test_i = utils.compute_mean_var_f(required_placeholders['test_xs'],
                    Xsamples_plc, Ysamples_plc,
                    ls[i,...], sigmas[i,...], sigma0s[i,...],
                    fullcov=True,
                    dtype=dtype)

            mean_tests_given_data.append(tf.squeeze(mean_test_i))
            cov_tests_given_data.append(cov_test_i)

        mean_tests_given_data = tf.stack(mean_tests_given_data)
        # (nhyp, ntest)
        cov_tests_given_data = tf.stack(cov_tests_given_data)
        # (nhyp, ntest, ntest)
        
        _, invpNKs = evaluate_mp.get_pNK_test_obs(
                        ls, sigmas, sigma0s, 
                        nhyp,
                        required_placeholders['X'],
                        required_placeholders['test_xs'],
                        dtype=dtype)
        
        intermediate_tensors['invpNKs'] = invpNKs
        intermediate_tensors['sample_max_xs'] = sample_max_xs
        intermediate_tensors['sample_max_fs'] = sample_max_fs
        intermediate_tensors['mean_test_given_data'] = mean_tests_given_data
        intermediate_tensors['cov_test_given_data'] = cov_tests_given_data

    if is_debug_mode:
        mean_f_debug = utils.compute_mean_f(
                                    xs_to_debug, 
                                    xdim, nhyp, 
                                    Xsamples_plc, Ysamples_plc, 
                                    ls, sigmas, sigma0s, 
                                    required_placeholders['invKs'], 
                                    dtype)

        crit_to_debug = evaluate_criterion(
                        xs_to_debug, xs_to_debug.shape[0],
                        criterion, crit_params,
                        ls, sigmas, sigma0s,
                        required_placeholders,
                        dtype=dtype)

        intermediate_tensors['mean_f_debug'] = mean_f_debug
        intermediate_tensors['crit_to_debug'] = crit_to_debug

    return intermediate_tensors


def get_placeholder_values(sess, 
        criterion, crit_params,
        required_placeholders,
        intermediate_tensors,

        X_np, Y_np,
        xs_to_optimize_np,
        
        dtype=tf.float32,
        is_debug_mode=False):
    xdim = crit_params['xdim']
    nmax = crit_params['nmax']
    # is_test_include_obs == True
    # nmax == ntest

    values = {'query_x': None}

    if 'max_observed_y' in intermediate_tensors:
        max_observed_y_np = sess.run(intermediate_tensors['max_observed_y'],
            feed_dict = { required_placeholders['Y']: Y_np })
        values['max_observed_y'] = max_observed_y_np

    if 'invKs' in intermediate_tensors:
        invKs_np = sess.run(intermediate_tensors['invKs'], 
            feed_dict = {
                required_placeholders['X']: X_np
            })
        values['invKs'] = invKs_np


    if 'sample_max_xs' in intermediate_tensors or 'sample_max_fs' in intermediate_tensors:
        # Optimize function samples
        sample_max_fs_np, sample_max_xs_np \
            = sess.run([
                intermediate_tensors['sample_max_fs'],
                intermediate_tensors['sample_max_xs'] ],
                feed_dict = {
                    required_placeholders['xs_to_optimize']: xs_to_optimize_np,
                    required_placeholders['X']: X_np,
                    required_placeholders['Y']: Y_np,
                    required_placeholders['invKs']: invKs_np
                })
        values['sample_max_fs'] = sample_max_fs_np 
        values['sample_max_xs'] = sample_max_xs_np

    # Optimize acquisition functions
    if criterion in ['pes', 'ftl', 'sftl']:
        print("Only implement for known GP hyperparameter only!")

        unique_sample_max_xs_np = utils.remove_duplicates_np(
                            sample_max_xs_np[0,...], 
                            resolution=duplicate_resolution)

        if unique_sample_max_xs_np.shape[0] == 1 and criterion in ['ftl', 'sftl']:
            values['query_x'] = unique_sample_max_xs_np
            return values

        if criterion == 'pes':
            invKmaxsams_np = sess.run(
                    intermediate_tensors['invKmaxsams'],
                    feed_dict = {
                        required_placeholders['X']: X_np,
                        required_placeholders['Y']: Y_np,
                        required_placeholders['sample_max_xs']: sample_max_xs_np
                    })
            values['invKmaxsams'] = invKmaxsams_np

        elif criterion in ['ftl', 'sftl']:
            max_n_test = crit_params['max_n_test']

            test_xs_np = unique_sample_max_xs_np.reshape(-1,xdim)
            if max_n_test > test_xs_np.shape[0]:
                # include more inputs from X_np into test_xs_np
                n_extra = max_n_test - test_xs_np.shape[0]
                n_extra = n_extra if n_extra < X_np.shape[0] else X_np.shape[0]

                print("Include {} more x from X_np into test_xs_np: randomly".format(n_extra))
                print("Another option is to select based on mean_f")

                idxs = np.array(list(range(X_np.shape[0])))
                np.random.shuffle(idxs)
                test_xs_np = np.concatenate([test_xs_np, X_np[idxs[:n_extra],:]], axis=0)

            test_xs_np = utils.remove_duplicates_np(
                        test_xs_np, 
                        resolution=duplicate_resolution)

            ntest = test_xs_np.shape[0]

            if ntest == 1:
                values['query_x'] = test_xs_np
                return values

            mean_tests_given_data_np, cov_tests_given_data_np \
                = sess.run([
                    intermediate_tensors['mean_test_given_data'],
                    intermediate_tensors['cov_test_given_data'] ],
                    feed_dict = {
                        required_placeholders['test_xs']: test_xs_np,
                        required_placeholders['X']: X_np,
                        required_placeholders['Y']: Y_np
                    })

            test_xs_np, max_probs_np, \
            mean_tests_given_data_np, \
            cov_tests_given_data_np, \
            post_test_v0, post_test_v1 \
                = utils.get_testidxs_stats(nhyp, 
                        test_xs_np, # (ntest, xdim)
                        mean_tests_given_data_np, # (nhyp, ntest)
                        cov_tests_given_data_np, # (nhyp, ntest, ntest)
                        mode=crit_params['mode'],
                        nsample = crit_params['ntestsample'])

            if test_xs_np.shape[0] == 1:
                values['query_x'] = test_xs_np
                return values
                
            invpNKs_np = sess.run(
                        intermediate_tensors['invpNKs'], 
                        feed_dict={
                            required_placeholders['test_xs']: test_xs_np,
                            required_placeholders['X']: X_np,
                            required_placeholders['Y']: Y_np
                        })

            values['test_xs'] = test_xs_np
            values['max_probs'] = max_probs_np
            values['invpNKs'] = invpNKs_np

            if criterion == 'sftl':
                post_test_samples_np = post_test_v0
                post_test_masks_np = post_test_v1.astype(bool)

                values['post_test_samples'] = post_test_samples_np 
                values['post_test_masks'] = post_test_masks_np
            elif criterion == 'ftl':
                post_mean_test_np = post_test_v0 
                post_cov_test_np = post_test_v1

                values['post_mean_tests'] = post_mean_test_np 
                values['post_cov_tests'] = post_cov_test_np

    return values



def evaluate_criterion(xs, nx,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=tf.float32):
    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    
    Xsamples = required_placeholders['X']
    Ysamples = required_placeholders['Y']
    nobs = tf.shape(Xsamples)[0]
 
    if criterion == 'mes':
        nmax = crit_params['nmax']
        
        sample_max_fs = required_placeholders['sample_max_fs']
        invKs = required_placeholders['invKs']

        vals = evaluate_mes.mes(xs, 
            xdim, nmax, nhyp, 
            Xsamples, Ysamples, 
            ls, sigmas, sigma0s, 
            sample_max_fs, invKs, 
            dtype=dtype)

    elif criterion == 'ei':
        print("EI uses max observed y.")
        max_observed_y = required_placeholders['max_observed_y']
        invKs = required_placeholders['invKs']

        vals = evaluate_ei.ei(xs, 
                xdim, nhyp, 
                Xsamples, Ysamples, 
                ls, sigmas, sigma0s, 
                invKs, 
                max_observed_y, 
                dtype=dtype)

    elif criterion == 'ucb':
        beta = 1.5
        print("UCB uses beta={}.".format(beta))

        invKs = required_placeholders['invKs']

        vals = evaluate_ucb.ucb(xs, 
                xdim, nhyp, 
                Xsamples, Ysamples, 
                ls, sigmas, sigma0s, 
                invKs, 
                dtype=dtype, beta=beta)

    elif criterion == 'pes':
        nmax = crit_params['nmax']

        max_observed_y = required_placeholders['max_observed_y']
        invKs = required_placeholders['invKs']
        invKmaxsams = required_placeholders['invKmaxsams']
        sample_max_xs = required_placeholders['sample_max_xs']

        vals = evaluate_pes.pes(xs, 
                xdim, nmax, nhyp, 
                Xsamples, Ysamples, 
                ls, sigmas, sigma0s,
                sample_max_xs, 
                invKs, invKmaxsams, 
                max_observed_y, 
                dtype=dtype, n_x=nx)

    elif criterion == 'ftl':
        nysample = crit_params['nysample']
        deterministic = crit_params['deterministic']
        nstoiter = crit_params['nstoiter']
        parallel_iterations = crit_params['parallel_iterations']

        test_xs = required_placeholders['test_xs']
        max_probs = required_placeholders['max_probs']
        post_mean_tests = required_placeholders['post_mean_tests']
        post_cov_tests = required_placeholders['post_cov_tests']

        invKs = required_placeholders['invKs']
        invpNKs = required_placeholders['invpNKs']
        
        vals = evaluate_mp.mp(xs,
                ls, sigmas, sigma0s,
                Xsamples, Ysamples,
                
                xdim, nx, nobs, nhyp,
                nysample,
                
                test_xs, max_probs,
                
                post_mean_tests,
                post_cov_tests,
                
                invKs,
                invpNKs,
                
                stochastic=1 - deterministic,
                dtype=dtype,
                niteration=nstoiter,
                use_loop=True,
                parallel_iterations=parallel_iterations)
    
    elif criterion == 'sftl':
        nysample = crit_params['nysample']
        nstoiter = crit_params['nstoiter']
        parallel_iterations = crit_params['parallel_iterations']
        
        test_xs = required_placeholders['test_xs']
        max_probs = required_placeholders['max_probs']
        post_test_samples = required_placeholders['post_test_samples']
        post_test_masks = required_placeholders['post_test_masks']

        invpNKs = required_placeholders['invpNKs']

        vals = evaluate_sample_mp.mp(xs, # nx, xdim
                    ls, sigmas, sigma0s,
                    Xsamples, Ysamples, # (nobs,xdim), (nobs,1)

                    xdim, nx, nobs, nhyp, 
                    nysample,

                    test_xs, # ntest, xdim (same for all hyp)
                    max_probs, # nhyp, nmax

                    post_test_samples, # nhyp, nmax, ntest, nsample
                    post_test_masks, # nhyp, nmax, nsample, dtype: tf.bool

                    invpNKs, # nhyp, ntest+nobs, ntest+nobs

                    dtype=dtype,

                    niteration=nstoiter,
                    use_loop=True,
                    parallel_iterations=parallel_iterations)

    else:
        raise Exception("Unknown criterion: {}".format(criterion))

    return vals



def optimize_criterion(xs, nx,
                criterion, crit_params,
                ls, sigmas, sigma0s,
                required_placeholders,
                dtype=tf.float32):

    fvals = evaluate_criterion(xs, nx,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=dtype)

    idx = tf.argmax(fvals) # TODO: check the shape of fvals
    return xs[idx,...], fvals[idx], idx
    # return assign, train, max_x






tf.reset_default_graph()

f_info = getattr(functions, func_name)()
print("Information of function:")
for k in f_info:
    if k != 'xs':
        print("{}: {}".format(k, f_info[k]))
    else:
        print("xs.shape: {}".format(f_info['xs'].shape))

f = f_info['function'] 
xmin = f_info['xmin']
xmax = f_info['xmax']
xs_to_optimize_np = f_info['xs']
nx = xs_to_optimize_np.shape[0]

xdim = f_info['xdim']

true_l = f_info['RBF.lengthscale']
true_sigma = f_info['RBF.variance']
true_sigma0 = f_info['noise.variance']
true_maximum = f_info['maximum']

ls_np = true_l.reshape(-1,xdim).astype(nptype)
sigmas_np = np.array([ true_sigma ], dtype=nptype)
sigma0s_np = np.array([ true_sigma0 ], dtype=nptype)

seed = 1

# seed_func = 1
# f = functions.func_gp_prior(xdim, true_l, true_sigma, seed=seed_func)
# nx = 400
# xmin = 0.0
# xmax = 10.0
# xs_to_optimize_np = np.linspace(xmin, xmax, nx).reshape(-1,1)

if DEBUG:
    xs_plot_np = np.linspace(np.min(xs_to_optimize_np), np.max(xs_to_optimize_np), 100).reshape(-1,1)


print("True GP hyperparameters: l:{} sigma:{} sigma0(noise var):{}".format(true_l, true_sigma, true_sigma0))
print("Discrete x: linspace({}, {}, {})".format(xmin, xmax, nx))


# print("Ground truth function: func_gp_prior with l:{}, sigma:{}, sigma0:{}, seed_func:{}".format(true_l, true_sigma, true_sigma0, seed_func))
print("nhyp: {}".format(nhyp))
print("nrun:{}, nqueries:{}".format(nrun, nquery))
print("____________________________________________")



ls_toload = tf.get_variable(dtype=dtype, shape=(nhyp,xdim), name='ls')
sigmas_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigmas') 
sigma0s_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigma0s')

if criterion == 'ftl':
    if mode not in ['empirical', 'ep']:
        raise Exception("Unknown mode ({}) for criterion {}".format(mode, criterion))
elif criterion == 'sftl':
    if mode != 'sample':
        raise Exception("Unknown mode ({}) for criterion {}".format(mode, criterion)) 

crit_params = {'nhyp': nhyp,
               'xdim': xdim,
               'nmax': nmax,
               'max_n_test': nmax,
               'ntestsample': 1000,
               'nysample': nysample,
               'mode': mode,
               'nstoiter': nstoiter,
               'parallel_iterations': parallel_iterations,
               'deterministic': deterministic,
               'nx_to_optimize': nx,
               'batchsize': 5 if criterion == 'pes' else nx}

print("crit_params:", crit_params)


required_placeholder_keys = {
    'mes': ['xs_batch', 'batchsize', 'X', 'Y', 'sample_max_fs', 'invKs'],
    'ei': ['xs_batch', 'batchsize', 'X', 'Y', 'max_observed_y', 'invKs'],
    'ucb': ['xs_batch', 'batchsize', 'X', 'Y', 'invKs'],
    'pes': ['xs_batch', 'batchsize', 'X', 'Y', 'invKs', 
            'invKmaxsams', 'sample_max_xs', 'max_observed_y'],
    'ftl': ['xs_batch', 'batchsize', 'X', 'Y', 'test_xs', 'max_probs',
            'post_mean_tests', 'post_cov_tests', 
            'invKs', 'invpNKs'],
    'sftl': ['xs_batch', 'batchsize', 'X', 'Y', 'test_xs', 'max_probs',
            'post_test_samples', 'post_test_masks', 
            'invpNKs'] }

required_placeholders = get_required_placeholders(criterion, 
                    crit_params, dtype=dtype, is_debug_mode=DEBUG)

intermediate_tensors = get_intermediate_tensors(criterion, crit_params,
                    required_placeholders,
                    ls_toload, sigmas_toload, sigma0s_toload,
                    dtype=dtype,
                    is_debug_mode=DEBUG)



all_guess_xx = np.zeros([nrun, nquery+1, xdim])
all_guesses = np.zeros([nrun, nquery+1])

all_regret_mean = np.zeros([nrun, nquery+1])
all_regret_std = np.zeros([nrun, nquery+1])

all_xx = np.zeros([nrun, nquery + n_initial_training_x, xdim])
all_ff = np.zeros([nrun, nquery + n_initial_training_x]) 
all_yy = np.zeros([nrun, nquery + n_initial_training_x])


with tf.Session(config=gpu_config) as sess:
    # tf.set_random_seed(seed)
    # print("tf random seed: {}".format(seed))

    for nr in range(nrun):
        rseed = seed + nr
        print("tf and np random seed: {}".format(rseed))
        np.random.seed(rseed)
        tf.set_random_seed(rseed)

        sample_idxs = np.random.randint(nx, size = n_initial_training_x)
        Xsamples_np = xs_to_optimize_np[sample_idxs,:]
        Fsamples_np = f(Xsamples_np).reshape(-1,1).astype(nptype)
        Ysamples_np = (Fsamples_np + np.random.randn(Xsamples_np.shape[0],1) * np.sqrt(true_sigma0)).astype(nptype)

        print("")
        
        for nq in range(nquery):

            startime_query = time.time()

            # for randomly drawing different functions
            sess.run(tf.global_variables_initializer())

            print("")
            print("{}:{}.=================".format(nr, nq))
            print("  X: {}".format(Xsamples_np.T))
            print("  Y: {}".format(Ysamples_np.T))

            ls_toload.load(ls_np, sess)
            sigmas_toload.load(sigmas_np, sess)
            sigma0s_toload.load(sigma0s_np, sess)
            
            placeholder_values = get_placeholder_values(sess,
                        criterion, crit_params,
                        required_placeholders, intermediate_tensors,
                        
                        Xsamples_np, Ysamples_np,
                        xs_to_optimize_np,
                        
                        dtype=dtype,
                        is_debug_mode=DEBUG)


            # Optimize for best guess
            max_meanf_x_np, max_meanf_f_np \
                        = sess.run([
                            intermediate_tensors['max_meanf_x'], 
                            intermediate_tensors['max_meanf_f'] ], 
                            feed_dict={
                                required_placeholders['xs_to_optimize']: xs_to_optimize_np,
                                required_placeholders['X']: Xsamples_np,
                                required_placeholders['Y']: Ysamples_np,
                                required_placeholders['invKs']: placeholder_values['invKs']
                            })

            all_guess_xx[nr,nq,:] = max_meanf_x_np
            all_guesses[nr,nq] = f(max_meanf_x_np.reshape(-1,xdim)).squeeze()

            print("GUESS: {} (f={})".format(max_meanf_x_np.squeeze(), 
                                           all_guesses[nr,nq]))
            print("all guesses: {}".format(all_guesses[nr,:(nq+1)]))
            print("")



            # print("Not reporting distribution of regret!")
            print("Reporting distribution of regret!")
            # Optimize for distributional performance measure
            x_to_opt_meanf_np, x_to_opt_covf_np \
                = sess.run([
                intermediate_tensors['x_to_opt_meanf'],
                intermediate_tensors['x_to_opt_covf']],
                feed_dict = {
                    required_placeholders['xs_to_optimize']: xs_to_optimize_np,
                    required_placeholders['X']: Xsamples_np,
                    required_placeholders['Y']: Ysamples_np,
                })

            regret_mean, regret_std \
                = utils.evaluate_discrete_regret_distribution_np(nhyp, 
                            true_maximum,
                            xs_to_optimize_np, 
                            x_to_opt_meanf_np, x_to_opt_covf_np, 
                            nsample=nsample_to_est_performance, func=f)
            
            all_regret_mean[nr,nq] = regret_mean
            all_regret_std[nr,nq] = regret_std
            print("REGRET: {} {}".format(regret_mean, regret_std))
            print("all regret mean: {}".format(all_regret_mean[nr,:(nq+1)]))
            print("")


            if placeholder_values['query_x'] is not None:
                # no need to optimize the criterion
                # as there is only 1 maximizer sample 
                # (for pes, ftl-related)
                max_crit_x_np = placeholder_values['query_x']
                max_crit_np = None

            else:
                max_crit_np = []
                max_crit_x_np = []

                for i in range(0, crit_params['nx_to_optimize'], crit_params['batchsize']):
                    xs_batch = xs_to_optimize_np[i:(i+crit_params['batchsize'])]

                    feed_dict = {
                                required_placeholders['xs_batch']: xs_batch,

                                required_placeholders['X']: Xsamples_np,
                                required_placeholders['Y']: Ysamples_np}

                    for key in required_placeholder_keys[criterion]:
                        if key not in ['xs_batch', 'batchsize', 'X', 'Y']:
                            feed_dict[ required_placeholders[key] ] = placeholder_values[key]

                    max_crit_x_batch_np, max_crit_batch_np =  sess.run(
                            [ intermediate_tensors['max_crit_x'],
                            intermediate_tensors['max_crit'] ],
                            feed_dict = feed_dict)

                    max_crit_np.append(max_crit_batch_np)
                    max_crit_x_np.append(max_crit_x_batch_np)
            
                max_crit_np = np.array(max_crit_np)
                max_crit_idx = np.argmax(max_crit_np)
                max_crit_x_np = max_crit_x_np[max_crit_idx]

            query_x = max_crit_x_np.reshape(-1,xdim)
            query_f = f(query_x).reshape(-1,1) 
            query_y = query_f + np.random.randn(query_x.shape[0],1) * np.sqrt(true_sigma0)
            
            if 'sample_max_xs' in placeholder_values:
                print("maximizer samples: {}".format(
                    placeholder_values['sample_max_xs']))
            
            if 'test_xs' in placeholder_values:
                print("test candidate: {}".format(
                    placeholder_values['test_xs']))
                print("max probs: {}".format(placeholder_values['max_probs']))
            print("QUERY: {}".format(query_x))

            print("end query in {:.4f}s".format(time.time() - startime_query))
            sys.stdout.flush()


            Xsamples_np = np.concatenate([Xsamples_np, query_x], axis=0)
            Fsamples_np = np.concatenate([Fsamples_np, query_f], axis=0)
            Ysamples_np = np.concatenate([Ysamples_np, query_y], axis=0)


        placeholder_values = get_placeholder_values(sess,
                    criterion, crit_params,
                    required_placeholders, intermediate_tensors,
                    
                    Xsamples_np, Ysamples_np,
                    xs_to_optimize_np,
                    
                    dtype=dtype,
                    is_debug_mode=DEBUG)


        # Optimize for best guess
        max_meanf_x_np, max_meanf_f_np \
                    = sess.run([
                        intermediate_tensors['max_meanf_x'], 
                        intermediate_tensors['max_meanf_f'] ], 
                        feed_dict={
                            required_placeholders['xs_to_optimize']: xs_to_optimize_np,
                            required_placeholders['X']: Xsamples_np,
                            required_placeholders['Y']: Ysamples_np,
                            required_placeholders['invKs']: placeholder_values['invKs']
                        })

        all_guess_xx[nr,nquery,:] = max_meanf_x_np
        all_guesses[nr,nquery] = f(max_meanf_x_np.reshape(-1,xdim)).squeeze()

        print("GUESS: {} (f={})".format(max_meanf_x_np.squeeze(), 
                                        all_guesses[nr,nquery]))
        print("all guesses: {}".format(all_guesses[nr,:]))
        print("")

    
        # Optimize for distributional performance measure
        x_to_opt_meanf_np, x_to_opt_covf_np \
            = sess.run([
            intermediate_tensors['x_to_opt_meanf'],
            intermediate_tensors['x_to_opt_covf']],
            feed_dict = {
                required_placeholders['xs_to_optimize']: xs_to_optimize_np,
                required_placeholders['X']: Xsamples_np,
                required_placeholders['Y']: Ysamples_np,
            })

        regret_mean, regret_std \
            = utils.evaluate_discrete_regret_distribution_np(nhyp, 
                        xs_to_optimize_np, x_to_opt_meanf_np, x_to_opt_covf_np, nsample=nsample_to_est_performance, func=f)
        
        all_regret_mean[nr,nquery] = regret_mean
        all_regret_std[nr,nquery] = regret_std
        print("REGRET: {} {}".format(regret_mean, regret_std))
        print("all regret mean: {}".format(all_regret_mean[nr,:]))
        print("")


        all_xx[nr,:,:] = Xsamples_np
        all_ff[nr,:] = Fsamples_np.squeeze()
        all_yy[nr,:] = Ysamples_np.squeeze()


        np.save('{}/{}_xx.npy'.format(folder, criterion), all_xx)
        np.save('{}/{}_ff.npy'.format(folder, criterion), all_ff)
        np.save('{}/{}_yy.npy'.format(folder, criterion), all_yy)
        np.save('{}/{}_guess_ff.npy'.format(folder, criterion), all_guesses)
        np.save('{}/{}_guess_xx.npy'.format(folder, criterion), all_guess_xx)
        np.save('{}/{}_regret_mean.npy'.format(folder, criterion), all_regret_mean)
        np.save('{}/{}_regret_std.npy'.format(folder, criterion), all_regret_std)