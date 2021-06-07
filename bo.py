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
    'sftl'
functions:
    'func_1d_8modes',
    'func_1d_4modes',
    'negative_hartmann3d',
    'negative_hartmann4d',
    'negative_Branin'
"""

DEBUG = False
use_GPU_for_sample_functions = False


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
                    default='ucb')
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
                    default=30)
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
                    default=10)
parser.add_argument('-u', '--nfeature', help='number of features to sample functions',
                    required=False,
                    type=int,
                    default=100)
parser.add_argument('-a', '--nparal', help='number of parallel iterations',
                    required=False,
                    type=int,
                    default=1)
parser.add_argument('-o', '--nsto', help='number of stochastic evaluation of acquisition',
                    required=False,
                    type=int,
                    default=60)
parser.add_argument('-y', '--nysample', help='number of y samples to evaluate acquisition',
                    required=False,
                    type=int,
                    default=5)
parser.add_argument('--ntrain', help='number of optimizing iterations',
                    required=False,
                    type=int,
                    default=100)
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
                    default='train_mnist_logistic')
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

# noise_var = args.noisevar
# lengthscale = args.lengthscale
nquery = args.numqueries
nrun = args.numruns
nhyp = args.numhyps

deterministic = args.deterministic
mode = args.mode

nmax = args.nmax # == nfunc
nfeature = args.nfeature

nstoiter = args.nsto
nysample = args.nysample
parallel_iterations = args.nparal

ntrain = args.ntrain
n_initial_training_x = args.ninit
func_name = args.function

print("nrun: {}".format(nrun))
print("nquery: {}".format(nquery))
print("nhyp: {}".format(nhyp))
print("nmax (nfunc): {}".format(nmax))
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
import utils_for_continuous
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


duplicate_resolution=0.001
print("duplicate_resolution = {} to avoid degenerate matrix if 2 testing points are too close to each other".format(duplicate_resolution))


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
        
        opt_fsample_maxima = required_placeholders['opt_fsample_maxima']
        invKs = required_placeholders['invKs']

        vals = evaluate_mes.mes(xs, 
            xdim, nmax, nhyp, 
            Xsamples, Ysamples, 
            ls, sigmas, sigma0s, 
            opt_fsample_maxima, invKs, 
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

        invKs = required_placeholders['invKs']

        vals = evaluate_ucb.ucb(xs, 
                xdim, nhyp, 
                Xsamples, Ysamples, 
                ls, sigmas, sigma0s, 
                invKs, 
                required_placeholders['iteration'],
                dtype=dtype)

    elif criterion == 'pes':
        nmax = crit_params['nmax']

        max_observed_y = required_placeholders['max_observed_y']
        invKs = required_placeholders['invKs']
        invKmaxsams = required_placeholders['invKmaxsams']
        opt_fsample_maximizers = required_placeholders['opt_fsample_maximizers']

        vals = evaluate_pes.pes(xs, 
                xdim, nmax, nhyp, 
                Xsamples, Ysamples, 
                ls, sigmas, sigma0s,
                opt_fsample_maximizers, 
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


def get_required_placeholders(criterion, crit_params,
                dtype,
                is_debug_mode = False):
    nhyp = crit_params['nhyp']
    xdim = crit_params['xdim']
    nmax = crit_params['nmax']
    nfeature = crit_params['nfeature']

    parallel_iterations = crit_params['parallel_iterations']

    X_plc = tf.placeholder(dtype=dtype, shape=(None, xdim), name='X_plc')
    Y_plc = tf.placeholder(dtype=dtype, shape=(None, 1), name='Y_plc')

    max_observed_y_plc = tf.placeholder(dtype=dtype, shape=(), 
        name = 'max_observed_y_plc')

    invKs_plc = tf.placeholder(shape=(nhyp,None,None), dtype=dtype, name='invKs_plc')
    # (nhyp, nobs, nobs)
    opt_fsample_maxima_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None), name='opt_fsample_maximizers_plc')
    # (nhyp,nmax)
    opt_fsample_maximizers_plc = tf.placeholder(dtype=dtype, shape=(nhyp, None, xdim), name='opt_fsample_maxima_plc')
    # (nhyp,nmax,xdim)

    opt_meanf_maximizer_plc = tf.placeholder(dtype = dtype, shape=(1,xdim), name = 'opt_meanf_maximizer_plc')
    # assuming same for all nhyp

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

    opt_meanf_candidate_xs_plc = tf.placeholder(dtype = dtype, 
                            shape = (None,xdim),
                            name = 'opt_meanf_candidate_xs')
    opt_fsample_candidate_xs_plc = tf.placeholder(dtype = dtype,
                            shape = (None,xdim),
                            name = 'opt_fsample_candidate_xs')
    opt_crit_candidate_xs_plc = tf.placeholder(dtype = dtype,
                            shape = (None,xdim),
                            name = 'opt_crit_candidate_xs')
    
    thetas_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, 1),
        name = 'thetas')
    Ws_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, xdim),
        name = 'Ws')
    bs_plc = tf.placeholder(dtype = dtype,
        shape = (nhyp, nmax, nfeature, 1),
        name = 'bs')

    iteration_plc = tf.placeholder(dtype=dtype, shape=(), name='iteration_plc')

    required_placeholders = {
        'iteration': iteration_plc,
        'X': X_plc,
        'Y': Y_plc,
        'opt_meanf_candidate_xs': opt_meanf_candidate_xs_plc,
        'opt_fsample_candidate_xs': opt_fsample_candidate_xs_plc,
        'opt_crit_candidate_xs': opt_crit_candidate_xs_plc,
        'invKs': invKs_plc,

        'thetas': thetas_plc,
        'Ws': Ws_plc,
        'bs': bs_plc,

        'opt_fsample_maximizers': opt_fsample_maximizers_plc,
        'opt_fsample_maxima': opt_fsample_maxima_plc,


        'opt_meanf_maximizer': opt_meanf_maximizer_plc
    }

    if criterion == 'mes':
        required_placeholders['opt_fsample_maxima'] = opt_fsample_maxima_plc
    
    elif criterion == 'ei':
        required_placeholders['max_observed_y'] = max_observed_y_plc
    
    elif criterion == 'ucb':
        pass
    
    elif criterion == 'pes':
        invKmaxsams_plc = tf.placeholder(dtype=dtype, 
                shape=(nhyp, None, None, None), # nhyp, nmax, nobs+1, nobs+1
                name='invKmaxsams_plc')

        required_placeholders['opt_fsample_maximizer'] = opt_fsample_maximizers_plc
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

    return required_placeholders


def get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls, sigmas, sigma0s,
            dtype,
            is_debug_mode=False):
    xdim = crit_params['xdim']
    nhyp = crit_params['nhyp']
    nmax = crit_params['nmax']
    nfeature = crit_params['nfeature']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']
    opt_meanf_top_init_k = crit_params['opt_meanf_top_init_k']
    opt_fsample_top_init_k = crit_params['opt_fsample_top_init_k']
    opt_crit_top_init_k = crit_params['opt_crit_top_init_k']

    X_plc = required_placeholders['X']
    Y_plc = required_placeholders['Y']

    intermediate_tensors = {}

    max_observed_y = tf.reduce_max(Y_plc)
    intermediate_tensors['max_observed_y'] = max_observed_y

    invKs = utils.precomputeInvKs(xdim, nhyp, 
                ls, sigmas, sigma0s, 
                X_plc, dtype)
    # nhyp x nobs x nobs
    intermediate_tensors['invKs'] = invKs


    # optimize mean function
    opt_meanf_func = lambda x: utils.compute_mean_f(
                                tf.reshape(x, shape=(-1,xdim)),
                                xdim, nhyp,
                                X_plc, Y_plc,
                                ls, sigmas, sigma0s,
                                required_placeholders['invKs'],
                                dtype=dtype)

    opt_meanf_assign, opt_meanf_train, opt_meanf_maximizer, opt_meanf_maximum \
        = utils_for_continuous.optimize_continuous_function(xdim, 
                opt_meanf_func,
                required_placeholders['opt_meanf_candidate_xs'],
                opt_meanf_top_init_k,
                parallel_iterations=parallel_iterations,
                xmin = xmin,
                xmax = xmax,
                dtype= dtype,
                name = 'opt_meanf')
    
    intermediate_tensors['opt_meanf_assign'] = opt_meanf_assign
    intermediate_tensors['opt_meanf_train'] = opt_meanf_train
    intermediate_tensors['opt_meanf_maximizer'] = opt_meanf_maximizer
    intermediate_tensors['opt_meanf_maximum'] = opt_meanf_maximum


    # sample function
    # move the computation of thetas, Ws, bs
    # to numpy instead of tensorflow 
    # to avoid numerical error (return None of tf.linalg.eigh)
    thetas_all, Ws_all, bs_all = utils_for_continuous.sample_function(
                        xdim, nhyp, nmax, nfeature,
                        ls, sigmas, sigma0s,
                        X_plc, Y_plc,
                        dtype=dtype)
    intermediate_tensors['thetas'] = thetas_all
    intermediate_tensors['Ws'] = Ws_all
    intermediate_tensors['bs'] = bs_all


    # optimize function samples
    opt_fsample_assigns, opt_fsample_trains, \
    opt_fsample_maximizers, opt_fsample_maxima \
        = utils_for_continuous.sample_xmaxs_fmaxs(
                xdim, nhyp, nmax, nfeature,
                ls, sigmas, sigma0s, 

                required_placeholders['thetas'],
                required_placeholders['Ws'],
                required_placeholders['bs'],

                required_placeholders['opt_fsample_candidate_xs'],
                opt_fsample_top_init_k,

                xmin, xmax,
                dtype=dtype,
                parallel_iterations=parallel_iterations,
                name='sample_xmaxs_fmaxs')

    intermediate_tensors['opt_fsample_assigns'] = opt_fsample_assigns
    intermediate_tensors['opt_fsample_trains'] = opt_fsample_trains
    intermediate_tensors['opt_fsample_maximizers'] = opt_fsample_maximizers
    intermediate_tensors['opt_fsample_maxima'] = opt_fsample_maxima


    # optimize acquisition function
    opt_fsample_maximizers = required_placeholders['opt_fsample_maximizers'] # (nhyp, nmax, xdim)
    opt_fsample_maxima = required_placeholders['opt_fsample_maxima'] # (nhyp, nmax)

    if criterion == 'mes':
        pass

    elif criterion == 'ei':
        pass

    elif criterion == 'ucb':
        pass

    elif criterion == 'pes':
        invKmaxsams = utils.eval_invKmaxsams(xdim, nhyp, nmax, 
                            ls, sigmas, sigma0s, 
                            required_placeholders['X'], 
                            required_placeholders['opt_fsample_maximizer'], 
                            dtype)
        intermediate_tensors['invKmaxsams'] = invKmaxsams

    elif criterion in ['ftl', 'sftl']:
        mean_tests_given_data = [] # nhyp, ntest
        cov_tests_given_data = [] # nhyp, ntest, ntest

        for i in range(nhyp):
            mean_test_i, cov_test_i = utils.compute_mean_var_f(required_placeholders['test_xs'],
                    X_plc, Y_plc,
                    ls[i,...], sigmas[i,...], sigma0s[i,...],
                    fullcov=True,
                    dtype=dtype)

            mean_tests_given_data.append(tf.squeeze(mean_test_i))
            cov_tests_given_data.append(cov_test_i)

        mean_tests_given_data = tf.stack(mean_tests_given_data)
        # (nhyp, ntest)
        cov_tests_given_data = tf.stack(cov_tests_given_data)
        # (nhyp, ntest, ntest)

        intermediate_tensors['mean_test_given_data'] = mean_tests_given_data
        intermediate_tensors['cov_test_given_data'] = cov_tests_given_data

        _, invpNKs = evaluate_mp.get_pNK_test_obs(
                        ls, sigmas, sigma0s, 
                        nhyp,
                        X_plc,
                        required_placeholders['test_xs'],
                        dtype=dtype)

        intermediate_tensors['invpNKs'] = invpNKs


    opt_crit_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    1,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=dtype)

    opt_crit_multiple_func = lambda x: evaluate_criterion(
                    tf.reshape(x, shape=(-1,xdim)), 
                    opt_crit_top_init_k,
                    criterion, crit_params,
                    ls, sigmas, sigma0s,
                    required_placeholders,
                    dtype=dtype)

    # opt_crit_func = lambda x: utils_for_continuous.func_with_2d_input(opt_crit_func, x, xdim)

    opt_crit_assign, opt_crit_train, \
    opt_crit_maximizer, opt_crit_maximum \
        = utils_for_continuous.optimize_continuous_function(
        xdim, opt_crit_func, 
        required_placeholders['opt_crit_candidate_xs'],
        opt_crit_top_init_k,
        parallel_iterations = parallel_iterations,
        xmin = xmin,
        xmax = xmax,
        dtype = dtype,
        name = 'optimize_crit',
        multiple_func = opt_crit_multiple_func)

    intermediate_tensors['opt_crit_assign'] = opt_crit_assign
    intermediate_tensors['opt_crit_train'] = opt_crit_train
    intermediate_tensors['opt_crit_maximizer'] = opt_crit_maximizer
    intermediate_tensors['opt_crit_maximum'] = opt_crit_maximum

    return intermediate_tensors



def get_placeholder_values(sess, 
        criterion, crit_params,
        required_placeholders,
        intermediate_tensors,

        ls, sigmas, sigma0s,
        X_np, Y_np,
        candidate_xs,

        iteration=1,
        dtype=tf.float32,
        is_debug_mode=False):
    xdim = crit_params['xdim']
    nmax = crit_params['nmax']
    ntrain = crit_params['ntrain']
    nfeature = crit_params['nfeature']
    xmin = crit_params['xmin']
    xmax = crit_params['xmax']

    values = {
            'iteration': iteration,
            'query_x': None}

    if 'max_observed_y' in intermediate_tensors:
        max_observed_y_np = sess.run(
            intermediate_tensors['max_observed_y'],
            feed_dict = { required_placeholders['Y']: Y_np }
        )

        values['max_observed_y'] = max_observed_y_np

    if 'invKs' in intermediate_tensors:
        invKs_np = sess.run(intermediate_tensors['invKs'], 
            feed_dict = {
                required_placeholders['X']: X_np
            })
        values['invKs'] = invKs_np
    

    # Optimize for best guess
    if candidate_xs['opt_meanf'] is None:
        rand_candidate_xs_np = np.linspace(xmin, xmax, 100).reshape(100,1)
        #np.random.rand(100, xdim) * (xmax - xmin) + xmin

        opt_meanf_candidate_xs_np = np.concatenate([rand_candidate_xs_np, X_np], axis=0)
    else:
        opt_meanf_candidate_xs_np = candidate_xs['opt_meanf']

    sess.run(intermediate_tensors['opt_meanf_assign'],
        feed_dict = {
            required_placeholders['opt_meanf_candidate_xs']: opt_meanf_candidate_xs_np,
            required_placeholders['X']: X_np,
            required_placeholders['Y']: Y_np,
            required_placeholders['invKs']: values['invKs']
        })

    for _ in range(crit_params['ntrain']):
        sess.run(intermediate_tensors['opt_meanf_train'],
            feed_dict = {
                required_placeholders['X']: X_np,
                required_placeholders['Y']: Y_np,
                required_placeholders['invKs']: values['invKs']
            })

    opt_meanf_maximizer_np, opt_meanf_maximum_np \
        = sess.run([
                intermediate_tensors['opt_meanf_maximizer'],
                intermediate_tensors['opt_meanf_maximum'] ],
            feed_dict = {
                required_placeholders['X']: X_np,
                required_placeholders['Y']: Y_np,
                required_placeholders['invKs']: values['invKs']
            })

    values['opt_meanf_maximizer'] = opt_meanf_maximizer_np
    values['opt_meanf_maximum'] = opt_meanf_maximum_np

    
    if 'opt_fsample_maximizer' in intermediate_tensors or 'opt_fsample_maxima' in intermediate_tensors:

        if use_GPU_for_sample_functions:
            # sample functions
            # move the computation of thetas, Ws, bs
            # to numpy instead of tensorflow 
            # to avoid numerical error (return None of tf.linalg.eigh)
            print("use GPU to sample functions.")
            thetas_np, Ws_np, bs_np = sess.run(
                [ intermediate_tensors['thetas'],
                intermediate_tensors['Ws'],
                intermediate_tensors['bs'] ],
                feed_dict = {
                    required_placeholders['X']: X_np,
                    required_placeholders['Y']: Y_np
                })

        else:
            print("use CPU to sample functions.")
            thetas_np = np.zeros([nhyp, nmax, nfeature, 1])
            Ws_np = np.zeros([nhyp, nmax, nfeature, xdim])
            bs_np = np.zeros([nhyp, nmax, nfeature, 1])

            for hyp_idx in range(nhyp):
                thetas_np[hyp_idx,...], Ws_np[hyp_idx,...], bs_np[hyp_idx,...] \
                    = optfunc.draw_random_init_weights_features_np(
                        xdim, nmax, nfeature,
                        X_np, Y_np,
                        ls[hyp_idx], sigmas[hyp_idx], sigma0s[hyp_idx])

        # optimize functions
        # assign initial values
        
        if candidate_xs['opt_fsample'] is None:
            print("generate opt_fsample_candidate_xs by combining Xsamples_np with 100 equi-distant points")

            rand_candidate_xs_np = np.linspace(xmin, xmax, 100).reshape(100,xdim)
            #np.random.rand(100, xdim) * (xmax - xmin) + xmin

            opt_fsample_candidate_xs_np = np.concatenate([rand_candidate_xs_np, Xsamples_np], axis=0)
        else:
            opt_fsample_candidate_xs_np = candidate_xs['opt_fsample']
        
        sess.run(intermediate_tensors['opt_fsample_assigns'],
            feed_dict = {
                required_placeholders['thetas']: thetas_np,
                required_placeholders['Ws']: Ws_np,
                required_placeholders['bs']: bs_np,
                required_placeholders['opt_fsample_candidate_xs']: opt_fsample_candidate_xs_np
            })
            
        for xx in range(ntrain):
            sess.run(intermediate_tensors['opt_fsample_trains'],
                feed_dict = {
                    required_placeholders['thetas']: thetas_np,
                    required_placeholders['Ws']: Ws_np,
                    required_placeholders['bs']: bs_np
                })
      
        opt_fsample_maximizers_np, opt_fsample_maxima_np \
            = sess.run([
                    intermediate_tensors['opt_fsample_maximizers'],
                    intermediate_tensors['opt_fsample_maxima'] ],
                feed_dict = {
                        required_placeholders['thetas']: thetas_np,
                        required_placeholders['Ws']: Ws_np,
                        required_placeholders['bs']: bs_np
                })

        values['opt_fsample_maximizers'] = opt_fsample_maximizers_np
        values['opt_fsample_maxima'] = opt_fsample_maxima_np

    # optimize acquisition function
    if criterion == 'pes':
        invKmaxsams_np = sess.run(
                intermediate_tensors['invKmaxsams'],
                feed_dict = {
                    required_placeholders['X']: X_np,
                    required_placeholders['Y']: Y_np,
                    required_placeholders['opt_fsample_maximizers']: opt_fsample_maximizers_np
                })
        values['invKmaxsams'] = invKmaxsams_np

    elif criterion in ['ftl', 'sftl']:
        unique_opt_fsample_maximizers_np = utils.remove_duplicates_np(
                            opt_fsample_maximizers_np[0,...], 
                            resolution=duplicate_resolution)

        print("Set test_xs = opt_fsample_maximizers unique by resolution = {}".format(duplicate_resolution))
        print(unique_opt_fsample_maximizers_np)
        test_xs_np = unique_opt_fsample_maximizers_np.reshape(-1,xdim)

        ntest = test_xs_np.shape[0]

        if ntest == 1:
            # there is only 1 unique opt_fsample_maximizer
            values['test_xs'] = test_xs_np
            values['max_probs'] = np.ones(ntest)
            values['max_probs'] /= np.sum(values['max_probs'])
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
                    nsample = crit_params['ntestsample'],
                    n_min_sample = crit_params['n_min_sample'])

        """
        depending on crit_params['mode']
            post_test_v0 and post_test_v1 are interpreted differently
        crit_params['mode']:
            'sample': post_test_sample, post_test_masks
            'ep', 'empirical': post_mean_tests, post_cov_tests
        """

        if test_xs_np.shape[0] == 1:
            values['test_xs'] = test_xs_np
            values['max_probs'] = max_probs_np
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
candidate_xs_to_optimize_np = f_info['xs']

xdim = f_info['xdim']

true_l = f_info['RBF.lengthscale']
true_sigma = f_info['RBF.variance']
true_sigma0 = f_info['noise.variance']
true_maximum = f_info['maximum']

ls_np = true_l.reshape(-1,xdim).astype(nptype)
sigmas_np = np.array([ true_sigma ], dtype=nptype)
sigma0s_np = np.array([ true_sigma0 ], dtype=nptype)

seed = 1


print("True GP hyperparameters: l:{} sigma:{} sigma0(noise var):{}".format(true_l, true_sigma, true_sigma0))
print("nhyp: {}".format(nhyp))
print("nrun:{}, nqueries:{}".format(nrun, nquery))
print("____________________________________________")



ls_toload = tf.get_variable(dtype=dtype, shape=(nhyp,xdim), name='ls')
sigmas_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigmas') 
sigma0s_toload = tf.get_variable(dtype=dtype, shape=(nhyp,), name='sigma0s')


# if DEBUG:
#     xs_plot_np = np.linspace(np.min(xs_to_optimize_np), np.max(xs_to_optimize_np), 100).reshape(-1,1)

if criterion == 'ftl':
    if mode not in ['empirical', 'ep']:
        raise Exception("Unknown mode ({}) for criterion {}".format(mode, criterion))
elif criterion == 'sftl':
    if mode != 'sample':
        raise Exception("Unknown mode ({}) for criterion {}".format(mode, criterion)) 


crit_params = {'nhyp': nhyp,
               'xdim': xdim,
               'nmax': nmax,
               'nfeature': nfeature,

               'xmin': xmin,
               'xmax': xmax,

               'mode': mode,
               'ntestsample': 1000,

               'nysample': nysample,
               'deterministic': deterministic,
               'nstoiter': nstoiter,
               'parallel_iterations': parallel_iterations,
               
               'opt_meanf_top_init_k': 5,
               'opt_fsample_top_init_k': 3,
               'opt_crit_top_init_k': 5,
               
               'ntrain': ntrain,
               'n_min_sample': 2}

print("crit_params: {}".format(crit_params))


required_placeholder_keys = {
    'mes': ['X', 'Y', 'opt_fsample_maxima', 'invKs'],
    'ei': ['X', 'Y', 'max_observed_y', 'invKs'],
    'ucb': ['X', 'Y', 'invKs', 'iteration'],
    'pes': ['X', 'Y', 'invKs', 
            'invKmaxsams', 'opt_fsample_maximizers', 'max_observed_y'],
    'ftl': ['X', 'Y', 'test_xs', 'max_probs',
            'post_mean_tests', 'post_cov_tests', 
            'invKs', 'invpNKs'],
    'sftl': ['X', 'Y', 'test_xs', 'max_probs',
            'post_test_samples', 'post_test_masks', 
            'invpNKs']
}


required_placeholders = get_required_placeholders(criterion, crit_params, dtype, is_debug_mode=False)

intermediate_tensors = get_intermediate_tensors(criterion, crit_params,
            required_placeholders,
            ls_toload, sigmas_toload, sigma0s_toload,
            dtype,
            is_debug_mode=False)



all_guess_xx = np.zeros([nrun, nquery+1, xdim])
all_guesses = np.zeros([nrun, nquery+1])

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

        Xsamples_np = np.random.rand(n_initial_training_x,xdim) * (xmax - xmin) + xmin
        # print("Xsamples_np.shape = ", Xsamples_np.shape); sys.stdout.flush()

        Fsamples_np = f(Xsamples_np).reshape(-1,1).astype(nptype)
        Ysamples_np = (Fsamples_np + np.random.randn(Xsamples_np.shape[0],1) * np.sqrt(true_sigma0)).astype(nptype)
        # print("done"); sys.stdout.flush()

        print("")

        mean_f_const = 0.0
        min_npoint_opt_hyper = 12
        opt_hyp_every = 3
        last_opt_hyp_iter = -100

        for nq in range(nquery):

            startime_query = time.time()

            # for randomly drawing different functions
            sess.run(tf.global_variables_initializer())

            print("")
            print("{}:{}.=================".format(nr, nq))
            print("  X: {}".format(Xsamples_np.T))
            print("  Y: {}".format(Ysamples_np.T))

            if Xsamples_np.shape[0] < min_npoint_opt_hyper:
                pass
            elif (nq - last_opt_hyp_iter) > opt_hyp_every:
                last_opt_hyp_iter = nq

                # optimize for the GP hyperparameters
                #     and the mean function: f(x) = a
                mean_f_const1, sigmas_np1, ls_np1, sigma0s_np1 \
                    = functions.get_gphyp_gpy(Xsamples_np, Ysamples_np, 
                                noise_var=true_sigma0, 
                                train_noise_var=False, 
                                max_iters=500)

                if np.all(ls_np1 < 1e3):
                    mean_f_const = mean_f_const1
                    sigmas_np = np.array(sigmas_np1).reshape(nhyp)
                    ls_np = np.array(ls_np1).reshape(nhyp,xdim)
                    sigma0s_np = np.array(sigma0s_np1).reshape(nhyp)

                    print("")
                    print("==============")
                    print("Updated GP hyperparameters and mean_const:")
                    print("  mean_f_const: {}".format(mean_f_const))
                    print("  signal varia: {}".format(sigmas_np))
                    print("  lengthscale : {}".format(ls_np))
                    print("  noise varian: {} ({})".format(sigma0s_np, true_sigma0))
                    print("==============")
                else:
                    print("")
                    print("==============")
                    print("Skip updating hyperparameters and mean_cost due to large learned lscale: {}".format(ls_np1))
                    print("==============")

            ls_toload.load(ls_np, sess)
            sigmas_toload.load(sigmas_np, sess)
            sigma0s_toload.load(sigma0s_np, sess)
            print("")

            candidate_xs = {
                'opt_meanf': candidate_xs_to_optimize_np,
                'opt_fsample': candidate_xs_to_optimize_np,
                'opt_crit': None
            }

            while True:
                # repeat if query_x is nan
                placeholder_values = get_placeholder_values(sess,
                            criterion, crit_params,
                            required_placeholders,
                            intermediate_tensors,
                            
                            ls_np, sigmas_np, sigma0s_np,
                            Xsamples_np, Ysamples_np - mean_f_const,
                            candidate_xs,
                            iteration=nq+1,
                            dtype=dtype,
                            is_debug_mode=False)

                print("meanf maximizes at {} ({})".format(
                            placeholder_values['opt_meanf_maximizer'], 
                            placeholder_values['opt_meanf_maximum']))
                all_guess_xx[nr,nq,:] = placeholder_values['opt_meanf_maximizer'].squeeze()
                all_guesses[nr,nq] = f(placeholder_values['opt_meanf_maximizer'].reshape(1,xdim)).squeeze()

                print("GUESS: {} (f={})".format(placeholder_values['opt_meanf_maximizer'].squeeze(), 
                                            all_guesses[nr,nq]))
                print("all guesses: {}".format(all_guesses[nr,:(nq+1)]))
                print("")


                if placeholder_values['query_x'] is not None:
                    # no need to optimize the criterion
                    # as there is only 1 maximizer sample 
                    # (for pes, ftl-related)
                    opt_crit_maximizer_np = placeholder_values['query_x']
                    opt_crit_maximum_np = None
                
                else:

                    feed_dict = {
                        required_placeholders['X']: Xsamples_np,
                        required_placeholders['Y']: Ysamples_np - mean_f_const
                    }

                    for key in required_placeholder_keys[criterion]:
                        if key not in ['X', 'Y']:
                            feed_dict[ required_placeholders[key] ] = placeholder_values[key]
                    

                    if candidate_xs['opt_crit'] is None:
                        if criterion in ['pes', 'ftl', 'sftl']:
                            opt_crit_candidate_xs_np = placeholder_values['opt_fsample_maximizers'][0,...]
                        else:
                            x_np_idxs = np.array(list(range(Xsamples_np.shape[0])))
                            np.random.shuffle(x_np_idxs)
                            opt_crit_candidate_xs_np = Xsamples_np[x_np_idxs[:100]].reshape(-1,xdim)
                    else:
                        opt_crit_candidate_xs_np = candidate_xs['opt_crit']
                    
                    feed_dict[ required_placeholders['opt_crit_candidate_xs'] ] = opt_crit_candidate_xs_np

                    
                    sess.run(intermediate_tensors['opt_crit_assign'],
                        feed_dict = feed_dict)

                    for _ in range(crit_params['ntrain']):
                        sess.run(intermediate_tensors['opt_crit_train'],
                            feed_dict = feed_dict)
                    
                    opt_crit_maximizer_np, opt_crit_maximum_np \
                        = sess.run([
                        intermediate_tensors['opt_crit_maximizer'],
                        intermediate_tensors['opt_crit_maximum'] ],
                        feed_dict = feed_dict)

                query_x = opt_crit_maximizer_np.reshape(-1,xdim)
                query_f = f(query_x).reshape(-1,1) 
                query_y = query_f + np.random.randn(query_x.shape[0],1) * np.sqrt(true_sigma0)


                if 'opt_fsample_maximizers' in placeholder_values:
                    print("maximizer samples: {}".format(
                        placeholder_values['opt_fsample_maximizers']))
                
                if 'test_xs' in placeholder_values:
                    print("test candidate: {}".format(
                        placeholder_values['test_xs']))
                    print("max probs: {}".format(placeholder_values['max_probs']))
                print("QUERY: {}".format(query_x))

                print("end query in {:.4f}s".format(time.time() - startime_query))
                sys.stdout.flush()

                if not np.any(np.isnan(query_x)):
                    break
                else:
                    print("Repeat finding QUERY due to NAN!")

            Xsamples_np = np.concatenate([Xsamples_np, query_x], axis=0)
            Fsamples_np = np.concatenate([Fsamples_np, query_f], axis=0)
            Ysamples_np = np.concatenate([Ysamples_np, query_y], axis=0)



        candidate_xs = {
            'opt_meanf': candidate_xs_to_optimize_np,
            'opt_fsample': candidate_xs_to_optimize_np,
            'opt_crit': None
        }

        placeholder_values = get_placeholder_values(sess,
                    criterion, crit_params,
                    required_placeholders,
                    intermediate_tensors,
                    
                    ls_np, sigmas_np, sigma0s_np,
                    Xsamples_np, Ysamples_np - mean_f_const,
                    candidate_xs,
                    iteration=nquery,
                    dtype=dtype,
                    is_debug_mode=False)

        print("meanf maximizes at {} ({})".format(
                    placeholder_values['opt_meanf_maximizer'], 
                    placeholder_values['opt_meanf_maximum']))

        all_guess_xx[nr,nquery,:] = placeholder_values['opt_meanf_maximizer'].squeeze()
        all_guesses[nr,nquery] = f(placeholder_values['opt_meanf_maximizer'].reshape(1,xdim)).squeeze()

        print("GUESS: {} (f={})".format(placeholder_values['opt_meanf_maximizer'].squeeze(), 
                                        all_guesses[nr,nquery]))
        print("all guesses: {}".format(all_guesses[nr,:]))
        print("")

        all_xx[nr,...] = Xsamples_np
        all_ff[nr,...] = Fsamples_np.squeeze()
        all_yy[nr,...] = Ysamples_np.squeeze()

        np.save('{}/{}_xx.npy'.format(folder, criterion), all_xx)
        np.save('{}/{}_ff.npy'.format(folder, criterion), all_ff)
        np.save('{}/{}_yy.npy'.format(folder, criterion), all_yy)
        np.save('{}/{}_guess_ff.npy'.format(folder, criterion), all_guesses)
        np.save('{}/{}_guess_xx.npy'.format(folder, criterion), all_guess_xx)









