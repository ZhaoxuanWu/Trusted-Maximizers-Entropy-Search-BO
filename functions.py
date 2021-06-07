import numpy as np 
import scipy.optimize as scopt

import importlib
import utils
import pickle

# FOR FACE_ATTACK
from face_attack.face_attack_fn import fr_br_distance_func
from face_attack.utils.paths import ATTACK_DIR
import os

# FOR CNN_CIFAR_10
import math
import time
import numpy as np
import tensorflow as tf
import random
import keras
from keras.datasets import cifar10

"""
meanf_candidate_xs: maximizer, and a random subset of size 100 of observations
fsample_candidate_xs: maximizer,  and a random subset of size 100 of observations
crit_candidate_xs: sample maximizers
or
    use 'xs' as candidate_xs for both meanf_candidate_xs and fsample_candidate_xs
    crit_candidate_xs: sample maximizers
"""

"""
to show:
    batchsize = 1:
    vs. EI, PES, MES, UCB
        synthetic functions: 1d, 2d
        branin
        goldstein
        hartmann3d

        log10P

    batchsize > 1:
    vs. GP-UCB-PE, GP-B-UCB, qEI(TODO)
        synthetic functions: 1d, 2d
        branin
        goldstein
        hartmann3d

        hartmann4d

        log10P
"""

"""
branin: 100
goldstein: 120
hartmann3d: 140
hartmann4d: 140
stylbinski: 110
    (numerical error: should run ~100iter)

rerun
ackley 100
egg_holder

ignore
hartmann6d
shubert
"""

def maximize_func(xdim, f, xs, xmin, xmax):

    negf = lambda x: -f(x.reshape(-1,xdim))
    fs = negf(xs).reshape(-1,)

    x0 = xs[np.argmax(-fs)].reshape(-1,xdim)

    res = scopt.minimize(fun=negf, 
                x0=x0, 
                method='L-BFGS-B', 
                bounds=[(xmin, xmax)]*xdim)

    maximum = -res.fun 
    maximizer = res.x.squeeze()

    res = scopt.minimize(fun=f, 
                x0=x0, 
                method='L-BFGS-B', 
                bounds=[(xmin, xmax)]*xdim)

    minimum = res.fun
    minimizer = res.x.squeeze()

    print("maximum value: {} at {}".format(maximum, maximizer))
    print("minimum value: {} at {}".format(minimum, minimizer))
    
    return maximizer, maximum, minimizer,minimum


def get_gphyp_gpy(X, Y, noise_var=None, train_noise_var=True, max_iters=500):
    # use gpflow to get the hyperparameters for the function

    try:
        import GPy
    except:
        raise Exception("Requires gpflow!")

    xdim = X.shape[1]

    kernel = GPy.kern.RBF(input_dim=xdim, variance=1., lengthscale=np.ones(xdim), ARD=True)
    meanf = GPy.mappings.Constant(input_dim=xdim, output_dim=1, value=0.0)

    if train_noise_var:
        m = GPy.models.GPRegression(X, Y, kernel=kernel, mean_function=meanf)
    elif noise_var is not None:
        m = GPy.models.GPRegression(X, Y, kernel=kernel, mean_function=meanf, noise_var=noise_var)
        m.Gaussian_noise.variance.fix() # unfix()
    else:
        raise Exception("functions.py get_gphyp_gpy:Require noise variance!")

    m.optimize(max_iters=max_iters)

    gpy_lscale = m.rbf.lengthscale.values
    gpy_signal_var = m.rbf.variance.values
    lscale = 1.0 / (gpy_lscale * gpy_lscale)
    mean_f_const = m.constmap.C.values

    # print("Mean: {}".format(mean_f_const))
    # print("Kernel: sigvar {}, lscale {}".format(gpy_signal_var, lscale))
    # print("Gaussian_noise variance: {}".format(m.Gaussian_noise.variance.values))

    return mean_f_const, gpy_signal_var, lscale, m.Gaussian_noise.variance


def get_meshgrid(xmin, xmax, nx, xdim):
    x1d = np.linspace(xmin, xmax, nx)
    vals = [x1d] * xdim
    xds = np.meshgrid(*vals)

    xs = np.concatenate([xd.reshape(-1,1) for xd in xds], axis=1)
    return xs


# for MATLAB
def get_info(func_name):
    f_info = globals()[func_name]()

    return f_info['xdim'], f_info['xmin'], f_info['xmax'], f_info['xs'],\
        f_info['noise.variance'], f_info['RBF.variance'], f_info['RBF.lengthscale'], \
        f_info['maximizer'], f_info['maximum']


def call_func(x, func_name, log_noise_std):
    f_info = globals()[func_name]()#func_1d_4modes()
    f = f_info['function']
    xdim = f_info['xdim']

    x = np.array(x).reshape(-1,xdim)
    n = x.shape[0]
    return f(x).squeeze() + np.random.randn(n) * np.exp(log_noise_std)


def func_gp_prior(xdim, l, sigma, seed):
    np.random.seed(seed)

    n_feats = 10000
    l = np.ones([1,xdim]) * l
    W = np.random.randn(n_feats, xdim) * np.tile( np.sqrt(l), (n_feats,1) )
    b = 2. * np.pi * np.random.rand(n_feats,1)
    theta = np.random.randn(n_feats,1)

    def f(x):
        x = x.reshape(-1,xdim)
        return ( theta.T.dot( np.sqrt(2. * sigma / n_feats) ).dot( np.cos(W.dot(x.T)
                    + np.tile(b, (1,x.shape[0])) )) ).squeeze()

    return f


def func_1d_8modes():
    xdim = 1
    xmin = 0.
    xmax = 10.
    seed = 1
    l = 10.0
    sigma = 2.0

    xs = np.linspace(xmin, xmax, 400).reshape(-1,1)
    
    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': sigma, # 2.1892,
            'RBF.lengthscale': np.array([l]), # 9.49,
            'maximizer': 0.796431278,
            'maximum': 1.95724434}
    

def func_1d_4modes():
    xdim = 1
    xmin = 0.
    xmax = 10.
    seed = 1
    l = 1.0
    sigma = 2.0

    xs = np.linspace(xmin, xmax, 400).reshape(-1,1)
    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': sigma, # 3.5855
            'RBF.lengthscale': np.array([l]), # 0.8121
            'maximizer': 2.518537,
            'maximum': 1.95724434}
    

def func_2d_smallls():
    xdim = 2
    xmin = 0.
    xmax = 10.
    seed = 1
    l = 4.0
    sigma = 2.0

    xs = get_meshgrid(xmin, xmax, 70, xdim)

    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': sigma,
            'RBF.lengthscale': np.array([l]*xdim),
            'maximizer': np.array([1.812655, 8.882240]),
            'maximum': 4.243544687}
    

def func_2d_largels():
    xdim = 2
    xmin = 0.
    xmax = 10.
    seed = 1
    l = 1.0
    sigma = 2.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)

    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': sigma,
            'RBF.lengthscale': np.array([l] * xdim),
            'maximizer': np.array([10., 3.02511199]),
            'maximum': 3.500}


def func_3d_largels():
    xdim = 3
    xmin = 0.
    xmax = 5.
    seed = 1
    l = 1.0
    sigma = 2.0

    xs = get_meshgrid(xmin, xmax, 10, xdim)

    f = func_gp_prior(xdim, l, sigma, seed)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': sigma,
            'RBF.lengthscale': np.array([l] * xdim),
            'maximizer': np.array([2.09414372, 0., 1.86934563]),
            'maximum': 3.354097574}


def log10K():
    xdim = 2
    xmin = 0.
    xmax = 1.

    name = 'log10K'
    X = np.loadtxt('pHdata/X_{}.txt'.format(name))
    Y = np.loadtxt('pHdata/Y_{}.txt'.format(name))
    hypers = np.loadtxt('pHdata/hyperparameters_{}.txt'.format(name))

    sigma = hypers[0]
    lengthscales = hypers[1:3]
    sigma0 = hypers[3]

    xs = get_meshgrid(xmin, xmax, 20, xdim)

    def f(x):
        x = x.reshape(-1,xdim)

        vals = utils.compute_mean_f_np(x, X, Y, lengthscales, sigma, sigma0)
        return vals

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': sigma0,
            'RBF.variance': sigma,
            'RBF.lengthscale': lengthscales,
            'maximizer': np.array([0.43437075, 0.68692362]),
            'maximum': 0.33465834}


def log10P():
    xdim = 2
    xmin = 0.
    xmax = 1.

    name = 'log10P'
    X = np.loadtxt('pHdata/X_{}.txt'.format(name))
    Y = np.loadtxt('pHdata/Y_{}.txt'.format(name))
    hypers = np.loadtxt('pHdata/hyperparameters_{}.txt'.format(name))

    sigma = hypers[0]
    lengthscales = hypers[1:3]
    sigma0 = hypers[3]

    xs = get_meshgrid(xmin, xmax, 20, xdim)

    def f(x):
        x = x.reshape(-1,xdim)

        vals = utils.compute_mean_f_np(x, X, Y, lengthscales, sigma, sigma0)
        return vals

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': sigma0,
            'RBF.variance': sigma,
            'RBF.lengthscale': lengthscales,
            'maximizer': np.array([0.68527863, 0.06831491]),
            'maximum': 1.09202265}


def pH():
    xdim = 2
    xmin = 0.
    xmax = 1.

    name = 'pH'
    X = np.loadtxt('pHdata/X_{}.txt'.format(name))
    Y = np.loadtxt('pHdata/Y_{}.txt'.format(name))
    hypers = np.loadtxt('pHdata/hyperparameters_{}.txt'.format(name))

    sigma = hypers[0]
    lengthscales = hypers[1:3]
    sigma0 = hypers[3]

    xs = get_meshgrid(xmin, xmax, 20, xdim)

    def f(x):
        x = x.reshape(-1,xdim)

        vals = utils.compute_mean_f_np(x, X, Y, lengthscales, sigma, sigma0)
        return vals

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': sigma0,
            'RBF.variance': sigma,
            'RBF.lengthscale': lengthscales,
            'maximizer': np.array([0.18600983, 0.17848398]),
            'maximum': 0.73425606}


def negative_hartmann3d():
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 3
    xmin = 0.
    xmax = 1.
    # maximum = 3.86277979
    # minimum = 0.00027354

    xs = get_meshgrid(xmin, xmax, 10, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    A = np.array([
            [3., 10., 30.],
            [0.1, 10., 35.],
            [3., 10., 30.],
            [0.1, 10., 35.]
        ])

    alpha = np.array([1., 1.2, 3., 3.2])

    P = 1e-4 * np.array([
            [3689., 1170., 2673.],
            [4699., 4387., 7470.],
            [1091., 8732., 5547.],
            [381., 5743., 8828.]
        ])


    def f(x):
        x = np.tile(x.reshape(-1,1,xdim), reps=(1,4,1))
        val = np.sum(alpha * np.exp(- np.sum(A * (x - P)**2, axis=2)), axis=1)
        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 0.5222,
            'RBF.lengthscale': np.array([2.0945, 10.4904, 32.5504]),
            'maximizer': np.array([0.11458923, 0.55564889, 0.85254695]),
            'maximum': 3.86277979}


def negative_hartmann4d():
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 4
    xmin = 0.
    xmax = 1.
    # maximum = 3.13449414
    # minimum = -1.30954062

    xs = get_meshgrid(xmin, xmax, 10, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    A = np.array([
            [10., 3., 17., 3.5, 1.7, 8.],
            [0.05, 10., 17., 0.1, 8., 14.],
            [3., 3.5, 1.7, 10., 17., 8.],
            [17., 8., 0.05, 10., 0.1, 14.]
        ])
    A = A[:,:4]

    alpha = np.array([1., 1.2, 3., 3.2])

    P = 1e-4 * np.array([
            [1312., 1696., 5569., 124., 8283., 5886.],
            [2329., 4135., 8307., 3736., 1004., 9991.],
            [2348., 1451., 3522., 2883., 3047., 6650.],
            [4047., 8828., 8732., 5743., 1091., 381.]
        ])
    P = P[:,:4]

    def f(x):
        x = np.tile(x.reshape(-1,1,xdim), reps=(1,4,1))
        val = -(1.1 - np.sum(alpha * np.exp(- np.sum(A * (x-P)**2, axis=2)), axis=1)) / 0.839
        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val
        # return np.sum(alpha * np.exp(- np.sum(A * (x - P)**2, axis=2)), axis=1)


    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 0.5985,
            'RBF.lengthscale': np.array([14.2876, 8.2371, 8.0386, 9.7550]),
            'maximizer': np.array([0.1873952, 0.19415135, 0.55791794, 0.26477959]),
            'maximum': 3.13449414}


def negative_rescaled_hartmann6d():
    # xdim = 3
    # range: (0,1) for all dimensions
    # global maximum: -3.86278 at (0.114614, 0.555649, 0.852547)
    xdim = 6
    xmin = 0.
    xmax = 1.
    # maximum = 3.13449414
    # minimum = -1.30954062

    xs = get_meshgrid(xmin, xmax, 5, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    A = np.array([
            [10., 3., 17., 3.5, 1.7, 8.],
            [0.05, 10., 17., 0.1, 8., 14.],
            [3., 3.5, 1.7, 10., 17., 8.],
            [17., 8., 0.05, 10., 0.1, 14.]
        ])

    alpha = np.array([1., 1.2, 3., 3.2])

    P = 1e-4 * np.array([
            [1312., 1696., 5569., 124., 8283., 5886.],
            [2329., 4135., 8307., 3736., 1004., 9991.],
            [2348., 1451., 3522., 2883., 3047., 6650.],
            [4047., 8828., 8732., 5743., 1091., 381.]
        ])

    def f(x):
        x = np.tile(x.reshape(-1,1,xdim), reps=(1,4,1))
        val = (2.58 + np.sum(alpha * np.exp(- np.sum(A * (x-P)**2, axis=2)), axis=1)) / 1.94
        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val


    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 1.423, #0.3446,
            'RBF.lengthscale': np.array([6.9512, 1.9341, 0.506, 4.2067, 5.0986, 3.5949]), # np.array([6.1398, 2.3368, 0.7698, 6.6535, 5.1594, 5.3718]),
            'maximizer': np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]),
            'maximum': 3.32237}


def negative_Branin():
    xdim = 2
    xmin = 0.
    xmax = 1.
    # maximum = 1.04739389
    # minimum = -4.87620974

    xs = get_meshgrid(xmin, xmax, 20, xdim)
    # xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1,xdim)
        x = 15. * x - np.array([5., 0.])

        val = -1.0 / 51.95 * (
            (x[:,1] - 5.1 * x[:,0]**2 / (4*np.pi**2) + 5.*x[:,0] / np.pi - 6.)**2
            + (10. - 10. / (8.*np.pi)) * np.cos(x[:,0])
            - 44.81
        )

        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val


    return {'function': f,
                'xdim': xdim,
                'xmin': xmin,
                'xmax': xmax,
                'xs': xs,
                'noise.variance': 0.0001,
                'RBF.variance': 0.589,
                'RBF.lengthscale': np.array([25.5866, 6.3735]),
                'maximizer': np.array([0.96165196, 0.16499956]),
                'maximum': 1.04739389}


# signal variance is too big
def negative_egg_holder():
    # xdim = 2
    # range: [-512,512] for all dimensions
    # global maximum: 959.6407 at (512, 404.2319)
    xdim = 2
    xmin = -1.0 # -512.0
    xmax = 1.0 # 512.0
    maximum = 959.64066272
    minimum = -1049.1316235

    # xs = get_meshgrid(xmin, xmax, 70, xdim)
    xs = np.random.rand(1000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1,xdim)
        val = (
            (x[:,1] * 512. + 47.) * np.sin(np.sqrt( np.abs(x[:,1] * 512. + 0.5*x[:,0] * 512. + 47.) ))
            + x[:,0] * 512. * np.sin(np.sqrt(np.abs(x[:,0] * 512. - (x[:,1] * 512. + 47.))))
            )
        val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 0.1299,
            'RBF.lengthscale': np.array([260.8678, 341.01061]),
            'maximizer': np.array([1., 0.78951524]),
            'maximum': 1.0}


def negative_Goldstein():
    xdim = 2
    xmin = 0.
    xmax = 1.
    # maximum = 2.18038839
    # minimum = -0.33341016

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    def f(x):
        x = x.reshape(-1,xdim)
        xb = x * 4. - 2. 

        val = - (
            np.log(
                (
                    1 
                    + (xb[:,0] + xb[:,1] + 1.)**2
                    * (19 - 14 * x[:,0] + 3 * x[:,0]**2 - 14 * x[:,1] + 6 * x[:,0] * x[:,1] + 3 * x[:,1]**2)
                )
                * (
                    30 
                    + (2 * x[:,0] - 3 * x[:,1])**2 
                    * (18 - 32 * x[:,0] + 12 * x[:,0]**2 + 48 * x[:,1] - 36 * x[:,0] * x[:,1] + 27 * x[:,1]**2)
                )
                ) - 8.693
            ) / 2.427

        # val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val

        
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance':0.3193,
            'RBF.lengthscale': np.array([73.5594, 74.9745]),
            'maximizer': np.array([0.45, 0.30]),
            'maximum': 2.180388}


def negative_shubert():
    # constraint to [-2.,2.]
    xdim = 2
    xmin = 0.
    xmax = 1

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    arr = np.array(list(range(1,6))).astype(float).reshape(-1,1)

    def f(x):
        x = x.reshape(-1,xdim)
        xb = x * 4. - 2. 
        
        val = - np.sum(arr * np.cos( (arr + 1.0) * xb[:,0] + arr), axis=0) \
            * np.sum(arr * np.cos( (arr + 1.0) * xb[:,1] + arr), axis=0)
        return val

        
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance':3.7443,
            'RBF.lengthscale': np.array([35.88, 28.3715]),
            'maximizer': np.array([0.3, 0.143718]),
            'maximum': 186.73091}


def negative_stylbinski():
    # constraint to [-5.,5.]
    xdim = 2
    xmin = 0.
    xmax = 1
    maximum = 78.33233141
    minimum = -200.0

    xs = get_meshgrid(xmin, xmax, 50, xdim)
    # xs = np.random.rand(2000, xdim) * (xmax - xmin) + xmin

    arr = np.array(list(range(1,6))).astype(float).reshape(-1,1)

    def f(x):
        x = x.reshape(-1,xdim)
        xb = x * 10. - 5. 
        
        val = -0.5 * np.sum(xb**4 - 16 * xb**2 + 5 * xb, axis=1)
        val = (val - minimum) / (maximum - minimum) * 2.0 - 1.0
        return val

        
    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance':8.8434e5,
            'RBF.lengthscale': np.array([0.2449, 0.2474]),
            'maximizer': np.array([0.20964659, 0.20964659]),
            'maximum': 1.0}


def face_attack():
    from joblib import Parallel, delayed
    xdim = 8
    xmin = -0.2
    xmax = 0.2

    #xs = get_meshgrid(xmin, xmax, 5, xdim)
    xs = np.random.rand(100000, xdim) * (xmax - xmin) + xmin

    ENCODINGS = os.path.join(ATTACK_DIR, 'fr_gallery41_encodings.npy')
    LABELS = os.path.join(ATTACK_DIR, 'fr_gallery41_me_encodings.npy')
    g_img_encodings, g_img_labels = np.load(ENCODINGS), np.load(LABELS)

    def f(x):
        # GLOBAL_BOUNDS = [(-.15, .2), (-.15, .2), (-.2, .15), (-.2, .15), (-.2, .15), (-.15, .2), (-.15, .2), (-.15, .15)]
        lower_bounds = [-.15, -.15, -.2, -.2, -.2, -.15, -.15, -.15]
        ranges = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3]
        x = x.reshape(-1, xdim)

        # Linear mapping to desired bounds
        x = (x + 0.2) / 0.4 * ranges + lower_bounds

        # JOBLIB PARALLEL
        results = np.array([Parallel(n_jobs=5)(
            delayed(fr_br_distance_func)(i, g_img_encodings, g_img_labels, debug=True, show_img=False) for i in x)])
        # results = np.apply_along_axis(fr_br_distance_func, 1, x, g_img_encodings, g_img_labels, debug=True, show_img=False)

        return -1 * results
        # return fr_br_distance_func(x, g_img_encodings, g_img_labels, debug=True, show_img=False)

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 1,
            'RBF.lengthscale': np.array([1, 1, 1, 1, 1, 1, 1, 1]),
            'maximizer': np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            'maximum': 0
            }


def set_global_determinism(seed_value=0, fast_n_close=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        fast_n_close (bool): whether to achieve efficient at the cost of determinism/reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    tf.set_random_seed(seed_value)
    np.random.seed(seed_value)
    if fast_n_close:
        return

    print("*******************************************************************************")
    print("*** set_global_determinism is called,setting full determinism, will be slow ***")
    print("*******************************************************************************")

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    from tfdeterminism import patch
    patch()


class CNN_CIFAR_10:

    def __init__(self, xdim, xmin, xmax, xs, seed_value=0):
        self.xdim = xdim
        self.xmin = xmin
        self.xmax = xmax
        self.xs = xs
        self.seed_value = seed_value

        self.range = {"batch_size": [16, 512],
                      "learning_rate": [1e-7, 1e-1],
                      "learning_rate_decay": [1e-7, 1e-3],
                      "l2_regular": [1e-7, 1e-3],
                      "conv_filters": [6, 256],
                      "dense_units": [128, 2048],
                      "dropout_rate": [0, 0.75]}

        self.params = {"batch_size": 32,
                       "learning_rate": 1e-7,
                       "learning_rate_decay": 1e-7,
                       "l2_regular": 1e-7,
                       "conv_filters": 128,
                       "dense_units": 256,
                       "dropout_rate": 0.25}

        #### load the CIFAR-10 dataset
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train/255
        self.x_test = x_test/255
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes)

        # self.graph = self.build_graph(seed_value=seed_value)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

    def convert(self, param_val, param_type):
        converted_val = param_val * (self.range[param_type][1] - self.range[param_type][0]) + self.range[param_type][0]

        if param_type in ["batch_size", "conv_filters", "dense_units"]:
            converted_val = int(converted_val)

        return converted_val

    def build_graph(self, seed_value=0):
        ### The tensorflow model of CNN is built below
        _IMAGE_SIZE = 32
        _IMAGE_CHANNELS = 3
        _NUM_CLASSES = 10

        graph = tf.Graph()
        set_global_determinism(seed_value=seed_value)

        with graph.as_default():
            with tf.name_scope('main_params'):
                x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='Input')
                y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
                x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

                global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
                learning_rate_placeholder = tf.placeholder(tf.float32, shape=[], name='learning_rate')

            with tf.variable_scope('conv1') as scope:
                conv = tf.layers.conv2d(
                    inputs=x_image,
                    filters=self.params['conv_filters'],
                    kernel_size=[3, 3],
                    padding='SAME',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=seed_value),
                    bias_initializer='zero',
                    kernel_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                    bias_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                    activity_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                )
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')

                conv = tf.layers.conv2d(
                    inputs=pool,
                    filters=self.params['conv_filters'],
                    kernel_size=[3, 3],
                    padding='SAME',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=seed_value),
                    bias_initializer='zero',
                    kernel_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                    bias_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                    activity_regularizer=tf.keras.regularizers.l2(self.params['l2_regular']),
                )
                pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
                drop = tf.layers.dropout(pool, rate=self.params['dropout_rate'], name=scope.name)

            with tf.variable_scope('fully_connected') as scope:
                flat = tf.reshape(drop, [-1, 8 * 8 * self.params['conv_filters']])

                fc = tf.layers.dense(inputs=flat, units=self.params['dense_units'], activation=tf.nn.relu,
                                     kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=seed_value),
                                     bias_initializer='zero'
                                     )
                drop = tf.layers.dropout(fc, rate=self.params['dropout_rate'], name=scope.name)
                softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, name=scope.name,
                                          kernel_initializer=tf.compat.v1.glorot_uniform_initializer(seed=seed_value),
                                          bias_initializer='zero'
                                          )

            self.y_pred_cls = tf.argmax(softmax, axis=1, name='y_pred_cls')

            # LOSS AND OPTIMIZER
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder,
                                               beta1=0.9,
                                               beta2=0.999,
                                               epsilon=1e-08).minimize(self.loss,
                                                                       global_step=global_step,
                                                                       name='optimizer')

            # PREDICTION AND ACCURACY CALCULATION
            correct_prediction = tf.equal(self.y_pred_cls, tf.argmax(y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Initialize the variables (i.e. assign their default value)
            self.init = tf.global_variables_initializer()

        return graph

    def evaluate(self, param, N=20):
        '''
            param: parameters
            N: the maximum number of epochs
        '''

        training_epochs = N

        param_types = ["batch_size", "learning_rate", "learning_rate_decay", "l2_regular", "conv_filters",
                       "dense_units", "dropout_rate"]
        for i in range(len(param_types)):
            self.params[param_types[i]] = self.convert(param[i], param_types[i])

        print(self.params)

        graph = self.build_graph()

        with tf.Session(graph=graph, config=self.config) as sess:
            # Run the initializer
            sess.run(self.init)

            val_epochs = []

            for epoch in range(training_epochs):
                total_batch = int(math.ceil(len(self.x_train) / self.params['batch_size']))

                lr = self.params['learning_rate'] * (1. / (1. + self.params['learning_rate_decay'] * epoch))

                for s in range(total_batch):
                    batch_xs = self.x_train[s * self.params['batch_size']: (s + 1) * self.params['batch_size']]
                    batch_ys = self.y_train[s * self.params['batch_size']: (s + 1) * self.params['batch_size']]

                    start_time = time.time()
                    i_global, _, batch_loss, batch_acc = sess.run(
                        ['main_params/global_step', 'optimizer', self.loss, self.accuracy],
                        feed_dict={'main_params/Input:0': batch_xs,
                                   'main_params/Output:0': batch_ys,
                                   'main_params/learning_rate:0': lr})

                i = 0
                predicted_class = np.zeros(shape=len(self.x_test), dtype=np.int)
                while i < len(self.x_test):
                    j = min(i + self.params['batch_size'], len(self.x_test))
                    batch_xs = self.x_test[i:j, :]
                    batch_ys = self.y_test[i:j, :]
                    predicted_class[i:j] = sess.run(
                        self.y_pred_cls,
                        feed_dict={'main_params/Input:0': batch_xs,
                                   'main_params/Output:0': batch_ys,
                                   'main_params/learning_rate:0': lr}
                    )
                    i = j

                correct = (np.argmax(self.y_test, axis=1) == predicted_class)
                acc = correct.mean()
                val_epochs.append(acc)
                # print("ACC: {}".format(acc))

        return val_epochs[-1]


def cnn_cifar_10():
    seed_value = 1
    xdim = 7
    xmin = 0
    xmax = 1
    xs = np.random.rand(100000, xdim) * (xmax - xmin) + xmin
    my_func = CNN_CIFAR_10(xdim=xdim, xmin=xmin, xmax=xmax, xs=xs, seed_value=seed_value)

    def f(x):
        x = x.reshape(-1, xdim)
        vals = np.zeros(x.shape[0])

        for i, param in enumerate(x):
            vals[i] = my_func.evaluate(param, N=10)

        return vals

    return {'function': f,
            'xdim': xdim,
            'xmin': xmin,
            'xmax': xmax,
            'xs': xs,
            'noise.variance': 0.0001,
            'RBF.variance': 1,
            'RBF.lengthscale': np.array([1, 1, 1, 1, 1, 1, 1]),
            'maximizer': np.array([0, 0, 0, 0, 0, 0, 0]),
            'maximum': 1,
            }
