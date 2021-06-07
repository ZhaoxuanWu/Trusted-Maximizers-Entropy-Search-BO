import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np 
import scipy as sp 
import scipy.stats as spst
import sys

import empirical_approximation


clip_min = 1e-100
print("clip_min = {}".format(clip_min))


def perturb(xs, duplicate_resolution, xmin, xmax):
    """
    xs: n x d
    if exist i,j dist(xs[i,:], xs[j,:]) <= duplicate_resolution
        perturb xs[i,:] such that dist(xs[i,:], xs[j,:]) > duplicate_resolution
            and xs[i,:] >= xmin and xs[i,:] <= xmax
    """
    n = xs.shape[0]
    d = xs.shape[1]

    is_duplicated = True

    while is_duplicated:

        is_duplicated = False 

        for i in range(n):
            for j in range(i+1,n):
                diff = xs[i,:] - xs[j,:]
                rms = np.sqrt(np.sum(diff * diff))

                while rms <= duplicate_resolution or np.any(xs[j,:] < xmin) or np.any(xs[j,:] > xmax):
                    # modify xs[j]
                    xs[j,:] = xs[i,:] + np.random.rand(1,d) * duplicate_resolution * 4.0 - duplicate_resolution * 2.0
                    
                    diff = xs[i,:] - xs[j,:]
                    rms = np.sqrt(np.sum(diff * diff))

                    is_duplicated = True

    return xs



def sqrtm(mat):
    # return tf.linalg.sqrtm(mat)
    # only valid for positive symmetric matrix
    s, u, _ = tf.svd(mat, full_matrices=True)
    return u * tf.sqrt(s) @ tf.transpose(u)


def get_uniform_random_vect(size, dim, xmin, xmax):
    xs = np.random.rand(size,dim) * (xmax - xmin) + xmin
    return xs

def sample_maxfd_maxfs_given_maxxs(
        X, Y,

        xdim, 
        nobs, nmaxloc,
        nmaxfd, # == n_maxfobs
        nmaxf, # == nmaxf_per_maxfobs

        maxlocs, # (nmaxloc, xdim)
        mean_maxfs, std_maxfs, # (nmaxloc,)
        post_data_f_mean, # (nobs,)
        sqrt_post_data_f_cov, # (nobs, nobs)

        dtype=tf.float32):
    """
    sample maxfd and maxfs given maxxs
    """

    obs_standard_norm = tfp.distributions.MultivariateNormalDiag(
                                loc        = tf.zeros(nobs, dtype=dtype),
                                scale_diag = tf.ones(nobs, dtype=dtype) )

    """
    1. a. sample f values at the observed inputs
        RETURN: shape = (nhyp, nmaxfd, n_obs)
        b. sample maximum f at observed inputs
        RETURN: shape = (nhyp, nmaxfd)
    """
    standardized_data_f_samples = obs_standard_norm.sample(sample_shape=(nmaxfd,))
    # shape = (nmaxfd, n_obs)

    data_f_samples = sqrt_post_data_f_cov @ tf.transpose(standardized_data_f_samples)\
                + tf.reshape(post_data_f_mean, shape=(nobs,1))
    # shape = (n_obs, nmaxfd)

    return_data_maxfs = tf.reduce_max(data_f_samples,axis=0)
    # shape = (nmaxfd,)

    data_maxfs = tf.reshape( return_data_maxfs, shape=(1,nmaxfd))

    """
    2. sample max_fs for each max_x in maxlocs
        RETURN: shape = (nhyp, nmaxloc, nmaxfd, nmaxf_per_maxfobs)
    """
    # (2) Sample fmax for each xmax in maxlocs
    mean_maxfs = tf.reshape(mean_maxfs, shape = (nmaxloc,1))
    std_maxfs  = tf.reshape(std_maxfs, shape = (nmaxloc,1))
    
    # truncated normal with mean_fmaxs_i, var_fmaxs_i
    #     lower bound by: data_maxfs
    #     mean_fmaxs_i.shape = (nmaxloc,1)
    #     data_maxfs.shape   = (1, nmaxfd)
    #     sample_shape       = nmaxf_per_maxfobs
    #     return shape = (nmaxf, nmaxloc, nmaxfd)
    max_fs = tfp.distributions.TruncatedNormal(
                loc   = mean_maxfs, 
                scale = std_maxfs, 
                low   = data_maxfs, 
                high  = np.infty, 
                allow_nan_stats=False ).sample( nmaxf )
    # shape = (nmaxf, nmaxloc, nmaxfd)

    max_fs = tf.transpose(max_fs, perm=[1,2,0])
    # shape = (nmaxloc, nmaxfd, nmaxf)

    return max_fs, return_data_maxfs
    # shape = (nmaxloc, nmaxfd, nmaxf)
    # shape = (nmaxfd,)


# def sample_maxfd_maxfs_given_maxxs(
#         X, Y,

#         xdim, 
#         nobs, nhyp, nmaxloc,
#         n_maxfobs, nmaxf_per_maxfobs,

#         maxlocs_all,
#         mean_maxfs, var_maxfs,
#         post_data_f_means, sqrt_post_data_f_covs,

#         dtype=tf.float32):
#     """
#     sample maxfd and maxfs given maxxs
#     """

#     obs_standard_norm = tfp.distributions.MultivariateNormalDiag(
#                                 loc        = tf.zeros(nobs, dtype=dtype),
#                                 scale_diag = tf.ones(nobs, dtype=dtype) )
#     maxfs_all = []
#     data_maxfs_all = []

#     for i in range(nhyp):
#         maxlocs = maxlocs_all[i,...]

#         post_data_f_mean = post_data_f_means[i,...]
#         sqrt_post_data_f_cov = sqrt_post_data_f_covs[i,...]

#         """
#         1. a. sample f values at the observed inputs
#             RETURN: shape = (nhyp, n_maxfobs, n_obs)
#            b. sample maximum f at observed inputs
#             RETURN: shape = (nhyp, n_maxfobs)
#         """
#         standardized_data_f_samples = obs_standard_norm.sample(sample_shape=n_maxfobs)
#         # shape = (n_maxfobs, n_obs)

#         data_f_samples = sqrt_post_data_f_cov @ tf.transpose(standardized_data_f_samples)\
#                     + tf.reshape(post_data_f_mean, shape=(nobs,1))
#         # shape = (n_obs, n_maxfobs)

#         data_maxfs = tf.reduce_max(data_f_samples,axis=0)
#         # shape = (n_maxfobs,)
#         data_maxfs_all.append(data_maxfs)

#         data_maxfs = tf.reshape( data_maxfs, shape=(1,n_maxfobs))

#         """
#         2. sample max_fs for each max_x in maxlocs
#             RETURN: shape = (nhyp, nmaxloc, n_maxfobs, nmaxf_per_maxfobs)
#         """
#         # (2) Sample fmax for each xmax in maxlocs
#         mean_maxfs_i = tf.reshape(mean_maxfs[i,...], shape = (nmaxloc,1))
#         std_maxfs_i  = tf.reshape(tf.sqrt(var_maxfs[i,...]), shape = (nmaxloc,1))
        
#         # truncated normal with mean_fmaxs_i, var_fmaxs_i
#         #     lower bound by: data_maxfs
#         #     mean_fmaxs_i.shape = (nmaxloc,1)
#         #     data_maxfs.shape   = (1, n_maxfobs)
#         #     sample_shape       = nmaxf_per_maxfobs
#         #     return shape = (nmaxf_per_maxfobs, nmaxloc, n_maxfobs)
#         max_fs = tfp.distributions.TruncatedNormal(
#                     loc   = mean_maxfs_i, 
#                     scale = std_maxfs_i, 
#                     low   = data_maxfs, 
#                     high  = np.infty, 
#                     allow_nan_stats=False ).sample( nmaxf_per_maxfobs )

#         nmaxf_per_maxloc = nmaxf_per_maxfobs * n_maxfobs
#         # assert max_fs.shape == (nmaxf_per_maxfobs, nmaxloc, n_maxfobs)
#         max_fs = tf.transpose(max_fs, perm=[1,0,2])
#         max_fs = tf.reshape(max_fs, shape=(nmaxloc, nmaxf_per_maxloc))
#         # assert max_fs.shape == (nmaxloc, nmaxf_per_maxloc)
        
#         maxfs_all.append(max_fs)
    
#     return tf.stack(maxfs_all), tf.stack(data_maxfs_all)
#     # shape = (nhyp, nmaxloc, nmaxf_per_maxloc)
#     # shape = (nhyp, n_maxfobs)





# def sample_tnorm(n, upper, r=None, dtype=tf.float32):
#     # r: init r of size (n, shape(upper))
#     # upper is the value at which 
#     #   the standardized normal r.v. is truncated above

#     upper_mat = tf.expand_dims(upper, 0)
#     upper_mat = tf.tile(upper_mat, multiples=tf.concat([ tf.constant([n], dtype=tf.int32), tf.ones_like(tf.shape(upper), dtype=tf.int32)], axis=0))

#     if r is None:
#         r = tf.random.normal(shape=(n,), dtype=dtype)
#         r = tf.reshape(r, tf.concat([ tf.constant([n], dtype=tf.int32), tf.ones_like(tf.shape(upper), dtype=tf.int32) ], axis=0))
#         r = tf.tile(r, multiples=tf.concat([ tf.constant([1]), tf.shape(upper) ], axis=0))

#     rt = r

#     def cond(rt, upper_mat):
#         return tf.less( tf.reduce_sum(tf.cast(tf.less_equal(rt, upper_mat), dtype=tf.int32)), tf.size(rt) )

#     def body(rt, upper_mat):
#         return tf.where(tf.less_equal(rt, upper_mat), rt, tf.random.normal(shape=tf.shape(rt), dtype=dtype)), upper_mat

#     # cond = lambda rt, upper_mat : tf.less( tf.reduce_sum(tf.cast(tf.less_equal(rt, upper_mat), dtype=tf.int32)), tf.size(rt) )
#     # body = lambda rt, upper_mat : ( tf.where(tf.less_equal(rt, upper_mat), rt, tf.random.normal(shape=tf.shape(rt), dtype=dtype)), upper_mat )
#     loop_vars = (rt, upper_mat)

#     rt, upper_mat = tf.while_loop(
#                         cond = cond,
#                         body = body,
#                         loop_vars = loop_vars,
#                         parallel_iterations=1
#                     )

#     return rt


def sample_tnorm_plus_norm(
                mn, sn, mf, sf, 
                maxfs,
                n_sample_each_distribution, 
                dtype=tf.float32):
    """
    for each distribution defined in mn, sn, mf, sf, maxfs
    sample n_sample_each_distribution samples
    for mni, sni, mfi, sfi, maxfsi in (mn, sn, mf, sf, maxfs):
        distribution is sum of Normal(mni,sni) 
                    and TruncatedNormal(mfi, sfi, low=-infty, high=maxfsi)

    Requires:
    # maxfs: (..., 1)
    # mn: (..., nx)
    # sn: (..., nx)
    # mf: (..., nx)
    # sf: (..., nx)
    # `...' in maxfs, mf, sf should be the same
    #       but can be multiple dimensions
    # for example:
    #   maxfs.shape = (100,10,1)
    #   mn.shape    = (100,10,500)
    #   sn.shape    = (100,10,500)
    #   mf.shape    = (100,10,500)
    #   sf.shape    = (100,10,500)
    # IMPORTANT: sf is updated in this function to avoid zero standard deviation
    # hence we need to update it accordingly outside the function

    Returns:
        ysamples (shape =(n_sample_each_distribution,..., nx))
    """

    r0_low = -np.infty

    # to avoid division by zero in r0_high
    #   if sf is zero
    #       set mf and sf to some feasible values
    #   create masked_sf to set zero entropy for zero sf when returning the result

    r0_high = (maxfs - mf) / sf

    # epsilon = tf.cast(1e-6, dtype=dtype)
    # mask = tf.where(tf.greater(sf, epsilon), tf.ones_like(sf, dtype=dtype), tf.zeros_like(sf, dtype=dtype))
    # sf = tf.where(tf.greater(sf, epsilon), sf, epsilon * tf.ones_like(sf, dtype=dtype)) # to avoid division by zero in r0_high    
    # r0_high = (maxfs - mf) * mask / sf + tf.cast(1e6, dtype=dtype) * (tf.cast(1.0, dtype=dtype) - mask)

    r0_loc = tf.zeros_like(r0_high, dtype=dtype)
    r0_scale = tf.ones_like(r0_high, dtype=dtype)
    # shape = (n_max, nx)

    r0 = tfp.distributions.TruncatedNormal(
            loc   = r0_loc, 
            scale = r0_scale, 
            low   = r0_low, 
            high  = r0_high ).sample( n_sample_each_distribution )

    r0 = tf.reshape(r0, 
                    shape = tf.concat([ [n_sample_each_distribution], 
                                        tf.shape(r0_loc)], 
                                    axis=0))
    # shape = (n_sample_each_maxf, n_max, nx)


    r1 = tf.random.normal(shape = tf.concat([ [n_sample_each_distribution], 
                                            tf.shape(r0_loc)], 
                                            axis=0), 
                        dtype = dtype)
    # shape = (n_sample_each_maxf, n_max, nx)

    rt = r0 * sf + mf
    rn = r1 * sn + mn
    y_samples = rn + rt 
    # shape = (n_sample_each_maxf, n_max, nx)

    # IMPORTANT: sf is updated in this function to avoid zero standard deviation
    # hence we need to update it accordingly outside the function
    return y_samples

def get_submatrices_subvects(covmat, n, covmatdim):
    """
    covmat.shape = (...,n,n)
    covmatdim: number of dimensions in covmat
    return
        list of submatrices
        list of subvects
    """
    varf = tf.linalg.diag_part(covmat)
    # (...,n)

    mat_permute_order = list(range(max(covmatdim-2,0), covmatdim)) + list(range(covmatdim-2))
    covmat = tf.transpose(covmat, perm=mat_permute_order)

    vec_permute_order = list(range(max(covmatdim-2,0), covmatdim-1)) + list(range(covmatdim-2))

    subcovfs = []
    invsubcovfs = []
    kdxs = []
    varfs = []

    for i in range(n):
        idxs = list(range(i+1))
        xv, yv = np.meshgrid(idxs, idxs)
        idxs = list(zip(yv.reshape(-1,), xv.reshape(-1,)))
        idxs = np.array(idxs).reshape(i+1, i+1, 2)

        subcovi = tf.gather_nd(covmat, idxs)
        # shape=(i+1,i+1,...)

        subcovi = tf.transpose(subcovi, perm=mat_permute_order)
        # shape=(...,i+1,i+1)
        invsubcovi = tf.linalg.inv(subcovi)
        # shape=(...,i+1,i+1)


        subcovfs.append( subcovi )
        invsubcovfs.append( invsubcovi )

        varfi = tf.gather(varf, indices=i, axis=-1)
        # shape=(...)
        varfs.append(varfi)

        if i > 0:
            kdx = tf.gather(subcovi, i)
            # shape=(i+1,...)
            kdx = tf.gather(kdx, np.array(list(range(i)), dtype=int) )
            # shape=(i,...)
            kdx = tf.transpose(kdx, perm=vec_permute_order)
            # shape=(...,i)
            kdxs.append(kdx)

    return subcovfs, invsubcovfs, kdxs, varfs
    # list (n) with i: (...,i+1,i+1)
    # list (n) with i: (...,i+1,i+1)
    # list (n-1) with i: (...,i+1)
    # list (n) with i: (...) (can be scalar)


def compute_conditional_mean_from_submatrices(
                    meanf, # (..., n)
                    invsubcovfs, # list (n) with i: (...,i+1,i+1)
                    kdxs, # list (n-1) with i: (...,i+1)
                    varfs, # list (n) with i: (...) (can be scalar)
                    n, dim, 
                    samples, # (..., n, nsample)
                    nsample):
    meanfi = tf.gather(meanf, indices=0, axis=-1)
    # shape=(...)
    meanfi = tf.expand_dims( tf.expand_dims(meanfi, axis=-1), axis=-1 )
    # shape=(...,1,1)
    meanfi = tf.tile(meanfi, multiples=([1] * (dim-1) + [nsample]))
    conditional_meanf = [meanfi]

    # second dimension of covf onwards
    for i in range(1,n):
        kdx = tf.expand_dims(kdxs[i-1], axis=-1)
        # shape = (...,i,1)
        kxd = tf.expand_dims(kdxs[i-1], axis=-2)
        # shape = (...,1,i)

        meanfi_1 = tf.gather(meanf, indices=list(range(i)), axis=-1)
        # shape=(...,i)
        meanfi_1 = tf.expand_dims(meanfi_1, axis=-1)
        # shape=(...,i,1)

        meanfi = tf.gather(meanf, indices=i, axis=-1)
        # shape=(...)
        meanfi = tf.expand_dims( tf.expand_dims(meanfi, axis=-1), axis=-1 )
        # shape=(...,1,1)

        samples_i = tf.gather(samples, indices=list(range(i)), axis=dim-2)
        # shape=(...,i,nsample)

        meanfi = meanfi + kxd @ invsubcovfs[i-1] @ (samples_i - meanfi_1)
        # shape=(...,1,nsample)
        conditional_meanf.append(meanfi)

    conditional_meanf = tf.concat(conditional_meanf, axis=dim-2)
    # shape=(...,n,nsample)
    return conditional_meanf


def compute_conditional_std_from_submatrices(
                    invsubcovfs, # list (n) with i: (...,i+1,i+1)
                    kdxs, # list (n-1) with i: (...,i+1)
                    varfs, # list (n) with i: (...) (can be scalar)
                    n, dim):
    # first dimension of covf
    varfi = tf.expand_dims(varfs[0], axis=-1)
    varfi = tf.expand_dims(varfi, axis=-1)
    # shape=(...,1,1)
    conditional_stdf = [tf.sqrt(varfi)]

    # second dimension of covf onwards
    for i in range(1,n):
        varfi = tf.expand_dims(varfs[i], axis=-1)
        varfi = tf.expand_dims(varfi, axis=-1)
        # shape = (...,1,1)

        kdx = tf.expand_dims(kdxs[i-1], axis=-1)
        # shape = (...,i,1)
        kxd = tf.expand_dims(kdxs[i-1], axis=-2)
        # shape = (...,1,i)
        varfi = varfi - kxd @ invsubcovfs[i-1] @ kdx
        # shape = (...,1,1)
        conditional_stdf.append( tf.sqrt(varfi) )

    conditional_stdf = tf.concat(conditional_stdf, axis=dim-2)
    # shape=(...,n,1)
    conditional_stdf = tf.gather(conditional_stdf, indices=0, axis=dim-1)
    # shape=(...,n)

    return conditional_stdf


def compute_conditional_statistics(
                    meanf, # (..., n)
                    covf, # (..., n, n)
                    n, dim, 
                    samples, # (..., n, nsample)
                    nsample,
                    invsubcovfs=None,
                    kdxs=None,
                    varfs=None):
    """
    Requires:
        dim: number of dimensions in covf
        meanf: shape = (..., n)
        covf: shape = (..., n, n)
        samples: shape = (..., n, nsample)
        n is not a tensor
            but is a numpy integer
    Returns:
        conditional_meanf: shape (..., n, nsample)
        conditional_stdf: shape (..., n)
    
    f ~ N(meanf, covf) # n-variate Normal distribution
    f = (f1,...,fn)

    fi|f1^j,...,f{i-1}^j ~ N(conditional_meanf[...,j,i], conditional_stdf[...,j,i])
        (conditioned on the j-th sample, i.e., samples[...,:i,j])
    """
    # get covariance matrix for each conditional
    _, invsubcovfs, kdxs, varfs = get_submatrices_subvects(covf, n, dim)
    # subcovs list (n) with i: (...,i+1,i+1)
    # kdxs list (n-1) with i: (...,i+1)
    # varfs list (n) with i: (...) (can be scalar)

    conditional_means = compute_conditional_mean_from_submatrices(
                    meanf,
                    invsubcovfs, kdxs, varfs,
                    n, dim, 
                    samples, nsample)

    conditional_stds = compute_conditional_std_from_submatrices(
                    invsubcovfs, kdxs, varfs,
                    n, dim)


    return conditional_means, conditional_stds

 

def sample_tnorm_mv(
            meanfs, # (..., batchsize)
            conditional_stdfs, # (..., batchsize)
            maxfs, # (..., 1)
            invsubcovfs, # list (batchsize) i: (..., i+1,i+1)
            kdxs, # list (batchsize-1) i: (..., i+1)
            batchsize, ndim,
            nsample, dtype=tf.float32):
    """
    NOTE: shape of maxfs: (...,1)
        which means the uppers of all dimension in batch are the same!

    ndim: number of dimensions of meanfs = number of dimensions of conditional_stdfs

    truncated multivariate Gaussian
        by sampling sequentially truncated normal?
        or rejection sampling from multivariate normal?
            the first method is easier for derivative?

    Requires:
        meanfs: (..., batchsize)
        conditional_stdfs: (..., batchsize)
        maxfs: (..., 1)
        invsubcovfs: list with item i: (..., i+1,i+1)
        kdxs: list with item i: (..., i)
    Returns:
        (..., batchsize, nsample)
    """
    
    # Sample for 1st dimension
    f_samples = tfp.distributions.TruncatedNormal(
                    loc=meanfs[...,0], 
                    scale=conditional_stdfs[...,0], 
                    low=-np.infty, 
                    high=maxfs[...,0]).sample(nsample)
    # (nsample, ...)
    f_samples = tf.transpose(f_samples, perm=(list(range(1,ndim)) + [0]))
    # (..., nsample)
    f_samples = tf.expand_dims(f_samples, axis=-2)
    # (..., 1, nsample)

    # Sample for 2nd dimension onwards
    for i in range(1, batchsize):
        # get conditional mean given previous dimension samples
        meanfi = meanfs[...,i] # (...)
        meanfi = tf.expand_dims( 
                    tf.expand_dims(meanfi, axis=-1), 
                    axis=-1)
        # (...,1,1)

        meanf_beforei = tf.gather(meanfs, 
                            indices=list(range(i)), 
                            axis=-1) # (...,i)
        meanf_beforei = tf.expand_dims(meanf_beforei, axis=-1)
        # (...,i,1)

        kdxsi = tf.expand_dims(kdxs[i-1], axis=-2)
        # (...,1,i)

        cond_meanfi = meanfi + kdxsi @ invsubcovfs[i-1] @ (f_samples - meanf_beforei)
        # (..., 1, nsample)
        
        # sample i-th dimension sample
        f_samples_i = tfp.distributions.TruncatedNormal(
                    loc=cond_meanfi, # (..., 1, nsample) 
                    scale=tf.expand_dims(
                            tf.expand_dims(conditional_stdfs[...,i], 
                                           axis=-1),
                            axis=-1), # (..., 1, 1) 
                    low=-np.infty, 
                    high=tf.expand_dims(
                            tf.expand_dims(maxfs[...,0],
                                           axis=-1),
                            axis=-1), # (..., 1, 1)
                    ).sample(1)
        # (1,..., 1, nsample)
        f_samples_i = tf.reshape(f_samples_i, shape=tf.shape(f_samples_i)[1:])
        # (...,1, nsample)
        f_samples = tf.concat([f_samples, f_samples_i], axis=-2)

    return f_samples 
    # (..., batchsize, nsample)


def evaluate_logpdf_no_z_part(mt, st, m, s, u, dtype=tf.float32):
    # logpdf = evaluate_logpdf_z_part - evaluate_logpdf_no_z_part
    st2 = st*st
    s2 = s*s
    return tf.log( tf.cast(tf.sqrt(2*np.pi), dtype=dtype) 
                   * (tf.constant(1.0,dtype=dtype) + tf.erf( (u - mt) / (tf.cast(np.sqrt(2),dtype=dtype)*st) )) * tf.sqrt(st2 + s2) )


def evaluate_logpdf_z_part(z, mt, st, m, s, u, dtype=tf.float32):
    """
    z is the value of the random variable = sum of truncated norm and norm
    mt, st: mean, standard deviation of truncated normal rv
    u: upper bound of truncated normal rv
    m, s: mean, standard deviation of normal rv
    """
    # logpdf = evaluate_logpdf_z_part - evaluate_logpdf_no_z_part
    st2 = st*st
    s2 = s*s

    return ( -tf.square(mt + m - z) / (2*(st2 + s2))
             + tf.log(tf.constant(1.0, dtype=dtype) + tf.erf( ( (u-z+m)*st2 + (u-mt)*s2 ) 
                                  / (tf.cast(np.sqrt(2), dtype=dtype)*st*s*tf.sqrt(s2 + st2)) )) )


def evaluate_tnorm_plus_norm_entropy(mn, sn, mf, sf, maxf, n, dtype=tf.float32, debug=False):
    """
    N ~ Norm(mn, sn^2)
    F ~ Norm(mf, sf^2) truncated above at maxf
    Y = N + F
    return the entropy of Y
        init_rt: initial random samples of the standardized F
    """
    r0 = tfp.distributions.TruncatedNormal(loc=tf.zeros_like(mf, dtype=dtype), scale=tf.ones_like(sf, dtype=dtype), low=-np.infty, high= (maxf - mf)/sf ).sample( n )
    r0 = tf.reshape(r0, shape=tf.concat([tf.constant([n], dtype=tf.int32), tf.shape(mf)], axis=0))
    
    r1 = tf.random.normal(shape=tf.concat([tf.constant([n], dtype=tf.int32), tf.shape(mf)], axis=0), dtype=dtype)

    rt = r0 * sf + mf
    rn = r1 * sn + mn 
    sumr = rn + rt 

    logpdf_no_z_part = evaluate_logpdf_no_z_part(mf, sf, mn, sn, maxf, dtype=dtype)
    logpdf_z_part = evaluate_logpdf_z_part(sumr, mf, sf, mn, sn, maxf, dtype=dtype)

    entropy =  logpdf_no_z_part - tf.reduce_mean(logpdf_z_part, axis=0) # - average of logpdf

    if debug:
        return entropy, r0, r1, rt, rn, sumr, logpdf_no_z_part, logpdf_z_part
    return entropy


def evaluate_tnorm_entropy(mf, sf, maxf, dtype=tf.float32):
    maxf2 = (maxf - mf) / sf
    norm = tf.distributions.Normal(loc=tf.constant(0., dtype=dtype), scale=tf.constant(1., dtype=dtype))
    return tf.constant(0.5, dtype=dtype) + tf.log( tf.cast(tf.sqrt(2.0 * np.pi), dtype=dtype) * sf) + norm.log_cdf(maxf2) - maxf2 * norm.prob(maxf2) / (tf.constant(2.0, dtype=dtype) * norm.cdf(maxf2))


def evaluate_norm_entropy(s, dtype=tf.float32):
    return tf.cast(0.5 * tf.log(2*np.pi), dtype=dtype) + tf.log(s) + tf.constant(0.5, dtype=dtype)


def chol2inv(mat, dtype=tf.float32):

    n = tf.shape(mat)[0]

    # _, mat, _ = tf.while_loop(
    #     cond = lambda i, mat, eigs: tf.reduce_min(eigs) < 1e-9,
    #     body = lambda i, mat, eigs: [0.0, 
    #             mat + tf.eye(n, dtype=dtype) * 1e-4, 
    #             tf.linalg.eigvalsh(mat + tf.eye(n, dtype=dtype) * 1e-4)],
    #     loop_vars = (0.0, mat, tf.linalg.eigvalsh(mat)) )

    # mat = mat + tf.reduce_max(mat) * 1e-3 * tf.eye(n, dtype=dtype)

    # lower = tf.linalg.cholesky(mat)
    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.eye(n, dtype=dtype))
    invmat = tf.transpose(invlower) @ invlower
    return invmat


def multichol2inv(mat, n_mat, dtype=tf.float32):
    # lower = tf.linalg.cholesky(mat)
    # mat: (n_mat, m,m)

    # eye_mat = tf.tile(tf.expand_dims(tf.eye(tf.shape(mat)[1], dtype=dtype), 
    #                                                   axis=0), 
    #                                    multiples=(n_mat,1,1) )
    # # (n_mat, m, m)

    # max_vals = tf.expand_dims(
    #             tf.expand_dims(
    #                 tf.reduce_max(tf.reduce_max(mat, axis=-1), axis=-1),
    #                 axis=-1),
    #            axis=-1)
    # # (n_max,1,1)

    # mat = mat + eye_mat * 1e-3 *  max_vals

    invlower = tf.matrix_solve(tf.linalg.cholesky(mat), 
                               tf.tile(tf.expand_dims(tf.eye(tf.shape(mat)[1], dtype=dtype), 
                                                      axis=0), 
                                       multiples=(n_mat,1,1) ) )
    invmat = tf.matmul(invlower, invlower, transpose_a=True)
    return invmat    


def computeKnm(X, Xbar, l, sigma, dtype=tf.float32):
    """
    X: n x d
    l: d
    """
    n = tf.shape(X)[0]
    m = tf.shape(Xbar)[0]

    X = X * tf.sqrt(l)
    Xbar = Xbar * tf.sqrt(l)

    Q = tf.tile(tf.reduce_sum( tf.square(X), axis=1 , keepdims=True ), multiples=(1,m))
    Qbar = tf.tile(tf.transpose(tf.reduce_sum(tf.square(Xbar), axis=1, keepdims=True )), multiples=(n,1)) 

    dist = Qbar + Q - 2 * X @ tf.transpose(Xbar)
    knm = sigma * tf.exp( -0.5 * dist )
    return knm


def computeKmm_old(X, l, sigma, dtype=tf.float32):
    """
    X: n x d
    l: 1 x d
    sigma: signal variance
    sigma * exp( - 0.5 * (X - X)^2 * lengthscale)
    X' = X * a
    lengthscale' = lengthscale / a^2
    """
    n = tf.shape(X)[0]
    X = X * tf.sqrt(l)
    Q = tf.tile(tf.reduce_sum( tf.square(X), axis=1, keepdims=True ), multiples=(1,n))
    dist = Q + tf.transpose(Q) - 2 * X @ tf.transpose(X)

    kmm = sigma * tf.exp(-0.5 * dist)
    return kmm


def computeKmm(X, l, sigma, nd=2, dtype=tf.float32):
    """
    X: (...,n,d)
    nd = len(tf.shape(X))
    l: (1,d)
    sigma: signal variance
    return (...,n,n)
    """
    n = tf.shape(X)[-2]
    X = X * tf.sqrt( tf.reshape(l, shape=(1,-1)) )
    # (...,n,d)
    Q = tf.reduce_sum( tf.square(X), axis=-1, keepdims=True )
    # (...,n,1)

    transpose_idxs = np.array(list(range(nd)))
    transpose_idxs[-2] = nd-1
    transpose_idxs[-1] = nd-2

    dist = Q + tf.transpose(Q, perm=transpose_idxs) - 2 * X @ tf.transpose(X, perm=transpose_idxs)

    kmm = sigma * tf.exp(-0.5 * dist)

    return kmm


def computeNKmm(X, l, sigma, sigma0, dtype=tf.float32):
    """
    X: n x d
    l: 1 x d
    sigma: signal variance
    sigma0: noise variance
    """
    # cond = tf.less(sigma0, 1e-2)
    # tf.where(cond)
    """
    if sigma0 >= 1e-2:
        perturb = 1e-10
    else:
        perturb = 1e-4
    """
    # print("Add jitter for computeNKmm")
    # return computeKmm(X, l, sigma) + tf.eye(tf.shape(X)[0], dtype=dtype) * (sigma0 + sigma * tf.constant(1e-10, dtype=dtype))
    print("No jitter for computeNKmm")
    return computeKmm(X, l, sigma, dtype=dtype) + tf.eye(tf.shape(X)[0], dtype=dtype) * sigma0


def compute_mean_var_f(x, Xsamples, Ysamples, l, sigma, sigma0, 
                    NKInv=None, fullcov=False, dtype=tf.float32):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: m x 1
    Xsamples: m x d
    x: n x d

    return: mean: n x 1
            var : n x 1
    """
    if NKInv is None:
        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)
        NKInv = chol2inv( NK, dtype=dtype )

    kstar = computeKnm(x, Xsamples, l, sigma, dtype=dtype)
    mean = tf.squeeze(kstar @ (NKInv @ Ysamples))

    if fullcov:
        Kx = computeKmm(x, l, sigma, dtype=dtype)
        var = Kx - kstar @ NKInv @ tf.transpose(kstar)
        diag_var = tf.linalg.diag_part(var)
        diag_var = tf.clip_by_value(diag_var, clip_value_min=clip_min, clip_value_max=np.infty)
        var = tf.linalg.set_diag(var, diag_var)
    else:
        var = sigma - tf.reduce_sum( (kstar @ NKInv) * kstar, axis=1 )
        var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

    return mean, var


def computeKmm_np(X, l, sigma):
    n = X.shape[0]
    xdim = X.shape[1]
    
    l = l.reshape(1,xdim)

    X = X * np.sqrt(l)

    Q = np.tile(
        np.sum( X * X, axis=1, keepdims=True ),
        reps=(1,n)
    )
    dist = Q + Q.T - 2 * X.dot(X.T)

    kmm = sigma * np.exp(-0.5 * dist)
    return kmm 


def computeKnm_np(X, Xbar, l, sigma):
    """
    X: n x d
    l: d
    """
    n = np.shape(X)[0]
    m = np.shape(Xbar)[0]
    xdim = np.shape(X)[1]

    l = l.reshape(1,xdim)
    
    X = X * np.sqrt(l)
    Xbar = Xbar * np.sqrt(l)

    Q = np.tile( 
        np.sum( X*X, axis=1, keepdims=True),
        reps = (1,m))
    Qbar = np.tile(
        np.sum( Xbar*Xbar, axis=1, keepdims=True).T,
        reps=(n,1))

    dist = Qbar + Q - 2 * X.dot(Xbar.T)
    knm = sigma * np.exp(-0.5 * dist)
    return knm


def compute_mean_f_np(x, Xsamples, Ysamples, l, sigma, sigma0):
    """
    x: n x xdim
    Xsample: m x xdim
    Ysamples: m x 1
    return mean: n x 1

    l: 1 x xdim
    sigma, sigma0: scalar
    """
    m = Xsamples.shape[0]
    xdim = Xsamples.shape[1]
    x = x.reshape(-1,xdim)
    n = x.shape[0]

    Ysamples = Ysamples.reshape(m,1)

    NKmm = computeKmm_np(Xsamples, l, sigma) + np.eye(m) * sigma0
    invNKmm = np.linalg.inv(NKmm)

    kstar = computeKnm_np(x, Xsamples, l, sigma)
    mean = kstar.dot(invNKmm.dot(Ysamples))

    return mean.reshape(n,)


def computeNKmm_multiple_data(nxs, Xsamples, xs, l, sigma, sigma0, dtype=tf.float32, inverted=False):
    """
    xs: shape = (nxs,xdim)
    compute covariance matrix of [Xsamples, x] for x in xs
        where Xsamples include noise
              x does not include noise
    return shape (nxs, n_data+1, n_data+1)
        where n_data = tf.shape(Xsamples)[0]
    """
    n_data = tf.shape(Xsamples)[0]
    noise_mat = tf.eye(n_data, dtype=dtype) * sigma0
    noise_mat = tf.pad(noise_mat, [[0,1], [0,1]], "CONSTANT")

    ret = []
    for i in range(nxs):
        X_concat = tf.concat([Xsamples, tf.expand_dims(xs[i,:],0) ], axis=0)
        NKmm = computeKmm(X_concat, l, sigma, dtype=dtype) + noise_mat

        if inverted:
            invNKmm = chol2inv(NKmm, dtype=dtype)
            ret.append(invNKmm)
        else:
            ret.append(NKmm)

    return tf.stack(ret)



def compute_mean_var_f_multiple_data(n_xs, n_ys_per_x, x, Xsamples, Ysamples, xs, fs, 
                    l, sigma, sigma0, NKInvs=None, fullcov=False, dtype=tf.float32):
    """
    x: nx x d
        
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    sigma: scalar
    Xsamples: n x d
    Ysamples: n x 1

    fs: n_xs x n_ys_per_x
    xs: n_xs x d

    NKInvs: n_xs x (n+1) x (n+1)
            NOTE: the first n element contain noise variance
                  the last element doesn't contain noise variance

    return: mean: n_xs x n_ys_per_x x nx
            var:  n_xs x nx if fullcov == False
                  n_xs x nx x nx if fullcov == True
    """
    if fullcov:
        Kx = computeKmm(x, l, sigma, 
                    dtype=dtype, inverted=False)

    if NKInvs is None:
        NKInvs = computeNKmm_multiple_data(n_xs, Xsamples, xs, 
                        l, sigma, sigma0, dtype=dtype, inverted=True)

    var_all = []
    mean_all = []

    for i in range(n_xs):
        NKInv = NKInvs[i,...]
        X_concat = tf.concat([Xsamples, tf.expand_dims(xs[i,:],0)], axis=0)

        kstar = computeKnm(x, X_concat, l, sigma)

        if fullcov:
            var = Kx - kstar @ NKInv @ tf.transpose(kstar)
            diag_var = tf.linalg.diag_part(var)
            diag_var = tf.clip_by_value(diag_var, clip_value_min=clip_min, clip_value_max=np.infty)
            var = tf.linalg.set_diag(var, diag_var)
        else:
            var = sigma - tf.reduce_sum( (kstar @ NKInv) * kstar, axis=1 )
            var = tf.clip_by_value(var, clip_value_min=clip_min, clip_value_max=np.infty)

        var_all.append(var)

        mean_i= []
        for j in range(n_ys_per_x):            
            mean = tf.squeeze(kstar @ (NKInv @ 
                            tf.concat([Ysamples, tf.reshape(fs[i,j], 
                                      shape=(1,1)) ], axis=0) ))
            mean_i.append(mean)
        mean_all.append(tf.stack(mean_i))
    
    print("utils.compute_mean_var_f_multiple_data: clip value of var at {}!".format(clip_min))
    var_all = tf.stack(var_all)
    mean_all = tf.stack(mean_all)

    return mean_all, var_all



def compute_mean_f(x, 
        xdim, n_hyp, 
        Xsamples, Ysamples, 
        ls, sigmas, sigma0s, 
        NKInvs, 
        dtype=tf.float32):
    """
    NKsampleInv = inv(KXsampleInv + eye(n)*sigma0)
    l: 1 x d
    Ysamples: n x 1
    Xsamples: n x d
    x: 1 x d

    return: mean: n x 1
            var : n x 1
    """

    mean = tf.constant(0.0, dtype=dtype)

    for i in range(n_hyp):
        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]
        NKInv = NKInvs[i]

        kstar = computeKnm(x, Xsamples, l, sigma)
        mean = mean + tf.squeeze(kstar @ (NKInv @ Ysamples)) / tf.constant(n_hyp, dtype=dtype)
    return mean


# def find_top_k(func, xs, k):
#     # find top-k values of func(x) for x in xs
#     ys = tf.squeeze(func(xs))
#     _, idxs = tf.math.top_k(ys, k, sorted=False)
#     idxs = tf.reshape(idxs, [-1,])
#     return tf.gather(xs, idxs), tf.gather(ys, idxs)


def find_top_k(ys, k):
    _, idxs = tf.math.top_k(tf.squeeze(ys), k, sorted=False)
    idxs = tf.reshape(idxs, (-1,))
    return idxs


def get_initializers(func, ngroups, groups, n_inits):
    # get top n_inits[i] points from groups[i]
    # n_inits[i]: scalar
    # groups[i]: array of shape n x xdim
    # func(x) -> scalar
    # func is a function need maximization
    # requires: all n_inits > 0
    top_k_inits = []

    for i in range(ngroups):
        inits = utils.find_top_k(func, groups[i], n_inits[i])
        top_k_inits.append(inits)

    return tf.concat(top_k_inits, axis=0)


def merge_2dicts(d1, d2):
    d = d1.copy()
    d.update(d2)
    return d


# def find_top_ks(sess, func, feed_dict, input_key, init_groups, ks, batchsize):
def find_top_ks(func, init_groups, ks, batchsize):
    """
    init_groups: list of lists of inputs
    ks: list of k values
    batchsize: batchsize for evaluating func at 1 sess.run
    return: concatenation of top ks[i] of init_groups[i] for all i
    func = lambda x: sess.run(f, feed_dict=merge_2dicts(train_dict, {'initializers': x}))
    # need to test
    """
    res = []
    xdim = init_groups[0].shape[1]
    
    for gi,g in enumerate(init_groups):
        remain_size = g.shape[0] % batchsize
        if remain_size:
            padding_size = batchsize - (g.shape[0] % batchsize)
            padding = np.tile(g[-1,:].reshape(1,-1), reps=(padding_size,1))
            g = np.concatenate([g, padding], axis=0)
        else:
            padding_size = 0

        vals = []
        for i in range(0, g.shape[0], batchsize):
            # feed_dict[input_key] = g[i:i+batchsize,:]
            # val = sess.run(func, feed_dict=feed_dict)
            val = func(g[i:i+batchsize,:])
            vals.append(val)

        vals = np.squeeze(np.concatenate(vals))

        if padding_size:
            idxs = np.argsort(vals[:-padding_size])[-ks[gi]:]
        else:
            idxs = np.argsort(vals)[-ks[gi]:]

        res.append(g[idxs,:].reshape(-1,xdim))
    res = np.concatenate(res,axis=0)
    return res


def evaluate_tnorm_plus_norm_entropy_given_y_samples(
                mn, sn, mf, sf, # (n_average, n_mixture, nx)

                mix_probs, # (n_average, n_mixture)
                avg_probs, # (n_average,)

                maxfs, # (n_average, n_mixture, 1)
                y_samples, # (n_ysample, n_average, n_mixture, nx)
                
                dtype=tf.float32,
                use_loop=True,
                parallel_iterations=15):
    """
    for mni, sni, mfi, sfi, maxfsi in (mn, sn, mf, sf, maxfs)
        for mnj, snj, mfj, sfj, maxfsj in (mni, sni, mfi, sfi, maxfsi)
            Yj is a sum of Normal(mnj, snj)
                and TruncatedNormal(mfj, sfj, low=-infty, high=maxfsj)
        Yi ~ mixture of Yj
        Hi = entropy of Yi
    return average of Hi

    Requires:
        maxf -> several xmax -> several mf
        H[y|maxf] average over maxf (mixture of xmax)
        H[y|xmax,maxf] average over xmax, maxf
            which one to average?
            which one to mixture?

        mix_probs: (n_average, n_mixture)
        # mix_probs[i,j] probability for distribution [i,j] in mixture i

        maxfs: (n_average, n_mixture, 1)
        mf:    (n_average, n_mixture, nx)
        sf:    (n_average, n_mixture, nx)
        mn:    (n_average, n_mixture, nx)
        sn:    (n_average, n_mixture, nx)
        y_samples: (n_ysample, n_average, n_mixture, nx)

        average over n_average dimension
        mixture (marginalizing) over n_mixture dimension

        H[y|x] = int p(x,y) log p(y|x)
                 int p(y|x) p(x) log p(y|x) dx dy
                 int p(x) [int p(y|x) log p(y|x) dy] dx
                 -> requires different samples y for different x
                    sample pairs of (x,y) compute log p(y|x) do not mix between x and y!!!
                    y_samples: (n_ysample, n_average, n_mixture, nx)

        problem: condition on y, sample x is not easy
        sample pair x,y condition on y only 1 x????
        sample several y
            p(x*|y*) \propto p(y*|x*,D) p(x*|D)
            p(y*|x*,D) = ???
        I[y; x_*|y_*] = I[y; x_*,y_*] - I[y; y_*]
        I[y; y_*|x_*] = I[y; x_*,y_*] - I[y; x_*]
        I[y; x_*] = I[y; x_*,y_*] - I[y; y_*|x_*]
    """
    nx = tf.shape(mf)[1]
    n_average = tf.shape(mf)[0]
    n_mixture = tf.shape(mf)[1]

    # 2. evaluate log p(y|x_*, D)

    # rename:
    #   n_average: n_mixture
    #   n_mixture: mixture_size

    logpdf_no_z_part = evaluate_logpdf_no_z_part(mf, sf, mn, sn, maxfs, dtype=dtype)
    # shape = (n_average, n_mixture, nx)
    logpdf_no_z_part = tf.expand_dims(logpdf_no_z_part, axis=0)
    # shape = (1, n_average, n_mixture, nx)
    
    if not use_loop:
        logpdf_z_part = evaluate_logpdf_z_part(y_samples, mf, sf, mn, sn, maxfs, dtype=dtype)
        # shape = (n_ysample, n_average, n_mixture, nx)

        # log p(y| each distribution)
        logpdf_y_given_fmax = (- logpdf_no_z_part + logpdf_z_part)# * masked_sf# - 1e6 * (1.0 - masked_sf)
        # shape = (n_ysample, n_average, n_mixture, nx)

        # - log p(y| each mixture of distribution)
        negative_logp_y = - tf.math.reduce_logsumexp(
                            logpdf_y_given_fmax + tf.log( tf.reshape(mix_probs, shape=(1, n_average, n_mixture, 1)) ),
                            axis=2, keepdims=False)
        # shape = (n_ysample, n_average, nx)

        # average [-log p(y)] over all mixtures
        negative_logp_y = tf.reduce_mean(negative_logp_y 
                                         * tf.reshape(avg_probs, 
                                                      shape=(1, n_average, 1)), 
                                         axis=1)
        # shape = (n_ysample, nx)

    else:
        while_loop_cond = lambda i, lprob: i < n_average

        # loop over average
        def while_loop_body(i, avg_negative_logp_y):
            logpdf_z_part = evaluate_logpdf_z_part(
                                y_samples[:,i,...], # (n_ysample, n_average, n_mixture, nx)
                                mf[i,...], # (n_average, n_mixture, nx)
                                sf[i,...], 
                                mn[i,...], 
                                sn[i,...], 
                                maxfs[i,...], 
                                dtype=dtype)
            # shape = (n_ysample, n_mixture, nx)

            # log p(y) for each distribution
            logpdf_y_given_fmax = (- logpdf_no_z_part[0,i,...] + logpdf_z_part)
            # shape = (n_ysample, n_mixture, nx)

            # log p(y) for each mixture
            negative_logp_y = - tf.math.reduce_logsumexp(
                                logpdf_y_given_fmax + tf.log( 
                                    tf.reshape(mix_probs[i,:], shape=(1,n_mixture,1))),
                                axis=1, keepdims=False)
            # shape = (n_ysample, nx)

            avg_negative_logp_y = avg_negative_logp_y + negative_logp_y * avg_probs[i]
            return i + 1, avg_negative_logp_y

        _, avg_negative_logp_y = tf.while_loop(while_loop_cond, 
                                while_loop_body, 
                                (tf.constant(0), 
                                 tf.zeros((tf.shape(y_samples)[0], tf.shape(y_samples)[3]), dtype=dtype)),
                                parallel_iterations=parallel_iterations)
        # shape = (n_ysample, nx)


    entropy = tf.reduce_mean(avg_negative_logp_y, axis=0)
    # shape = (nx,)

    return entropy





def estimate_posterior_maxx_tensor(
            nhyp,
            meanf, # (nhyp, nmaxloc)
            stdf, # (nhyp, nmaxloc)
            maxfd, # (nhyp, nmaxfd)
            nsample): 
    nmaxloc = tf.shape(meanf)[1]
    nmaxfd = tf.shape(maxfd)[1]

    maxfd = tf.reshape(maxfd, shape=(nhyp, 1, nmaxfd))
    meanf = tf.reshape(meanf, shape=(nhyp, nmaxloc, 1))
    stdf = tf.reshape(stdf, shape=(nhyp, nmaxloc, 1))

    standardized_maxfd = (maxfd - meanf) / stdf

    standardized_samples = tfp.distributions.TruncatedNormal(
                loc = tf.zeros_like(standardized_maxfd),
                scale = tf.ones_like(standardized_maxfd),
                low = standardized_maxfd,
                high = np.infty * tf.ones_like(standardized_maxfd)
                ).sample(nsample)
    # shape = (nsample, nhyp, nmaxloc, nmaxfd)

    # no nhyp
    for i in range(nhyp):
        log_cdf = tfp.distributions.Normal(
                    loc=tf.expand_dims(meanf[i,...],0), 
                    scale=tf.expand_dims(stdf[i,...],0)
                ).log_cdf(standardized_samples[:,i,...])



    unnormalized_logprob = np.zeros([nhyp, nmaxfd, nmaxloc])
    
    for i in range(nhyp):
        for j in range(nmaxloc):
            samples = standardized_samples[:,i,j,:] * stdf[i,j] + meanf[i,j]
            # (nsample, nmaxfd)
            samples = samples.reshape(1, nsample, nmaxfd)
            # (1, nsample, nmaxfd)

            other_meanf = np.concatenate([meanf[i,:j], meanf[i,(j+1):]]).reshape(nmaxloc - 1, 1, 1)
            other_stdf = np.concatenate([stdf[i,:j], stdf[i,(j+1):]]).reshape(nmaxloc - 1, 1, 1)

            logcdf_samples = spst.norm.logcdf(samples, 
                                loc=other_meanf, scale=other_stdf)
            # (nmaxloc-1, nsample, nmaxfd)
            unnormalized_logprob_maxloc = \
                    sp.special.logsumexp(np.sum(logcdf_samples, axis=0), axis=0) \
                    - np.log(nsample) \
                    + np.log(spst.norm.sf(maxfd[i,:], loc=meanf[i,j], scale=stdf[i,j]))
            # (nmaxfd,)
            unnormalized_logprob[i,:,j] = unnormalized_logprob_maxloc

    log_normalizer = sp.special.logsumexp(unnormalized_logprob, axis=2).reshape(nhyp, nmaxfd, 1)

    prob = np.exp(unnormalized_logprob - log_normalizer)
    # (nhyp, nmaxfd, nmaxloc)

    return prob

"""
Issues:
    posterior_maxf (no sampling)
        meanf are correlated, should use the full matrix
            -> what is the cdf of multivariate Gaussian? (Tensorflow has built-in method)
        is it the correct implementation?
            higher maxf, more likely? No,
            current implementation, probability increases and stops increasing up to some point
            cdf(maxf) of function values at all observed inputs jointly
            * (sum pdf(maxf) of function values at all observed inputs)

    posterior_maxx
        (1) correlated between maxf

        (2) if sampling is used in this function -> stochastic optimizing for the acquisition function does not work!
        Solution:
            (2.1) avoid conditioning on fD: use the probability of the function sampling method. not correct as that is the probability of the whole function.
            (2.2) fixed the set of samples of fvalues at maximizer samples (required update full-cov of maxima give a fvalue sample), what if no samples > fD??? use deterministic sampling of fvalues: multi-dimensional: mean +- k*delta 2d samples, where delta is some small number
                standardized fD * np.ones(d) -> sfD, max(sfD) -> 2d samples
            (2.3) assuming independent
                p(x > y) ~ Gumbel distribution
                p(x > y) * p(x > fD)

                p(x > y) = int_x p(x) int_y p(x > y)
                    = int_x p(x) \Phi_y(x)
                    d/dx \Phi_x(x) \Phi_y(x) = p(x) \Phi_y(x) + \Phi_x(x) p_y(x)
                p(x > y)
                p(z*std_x + mu_x > z'*std_y + mu_y)
                    = \int_z p(z) int_z' p(z*std_x + mu_x > z'*std_y + mu_y)
                    = \int_z p(z) int_z' p( (z*std_x + mu_x - mu_y) / std_y > z')
                    = \int_z p(z) \Phi( (z*std_x + mu_x - mu_y) / std_y )
                k = (z*std_x + mu_x - mu_y) / std_y
                d/dx \Phi(z) \Phi(k) = p(z) \Phi(k) + \Phi(z) p(k) std_x / stdy
"""


def estimate_posterior_maxx_fullcov_tf(
            nhyp,
            nmaxloc,
            meanf, # (nhyp, nmaxloc)
            covf, # (nhyp, nmaxloc, nmaxloc)
            dtype=tf.float32): 
    """
    Requires:
        meanf: column vector (nx,1)
        covf: (nx,nx)
    using full covariance matrix of f_max
    not considering fD
    p(x_i \in X is max) = prod_j p(x_i > x_j) (assuming independent between pair of x's)
    p((xi,xj)) = N((xi,xj)| (mean_xi,mean_xj), cov)
    z = (xi,xj).T * [1, -1] ~ N( mean_xi - mean_xj, [1,-1].T * cov * [1,-1] )
    p(z > 0) = p(x_i > x_j) = Phi_z(0)

    IMPORTANT: all maxlocs must be different
        if not, a submatrix of covf is not positive definite
    """
    idxs = tf.constant(list(range(nmaxloc)), dtype=dtype)
    prob_max = []

    # wrong dimension for nhyp!!!!!!!!!
    for i in range(nmaxloc): 
        diff_eye = -tf.eye(nmaxloc, dtype=dtype) + tf.constant( np.eye(nmaxloc)[:,i].reshape(nmaxloc,1), dtype=dtype )
        prob_max_i = []

        for j in range(nhyp):
            varfj = tf.linalg.diag_part(tf.transpose(diff_eye) @ covf[j,...] @ diff_eye) # nmaxloc,
            meanfj = tf.reshape(meanf[j,...], shape=(1,nmaxloc)) @ diff_eye # nmaxloc,
            varfj = tf.squeeze(varfj)
            meanfj = tf.squeeze(meanfj)
            
            varfj = tf.where(tf.not_equal(idxs, i), varfj, tf.ones_like(varfj))
            meanfj = tf.where(tf.not_equal(idxs, i), meanfj, tf.zeros_like(meanfj))

            # avoid the issue that 2 maxlocs are the same
            # std of the difference of 2 variables is zero
            varfj_greater_clip_min = tf.greater(varfj, clip_min)
            varfj = tf.where(varfj_greater_clip_min, varfj, tf.ones_like(varfj))
            meanfj = tf.where(varfj_greater_clip_min, meanfj, tf.zeros_like(meanfj))

            stdfj = tf.sqrt(varfj)

            cdf_max = tfp.distributions.Normal(loc=meanfj, scale=stdfj).cdf(0.0)
            sf_max = tf.constant(1.0, dtype=dtype) - cdf_max
            # nmaxloc,
            
            prob_max_i.append( tf.reduce_prod(sf_max) )

        prob_max_i = tf.stack(prob_max_i) # (nhyp)
        prob_max.append(prob_max_i)

    prob_max = tf.stack(prob_max) # (nmaxloc, nhyp)
    prob_max = tf.transpose(prob_max) # (nhyp, nmaxloc)
    prob_max = prob_max / tf.expand_dims(tf.reduce_sum(prob_max, axis=1), 1)
    return prob_max # (nhyp, nmaxloc)


# def test_estimate_posterior_maxx_fullcov_tf():
#     nhyp = 1
#     nmaxloc = 3

#     meanf_np = np.array([ [1., 2., 3.] ])
#     covf_np = np.array([ [1.0, 0.0, 0.0],
#                          [0.0, 1.0, 0.0],
#                          [0.0, 0.0, 1.0] ])
    
#     meanf = tf.constant(meanf_np, dtype=tf.float32)
#     covf = tf.constant(covf_np, dtype=tf.float32)

#     probs = estimate_posterior_maxx_fullcov_tf(nhyp, nmaxloc,
#                     meanf, covf, dtype=tf.float32)
    
#     with tf.Session() as sess():
#         probs_val = sess.run(probs)

#         print(probs_val)


# def estimate_posterior_maxx_numpy(
#             meanf, # (nhyp, nmaxloc)
#             stdf, # (nhyp, nmaxloc)
#             maxfd, # (nhyp, nmaxfd)
#             nsample): 
#     """
#     using numpy
#     probability x being x*
#         p(f(x) > maxfd) * p(f(x) > f(x') for other x' in the set of maximizer)

#     Requires:
#         meanf, stdf: statistics of maximum at maximizers
#         maxfd: samples of maximum of function values of observed inputs so far
#         nsample: number of sample to estimate  the posterior
    
#     Returns:
#         the probabilities of each maximizers given hyp, maxfd
#             shape = (nhyp, nmaxfd, nmaxloc)
#     """
#     nhyp = meanf.shape[0]
#     nmaxloc = meanf.shape[1]
#     nmaxfd = maxfd.shape[1]

#     maxfd = maxfd.reshape(nhyp, 1, nmaxfd)
#     meanf = meanf.reshape(nhyp, nmaxloc, 1)
#     stdf = stdf.reshape(nhyp, nmaxloc, 1)

#     standardized_maxfd = (maxfd - meanf) / stdf

#     standardized_samples = spst.truncnorm.rvs(
#                     a=standardized_maxfd, 
#                     b=np.infty, 
#                     loc=np.zeros([nhyp, nmaxloc, nmaxfd]), 
#                     scale=np.ones([nhyp, nmaxloc, nmaxfd]),
#                     size=(nsample, nhyp, nmaxloc, nmaxfd) )

#     unnormalized_logprob = np.zeros([nhyp, nmaxfd, nmaxloc])
    
#     for i in range(nhyp):
#         for j in range(nmaxloc):
#             samples = standardized_samples[:,i,j,:] * stdf[i,j] + meanf[i,j]
#             # (nsample, nmaxfd)
#             samples = samples.reshape(1, nsample, nmaxfd)
#             # (1, nsample, nmaxfd)

#             other_meanf = np.concatenate([meanf[i,:j], meanf[i,(j+1):]]).reshape(nmaxloc - 1, 1, 1)
#             other_stdf = np.concatenate([stdf[i,:j], stdf[i,(j+1):]]).reshape(nmaxloc - 1, 1, 1)

#             logcdf_samples = spst.norm.logcdf(samples, 
#                                 loc=other_meanf, scale=other_stdf)
#             # (nmaxloc-1, nsample, nmaxfd)
#             unnormalized_logprob_maxloc = \
#                     sp.special.logsumexp(np.sum(logcdf_samples, axis=0), axis=0) \
#                     - np.log(nsample) \
#                     + np.log(spst.norm.sf(maxfd[i,:], loc=meanf[i,j], scale=stdf[i,j]))
#             # (nmaxfd,)
#             unnormalized_logprob[i,:,j] = unnormalized_logprob_maxloc

#     log_normalizer = sp.special.logsumexp(unnormalized_logprob, axis=2).reshape(nhyp, nmaxfd, 1)

#     prob = np.exp(unnormalized_logprob - log_normalizer)
#     # (nhyp, nmaxfd, nmaxloc)

#     return prob
    

def estimate_posterior_maxf_numpy(
            meanf, # (nhyp, nobs)
            stdf, # (nhyp, nobs)
            maxf # (nhyp, nmaxf)
            ):
    """
    using numpy
    probability maxf being f*
        p(f(x) < maxf for other x in the set of observed inputs so far)

    Requires:
        meanf, stdf: statistics of function values at observed inputs so far
        maxf: samples of the maximum
        nsample: number of sample to estimate the posterior
    
    Returns:
        the probabilities of each maximizers given hyp
            shape = (nhyp, nmaxf)
    """
    raise Exception("Should we include p(maxf) term? to reduce the probability of very large maxf?")
    nhyp = meanf.shape[0]
    nobs = meanf.shape[1]
    nmaxf = maxf.shape[1]

    maxf = maxf.reshape(nhyp, nmaxf, 1)
    meanf = meanf.reshape(nhyp, 1, nobs)
    stdf = stdf.reshape(nhyp, 1, nobs)

    standardized_maxf = (maxf - meanf) / stdf

    logcdf_maxf = spst.norm.logcdf(standardized_maxf, 
                    loc=np.zeros_like(standardized_maxf), 
                    scale=np.ones_like(standardized_maxf))
    # (nhyp, nobs, nmaxf)

    unnormalized_logprob_maxf = np.sum(logcdf_maxf, axis=2)
    # (nhyp, nmaxf)

    log_normalizer = sp.special.logsumexp(unnormalized_logprob_maxf, axis=1).reshape(nhyp,1)
    # (nhyp,1)

    prob_maxf = np.exp(unnormalized_logprob_maxf - log_normalizer)

    return prob_maxf


def test_estimate_posterior_maxx_numpy(): 
    mean_f = np.array([1., 2., 3., 4.]).reshape(1, -1)
    std_f = np.array( [2., 2., 2., 2.]).reshape(1, -1)

    # mean_f = np.array([1.]).reshape(1, -1)
    # std_f = np.array( [2.]).reshape(1, -1)

    nsample = 10
    maxfd = np.array([5., 10.]).reshape(1,-1)

    probs = estimate_posterior_maxx_numpy(
                mean_f,
                std_f,
                maxfd,
                nsample)

    return probs


def test_estimate_posterior_maxf_numpy():
    # mean_f = np.array([1., 2., 3., 4.]).reshape(1, -1)
    # std_f = np.array( [2., 2., 2., 2.]).reshape(1, -1)

    mean_f = np.array([1.]).reshape(1, -1)
    std_f = np.array( [2.]).reshape(1, -1)

    maxfd = np.array([5., 4.8]).reshape(1,-1)

    probs = estimate_posterior_maxf_numpy(
                mean_f,
                std_f,
                maxfd)

    return probs


def get_duplicate_mask_np(xs, resolution=1e-5):
    """
    duplicate_mask[i] = 1 if xs[i,:] is already in xs[:i,:]
    """
    n = xs.shape[0]

    duplicate_mask = np.zeros(n)

    for i in range(n):
        if duplicate_mask[i] == 1.0:
            # already duplicated
            continue 

        for j in range(i+1,n):
            adiff = xs[j,:] - xs[i,:]
            dist = np.sqrt( np.sum(adiff * adiff) )
            if dist <= resolution:
                duplicate_mask[j] = 1.0

    return duplicate_mask


def remove_duplicates_np(xs, resolution=1e-5):
    invalid_tests = get_duplicate_mask_np(xs, resolution)

    remove_idxs = np.where(invalid_tests == 1.0)[0]
    xs = np.delete(xs, remove_idxs, axis=0)
    return xs


def compute_post_maxidxs_np(
                mean, # ntest,1 or ntest,
                cov, # ntest, ntest
                max_idxs, # nmax,
                zero_prob=0.0,
                nlimit=None):
    """
    Requires: cov to be non-degenerate
        i.e., no duplicate x
    """
    # tested in test_ep.py 
    
    nmax = max_idxs.shape[0]
    ntest = mean.shape[0]
    mean = mean.reshape(ntest,1)
    
    if nlimit is None:
        nlimit = nmax
    
    # find top nlimit based on min zscore
    # in nmax
    var = np.diag(cov).reshape(ntest,1)
    eye = np.eye(ntest)
    minzscores = np.zeros(nmax)

    for i in range(nmax):
        c = eye.copy()
        c = np.delete(c, max_idxs[i], axis=1)
        c[max_idxs[i],:] = 1.0 

        mean_diff = c.T.dot(mean)
        # (ntest-1,1)
        std_diff = np.sqrt(c.T.dot(var)) # (ntest-1,1)
        zscore = -mean_diff / std_diff
        # (ntest-1,)
        minzscores[i] = np.min(zscore)

    sorted_minzscore_idxs = np.argsort(minzscores)
    compute_post_idxs = sorted_minzscore_idxs[:nlimit]


    probs = np.zeros(nmax)
    perturb = 1e-5
    for i in range(nmax):
        if i not in compute_post_idxs:
            probs[i] = 0.0

        c = eye.copy()
        c = np.delete(c, max_idxs[i], axis=1)
        c[max_idxs[i],:] = -1.0 

        mean_diff = c.T.dot(mean)
        # (ntest-1,1)

        cov_diff = c.T.dot(cov).dot(c)
        # (ntest-1,ntest-1)

        while np.linalg.eigvalsh(cov_diff)[0] < 1e-6:
            print("utils.py:compute_post_maxidxs_np: degenerate cov_diff, add perturb {}".format(perturb))
            cov_diff += np.eye(ntest-1) * perturb
            
        probs[i] = spst.multivariate_normal.cdf(np.zeros(ntest-1), mean=mean_diff.squeeze(), cov=cov_diff)

    probs /= np.sum(probs)    
    probs[np.where(probs < zero_prob)[0]] = 0.0
    probs /= np.sum(probs)

    return probs


def sample_multivariate_normal_maxidx_np(mean, cov, nsample, n_min_sample=1, get_sample=True, remove_correlated_dims=True):
    """
    mean: (n,1)
    cov: (n,n)

    Returns:
        as the numbers of samples for different maxidxs are different
        but we return a matrix of sample size for all maxidxs
        we need a mask array to say which samples are used
    """

    n = cov.shape[0]
    mean = mean.reshape(n,)

    eigvalues, eigvects = np.linalg.eig(cov)
    eigvalues = np.clip(eigvalues, a_min=0.0, a_max=np.infty)
    transform_cov_mat = eigvects.dot(np.diag(np.sqrt(eigvalues)))
    transform_cov_mat = np.expand_dims(transform_cov_mat, 0)
    # (1,n,n)

    standard_normal_sample = np.random.normal(size=(nsample,n,1))
    samples = np.mean(mean.reshape(1,n) + np.matmul(transform_cov_mat, standard_normal_sample), axis=-1)
    # (nsample, n)

    corr_idxs = []

    if remove_correlated_dims:
        # find dimensions that are highly correlated
        correlations = np.tril( np.corrcoef(samples.T), k=-1 )
        # remove highly correlated dimensions
        rows, cols = np.where(correlations > 1 - 1e-6)
        corr_idxs = rows
    
    ignore_correlated_samples = samples.copy()
    ignore_correlated_samples[:,corr_idxs] = -np.infty


    # samples = spst.multivariate_normal.rvs(mean=mean, cov=cov, size=nsample)
    # # (nsample,n)
    maxidxs = np.argmax(ignore_correlated_samples, axis=1)
    # (nsample,)

    unique_maxidxs = []
    max_samples = [] # list of samples with same maximizer
    max_samples_size = [] 

    for i in range(n):
        idxs = np.where(maxidxs == i)[0]
        ni = len(idxs)
        if ni >= n_min_sample:
            maxi_samples = samples[idxs, :]
            max_samples.append(maxi_samples)
            max_samples_size.append(ni)
            unique_maxidxs.append(i)
            
    nmax = len(unique_maxidxs)
    unique_maxidxs = np.array(unique_maxidxs)
    max_samples_size = np.array(max_samples_size)
    
    if get_sample:
        # return samples of different maximizer with same size
        # so we need return_masks to indicate which samples are used
        max_nsample = np.max(max_samples_size)
        return_samples = np.zeros([nmax, n, max_nsample]) 
        return_masks = np.zeros([nmax, max_nsample]).astype(int)

        for i,maxi in enumerate(unique_maxidxs):
            return_samples[i,:,:max_samples_size[i]] = max_samples[i].T
            return_samples[i,:,max_samples_size[i]:] = np.tile(mean.reshape(n,1), 
                                        reps=(1,max_nsample-max_samples_size[i]))
            return_masks[i,:max_samples_size[i]] = 1

        return nmax, unique_maxidxs, max_nsample, return_samples, return_masks, max_samples_size
    
    else:
        return unique_maxidxs, max_samples_size



def get_testidxs_stats(nhyp, 
                test_xs_np, # (ntest, xdim)
                test_means, # (nhyp, ntest)
                test_covs, # (nhyp, ntest, ntest)
                mode="sample",
                nsample=1000,
                n_min_sample=2): 
    """
    mode in {'sample', 'ep', 'empirical'}
    """

    ntest = test_xs_np.shape[0]
    maxidxs = set(range(ntest))
    max_probs = np.ones([nhyp, ntest]) / ntest
    max_nsample_all = 0

    post_test_samples = np.zeros([nhyp, ntest, ntest, nsample])
    post_test_masks = np.zeros([nhyp, ntest, nsample]).astype(bool)

    print("min empirical probability: {}%".format(n_min_sample / nsample * 100))

    for i in range(nhyp):

        nmax, unique_maxidxs, max_nsample, \
        return_samples, return_masks, max_samples_size \
            = sample_multivariate_normal_maxidx_np(
                test_means[i,:], 
                test_covs[i,:,:], 
                nsample = nsample, 
                n_min_sample=n_min_sample) 

        max_nsample_all = max_nsample_all if max_nsample_all > max_nsample else max_nsample
        maxidxs = maxidxs.intersection(unique_maxidxs)

        max_probs[i,unique_maxidxs] = max_samples_size / np.sum(max_samples_size)

        for j, maxidx in enumerate(unique_maxidxs):
            post_test_samples[i,maxidx,:,:max_nsample] = return_samples[j,:,:]
            post_test_masks[i,maxidx,:max_nsample] = return_masks[j,:]


    remove_idxs = np.array(list(set(range(ntest)).difference(maxidxs)))
    print(remove_idxs)
    if len(remove_idxs) != 0:
        test_xs_np = np.delete(test_xs_np, remove_idxs, axis=0)
        test_means = np.delete(test_means, remove_idxs, axis=1)
        test_covs = np.delete(test_covs, remove_idxs, axis=1)
        test_covs = np.delete(test_covs, remove_idxs, axis=2)
        max_probs = np.delete(max_probs, remove_idxs, axis=1)
    max_probs = max_probs / np.sum(max_probs, axis=1, keepdims=True)

    ntest = test_xs_np.shape[0]
    max_idxs = np.array(list(range(ntest)), dtype=int)

    if len(remove_idxs) != 0:
        post_test_samples = np.delete(post_test_samples, remove_idxs, axis=1)
        post_test_samples = np.delete(post_test_samples, remove_idxs, axis=2)
        post_test_masks = np.delete(post_test_masks, remove_idxs, axis=1)
    post_test_samples = np.delete(post_test_samples, 
                            list(range(max_nsample_all,nsample)), axis=3)
    post_test_masks = np.delete(post_test_masks, 
                            list(range(max_nsample_all,nsample)), axis=2)

    if mode == 'sample':
        return test_xs_np, max_probs, test_means, test_covs, post_test_samples, post_test_masks

    elif mode == 'empirical':
        post_mean_tests = np.zeros([nhyp, ntest, ntest])
        post_cov_tests = np.zeros([nhyp, ntest, ntest, ntest])

        for i in range(nhyp):
            for j in range(ntest):
                m,c = empirical_approximation.get_empirical_stat_from_samples(
                        post_test_samples[i,j,:,:].T, 
                        weight = post_test_masks[i,j,:].astype(int))

                post_mean_tests[i,j,:] = m.squeeze()
                post_cov_tests[i,j,:,:] = c
        
        return test_xs_np, max_probs, test_means, test_covs, post_mean_tests, post_cov_tests

    elif mode == 'ep':
        post_mean_tests = np.zeros([nhyp, ntest, ntest])
        post_cov_tests = np.zeros([nhyp, ntest, ntest, ntest])
        
        for i in range(nhyp):
            for j in range(ntest):
                m,c = ep.approximate_EP_np(
                    j,
                    test_means[i,:].reshape(-1,1),
                    test_covs[i,:,:],
                    resolution = 1e-9,
                    max_niter = 100)

                post_mean_tests[i,j,:] = m.squeeze()
                post_cov_tests[i,j,:,:] = c
        
        return test_xs_np, max_probs, test_means, test_covs, post_mean_tests, post_cov_tests

    else:
        raise Exception("Unknown mode in get_testidxs_stats!")




def precomputeInvKs(xdim, nhyp, ls, 
                    sigmas, sigma0s, 
                    Xsamples, 
                    dtype):

    invKs = []
    for i in range(nhyp):

        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        NK = computeNKmm(Xsamples, l, sigma, sigma0, dtype=dtype)

        invK = chol2inv(NK, dtype=dtype)
        invKs.append(invK)

    invKs = tf.stack(invKs)
    return invKs


def eval_invKmaxsams(xdim, nhyp, nmax, 
                    ls, sigmas, sigma0s, 
                    Xsamples, 
                    maxima, 
                    dtype=tf.float32):
    # only required for PES criterion
    invKmaxsams = []
    for i in range(nhyp):

        l = tf.reshape(ls[i,:], shape=(1,xdim))
        sigma = sigmas[i]
        sigma0 = sigma0s[i]

        invKmaxsams_i = []
        for j in range(nmax):
            xmax_xsam = tf.concat([tf.reshape(maxima[i,j,...], shape=(1,xdim)), Xsamples], axis=0)
            NKmaxsam = computeNKmm(xmax_xsam, l, sigma, sigma0, dtype=dtype)
            invKmaxsam = chol2inv(NKmaxsam, dtype=dtype)
            invKmaxsams_i.append(invKmaxsam)

        invKmaxsams.append(tf.stack(invKmaxsams_i))
    invKmaxsams = tf.stack(invKmaxsams)
    # nhyp x nmax x (Xsamples.shape[0] + maxima.shape[0]) x (Xsamples.shape[0] + maxima.shape[0])

    return invKmaxsams


# Performance measure
def evaluate_discrete_maximum_distribution_np(xs, xs_meanf, xs_covf, nsample):
    """
    Requires:
        xs: nx,xdim
        xs_meanf: nx
        xs_covf: nx,nx
    """
    unique_maxidxs, max_samples_size \
        = sample_multivariate_normal_maxidx_np(
                xs_meanf.squeeze(), xs_covf, nsample, 
                n_min_sample=1, get_sample=False)
    
    maximizers = xs[unique_maxidxs,:]
    frequency = max_samples_size / np.sum(max_samples_size)

    # sorted_idxs = np.argsort(frequency.squeeze())[-10:]
    # print("maximizers: {}".format(list(zip(maximizers[sorted_idxs].squeeze(), frequency[sorted_idxs].squeeze())) ))

    return maximizers, frequency


def evaluate_discrete_regret_distribution_np(nhyp, 
                            true_maximum,
                            xs, xs_meanf, xs_covf, 
                            nsample, func,
                            debug=False):
    """
    Requires:
        xs: nx,xdim

        xs_meanf: nhyp, nx
        xs_cov: nhyp, nx, nx
    """
    regret_mean = 0.0
    regret2_mean = 0.0
    regret_all = np.array([])
    frequency_all = np.array([])

    # true_maximum = np.max(func(xs)).squeeze()
    for i in range(nhyp):
        maximizers, frequency = evaluate_discrete_maximum_distribution_np(
                                xs, xs_meanf[i,:], xs_covf[i,:,:], nsample)
        maxima = func(maximizers).squeeze()

        regret = true_maximum - maxima
        regret_mean += np.sum( regret * frequency ) / nhyp
        regret2_mean += np.sum( regret * regret * frequency ) / nhyp

        if debug:
            regret_all = np.concatenate([regret_all, regret])
            frequency_all = np.concatenate([frequency_all, frequency / nhyp])

    regret_std = np.sqrt(regret2_mean - regret_mean * regret_mean)

    if debug:
        return regret_all, frequency_all, regret_mean, regret_std

    return regret_mean, regret_std


def find_duplicate_resolution(xdim, sigmas_np, ls_np, size=2, min_resolution=1e-6, max_resolution = 10):
    # find a proper duplicate resolution
    print("Finding a duplicate resolution for GP hyperparameters:")
    duplicate_resolution = 1.0

    while True:
        duplicate_resolution = (min_resolution + max_resolution) / 2.0

        dummy_x = np.ones(xdim) * duplicate_resolution**2
        dummy_cov = sigmas_np[0] * np.sum(np.exp(-0.5 * ls_np[0] * dummy_x))


        dummy_covmat = np.eye(size) * sigmas_np[0] + (1.0 - np.eye(size)) * dummy_cov

        test_cond = np.linalg.cond(dummy_covmat)

        if test_cond < 1e3:
            max_resolution = duplicate_resolution
            if max_resolution - min_resolution < 1e-3:
                break
        else:
            min_resolution = duplicate_resolution
    
    return duplicate_resolution
