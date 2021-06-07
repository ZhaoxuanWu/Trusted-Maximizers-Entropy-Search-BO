import numpy as np 
import scipy 
import scipy.stats

import matplotlib.pyplot as plt 



def get_samples_by_importance_sampling(mean_xs, cov_xs, max_idx, nsample=100, ntrial=1000):
    xdim = cov_xs.shape[0]
    mean_xs = mean_xs.reshape(-1,)
    
    all_samples = np.array([])

    while all_samples.shape[0] < nsample:
            
        samples = scipy.stats.multivariate_normal.rvs(mean_xs, cov_xs, ntrial)
        # (nsample, xdim)

        valid = np.ones(ntrial, dtype=int)
        for i in range(xdim):
            valid_i = (samples[:,max_idx] >= samples[:,i]).astype(int)
            valid = valid * valid_i
        valid_samples = samples[np.where(valid == 1)[0],:]

        if all_samples.shape[0] == 0:
            all_samples = valid_samples
        else:
            all_samples = np.concatenate([all_samples, valid_samples], axis=0)

    all_samples = all_samples[:nsample,:]
    return all_samples


def get_samples_by_conditional_sampling(mean_xs, cov_xs, max_idx, nsample=100):
    xdim = cov_xs.shape[0]
    mean_xs = mean_xs.reshape(-1,)
    
    max_idx_mean = mean_xs[max_idx]
    max_idx_var = cov_xs[max_idx,max_idx]

    non_max_mean = np.concatenate([mean_xs[:max_idx], mean_xs[(max_idx+1):]])
    non_max_cov = np.concatenate([cov_xs[:max_idx,:], cov_xs[(max_idx+1):]], axis=0)
    non_max_cov = np.concatenate([non_max_cov[:,:max_idx], non_max_cov[:,(max_idx+1):]], axis=1)
            
    non_max_samples = scipy.stats.multivariate_normal.rvs(non_max_mean, non_max_cov, nsample).reshape(nsample,-1)
    # (nsample,nx-1)
    max_non_max_samples = np.max(non_max_samples, axis=1)
    # (nsample,)

    # compute conditional max_idx distribution
    # shape (nsample,)
    k_maxidx_rest = np.concatenate([cov_xs[max_idx,:(max_idx)], cov_xs[max_idx,(max_idx+1):]])
    invk_rest_rest = np.linalg.inv(non_max_cov)
    conditional_max_idx_mean = np.squeeze(
                        max_idx_mean
                        + k_maxidx_rest.dot(invk_rest_rest).dot((non_max_samples - non_max_mean.reshape(1,-1)).T))
    conditional_max_idx_var = np.squeeze(max_idx_var - k_maxidx_rest.dot(invk_rest_rest).dot(k_maxidx_rest.T))

    logweight = scipy.stats.norm.logsf(max_non_max_samples, loc=max_idx_mean, scale=np.sqrt(max_idx_var))
    # (xdim,)

    logweight = logweight - scipy.special.logsumexp(logweight)
    weight = np.exp(logweight)

    max_idx_samples = scipy.stats.truncnorm.rvs(a=(max_non_max_samples-conditional_max_idx_mean)/np.sqrt(conditional_max_idx_var), b=np.infty, loc=conditional_max_idx_mean, scale=np.sqrt(conditional_max_idx_var)).reshape(nsample,1)
    # (nsample,1)



    infidxs = np.where(np.isinf(max_idx_samples))[0]

    if np.any(np.isinf(max_idx_samples)):
        print("empirical_approximation.py:get_samples_by_conditional_sampling:truncnorm sampling return INF sample! Remove {} inf samples".format(infidxs.shape[0]))

    samples = np.concatenate([non_max_samples[:,:max_idx], max_idx_samples, non_max_samples[:,max_idx:]], axis=1)
    samples = np.delete(samples, infidxs, axis=0)
    weight = np.delete(weight, infidxs)
    weight *= weight.shape[0]

    return samples, weight




def get_empirical_stat_from_samples(samples, weight=None):
    # samples: nx, xdim
    # weight: nx,

    n = samples.shape[0]
    if weight is None:
        weight = np.ones(n)

    total_weight = np.sum(weight)

    weight = weight.reshape(-1,1)

    emean = np.sum(samples * weight, axis=0, keepdims=True) / total_weight
    # (1,xdim)

    zero_mean_samples = (samples - emean) * np.sqrt(weight)
    ecov = zero_mean_samples.T.dot(zero_mean_samples) / (total_weight - 1.0)
    # (xdim,xdim)

    return emean.reshape(-1,1), ecov
    # (xdim,1)
    # (xdim,xdim)


def get_empirical_stat(mean_xs, cov_xs, max_idx, nsample=100, ntrial=1000, importance_sampling=False):
    mean_xs = mean_xs.reshape(-1,)

    if importance_sampling:
        samples = get_samples_by_importance_sampling(mean_xs, cov_xs, max_idx, nsample, ntrial)
        weights = None
    else:
        samples, weights = get_samples_by_conditional_sampling(mean_xs, cov_xs, max_idx, nsample)
    
    if len(samples) >  0:
        mean, cov = get_empirical_stat_from_samples(samples, weights)
        return mean, cov 

    return None, None

