import numpy as np 
import scipy as sp 
import scipy.stats as spst
import sys



def get_approx_stat_xs(
                cs, # (nx, nx-1)
                mean_xs, # (nx,1)
                inv_cov_xs, # (nx,nx)
                site_means, site_vars # (nx-1,)
                ):
    """
    Returns:
        approx_mean: (nx,1)
        approx_cov: (nx,nx)
    """

    cs_mul_prec = cs / np.sqrt(site_vars.reshape(1,-1))

    approx_cov = np.linalg.inv(inv_cov_xs + cs_mul_prec.dot(cs_mul_prec.T))
    approx_mean = approx_cov.dot(
                            inv_cov_xs.dot(mean_xs) 
                            + np.sum(cs * site_means.reshape(1,-1) / site_vars.reshape(1,-1), axis=1, keepdims=True)
                    )
    
    return approx_mean, approx_cov


def get_cavity_parameters(
                    max_idx,
                    cs, # (nx, nx-1)
                    approx_mean, # (nx,1)
                    approx_cov, # (nx,nx)
                    site_means, # (nx-1,)
                    site_vars # (nx-1,)
                    ):
    cav_vars = 1.0 / (
        1.0 / np.diag(cs.T.dot(approx_cov).dot(cs)) - 
        1.0 / site_vars)
    # (nx-1,)

    cav_means = cav_vars * (
        cs.T.dot(approx_mean).squeeze() / np.diag(cs.T.dot(approx_cov).dot(cs))
        - site_means / site_vars)
    # (nx-1)

    return cav_means, cav_vars 


def get_cf_statistics(cav_means, # (nx-1,)
                    cav_vars): # (nx-1,)
    beta = cav_means / np.sqrt(cav_vars)

    pdf_over_cdf = spst.norm.pdf(beta) / spst.norm.cdf(beta)
    cf_means = cav_means + np.sqrt(cav_vars) * pdf_over_cdf
    cf_vars = - cav_means * np.sqrt(cav_vars) * pdf_over_cdf - cf_means * cf_means + 2.*cav_means*cf_means - cav_means*cav_means + cav_vars
    
    return cf_means, cf_vars


def update_site_parameters(cf_means, cf_vars, cav_means, cav_vars):
    """
    Requires: all parameters of shape (nx-1,)
    """
    site_vars = 1.0 / (1.0 / cf_vars - 1.0 / cav_vars)
    site_means = site_vars * (cf_means / cf_vars - cav_means / cav_vars)

    return site_means, site_vars # (nx-1,)


def approximate_EP_np( 
            max_idx,
            mean_xs,
            cov_xs,
            resolution=1e-9,
            max_niter=1000):
    """
    Requires:
        l: (xdim,)
        sigma, sigma0: scalar
        sigma0: noise variance
        max_idx: integer in: 0,..,nx (exclusive nx)
        X: (nxdata, xdim)
        Y: (nxdata,1)

        mean_xs: (nx,1)
        cov_xs: (nx,nx)
    """
    mean_xs = mean_xs.reshape(-1,1)
    nx = mean_xs.shape[0]

    # there are nx - 1: vector c
    # for nx - 1 differences: xi - x_max_idx
    cs = np.zeros([nx,nx-1])

    j = 0
    for i in range(nx):
        if i != max_idx:
            cs[max_idx][j] = 1
            cs[i][j] = -1
            j += 1

    # site parameters: (nx - 1)
    Zs = np.ones(nx-1) # \tilde{Z}_i
    site_means = cs.T.dot(mean_xs).reshape(nx-1) #np.zeros(nx-1) # \tilde{\mu}_i
    site_vars = np.diag(cs.T.dot(cov_xs).dot(cs)).copy() #np.ones(nx-1) # \tilde{\tau}_i

    inv_cov_xs = np.linalg.inv(cov_xs)

    approx_mean, approx_cov = get_approx_stat_xs(cs, 
                        mean_xs, 
                        inv_cov_xs, 
                        site_means, site_vars)
    # (nx,1), (nx,nx)

    count_niter = 0
    converge_diff = 1e10
    while converge_diff > resolution and count_niter < max_niter:
        count_niter += 1

        # cavity distribution parameters
        # of cf
        cav_means, cav_vars = get_cavity_parameters(
                        max_idx,
                        cs, # (nx, nx-1)
                        approx_mean, # (nx,1)
                        approx_cov, # (nx,nx)
                        site_means, # (nx-1,)
                        site_vars # (nx-1,)
                        )
        # (nx-1,)

        # if np.any(cav_vars <= 0.):
        #     print("NEGATIVE CAVITY VAR")
        #     print(site_means, site_vars)
        #     print(cav_means, cav_vars)
        #     raise Exception("err")

        # statistics of cavity distribution * original distribution[i]
        # t(cf)w^{\i}(cf)
        cf_means, cf_vars = get_cf_statistics(cav_means, cav_vars)

        # if np.any(cf_vars <= 0.):
        #     print("NEGATIVE CF VAR")
        #     print(site_means, site_vars)
        #     print(cav_means, cav_vars)
        #     print(cf_means, cf_vars)
        #     raise Exception("err")

        # update site parameters
        updated_site_means, updated_site_vars = update_site_parameters(cf_means, cf_vars, cav_means, cav_vars)
        
        # if np.any(updated_site_vars <= 0.):
        #     print("NEGATIVE SITE VAR")
        #     print(cf_means, cf_vars)
        #     print(cav_means, cav_vars)
        #     print(updated_site_means, updated_site_vars)
        #     raise Exception("err")

        updated_idx = count_niter % (nx-1)
        
        # if site_var is negative for current updated_idx, update another idx
        last_count_niter = count_niter - 1 if count_niter >= 1 else nx - 2

        while (updated_site_vars[updated_idx] <= 0 
                or np.isnan(updated_site_vars[updated_idx])
                or np.isinf(updated_site_vars[updated_idx])
        ) and updated_idx != last_count_niter:
            count_niter += 1    
            updated_idx = count_niter % (nx-1)
        
        # if count_niter == last_count_niter:
        #     print("All updates return nan!")
        #     print("EP stops at {} iteration, parameters difference = {:.2e}".format(count_niter, converge_diff))
        #     return approx_mean, approx_cov
        
        site_means[updated_idx] = updated_site_means[updated_idx]
        site_vars[updated_idx] = updated_site_vars[updated_idx]

        updated_approx_mean, updated_approx_cov = get_approx_stat_xs(cs, 
                            mean_xs, 
                            inv_cov_xs, 
                            site_means, site_vars)

        diff_mean = updated_approx_mean - approx_mean
        diff_cov = updated_approx_cov - approx_cov

        converge_diff = np.mean(diff_mean * diff_mean + diff_cov * diff_cov)

        is_cov_nan = np.any(np.isnan(updated_approx_cov))
        is_mean_nan = np.any(np.isnan(updated_approx_mean))
        if is_cov_nan or is_mean_nan:
            print("Early stopping EP due to nan! mean:{} cov:{}".format(is_mean_nan, is_cov_nan))
            print("EP stops at {} iteration, parameters difference = {:.2e}".format(count_niter, converge_diff))
            return approx_mean, approx_cov

        approx_mean, approx_cov = updated_approx_mean, updated_approx_cov 


    print("EP stops at {} iteration, parameters difference = {:.2e}".format(count_niter, converge_diff))
    
    return approx_mean, approx_cov

