from __future__ import division
import numpy as np
import scipy.spatial.distance as ssd
import scipy.spatial as sp_spat
from rapprentice import tps, svds, math_utils
from collections import defaultdict
import IPython as ipy
from pdb import pm
from rapprentice.registration import loglinspace, ThinPlateSpline, fit_ThinPlateSpline
import rapprentice.registration as registration

def balance_matrix3(prob_nm, max_iter, p, outlierfrac, r_N = None):
    
    n,m = prob_nm.shape
    prob_NM = np.empty((n+1, m+1), 'f4')
    prob_NM[:n, :m] = prob_nm
    prob_NM[:n, m] = p
    prob_NM[n, :m] = p
    prob_NM[n, m] = p*np.sqrt(n*m)
    
    a_N = np.ones((n+1),'f4')
    a_N[n] = m*outlierfrac
    b_M = np.ones((m+1),'f4')
    b_M[m] = n*outlierfrac
    
    if r_N is None: r_N = np.ones(n+1,'f4')

    for _ in xrange(max_iter):
        c_M = b_M/r_N.dot(prob_NM)
        r_N = a_N/prob_NM.dot(c_M)

    prob_NM *= r_N[:,None]
    prob_NM *= c_M[None,:]
    
    return prob_NM[:n, :m], r_N, c_M

def tps_rpm_bij_corr(x_nd, y_md, n_iter = 50, reg_init = 10, reg_final = .1, rad_init = .1, rad_final = .005, rot_reg = np.r_[1e-4, 1e-4, 1e-1], 
            plotting = False, plot_cb = None, outlierprior = .1, outlierfrac = 1e-2):
    """
    tps-rpm algorithm mostly as described by chui and rangaran
    reg_init/reg_final: regularization on curvature
    rad_init/rad_final: radius for correspondence calculation (meters)
    plotting: 0 means don't plot. integer n means plot every n iterations
    interest_pts are points in either scene where we want a lower prior of outliers
    """
    x_nd, x_params = registration.unit_boxify(x_nd)
    y_md, y_params = registration.unit_boxify(y_md)
    
    _,d=x_nd.shape
    regs = loglinspace(reg_init, reg_final, n_iter)
    rads = loglinspace(rad_init, rad_final, n_iter)

    f = ThinPlateSpline(d)
    f.trans_g = np.median(y_md,axis=0) - np.median(x_nd,axis=0) # align the medians
    # do a coarse search through rotations
    # fit_rotation(f, x_nd, y_md)
    
    g = ThinPlateSpline(d)
    g.trans_g = -f.trans_g

    # set up outlier priors for source and target scenes
    n, _ = x_nd.shape
    m, _ = y_md.shape

    for i in xrange(n_iter):
        xwarped_nd = f.transform_points(x_nd)
        ywarped_md = g.transform_points(y_md)
        
        fwddist_nm = ssd.cdist(xwarped_nd, y_md,'euclidean')
        invdist_nm = ssd.cdist(x_nd, ywarped_md,'euclidean')
        
        r = rads[i]
        prob_nm = np.exp( -(fwddist_nm + invdist_nm) / (2*r) )
        corr_nm, r_N, _ =  balance_matrix3(prob_nm, 10, outlierprior, outlierfrac) # edit final value to change outlier percentage
        corr_nm += 1e-9
        
        wt_n = corr_nm.sum(axis=1)
        wt_m = corr_nm.sum(axis=0)

        xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
        ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)
        
        if plotting and i%plotting==0:
            plot_cb(x_nd, y_md, xtarg_nd, corr_nm, wt_n, f)
        
        f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef = regs[i], wt_n=wt_n, rot_coef = rot_reg)
        g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef = regs[i], wt_n=wt_m, rot_coef = rot_reg)
    return corr_nm

def tps_rpm_bij2(x_nd, y_md, n_iter = 50, reg_init = 10, reg_final = .1, rad_init = .1, rad_final = .005, rot_reg = np.r_[1e-4, 1e-4, 1e-1], 
            plotting = False, plot_cb = None, outlierprior = .1, outlierfrac = 1e-2):
    corr_nm = tps_rpm_bij_corr(x_nd, y_md, n_iter=n_iter, reg_init=reg_init, reg_final=reg_final, rad_init=rad_init, rad_final=rad_final, rot_reg=rot_reg, 
                                                   plotting=plotting, plot_cb=plot_cb, outlierprior=outlierprior, outlierfrac=outlierfrac)
    return fit_ThinPlateSpline_bij(x_nd, y_md, corr_nm, reg_final, rot_reg)

def fit_ThinPlateSpline_bij(x_nd, y_md, corr_nm, bend_coef = .1, rot_coef = np.r_[1e-4, 1e-4, 1e-1]):
    wt_n = corr_nm.sum(axis=1)
    wt_m = corr_nm.sum(axis=0)

    xtarg_nd = (corr_nm/wt_n[:,None]).dot(y_md)
    ytarg_md = (corr_nm/wt_m[None,:]).T.dot(x_nd)

    f = fit_ThinPlateSpline(x_nd, xtarg_nd, bend_coef=bend_coef, wt_n=wt_n, rot_coef=rot_coef)
    g = fit_ThinPlateSpline(y_md, ytarg_md, bend_coef=bend_coef, wt_n=wt_m, rot_coef=rot_coef)
    
    f._bend_coef = bend_coef
    f._rot_coef = rot_coef
    g._bend_coef = bend_coef
    g._rot_coef = rot_coef

    return f, g, corr_nm
    