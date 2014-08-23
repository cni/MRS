"""
Utility functions for analysis of MRS data.

""" 

import warnings
import numpy as np
import scipy.linalg as la
import scipy.stats as stats
import nitime.timeseries as nts

"""
lowess: Locally linear regression  
==================================

Implementation of the LOWESS algorithm in n dimensions.

[FHT] Friedman, Hastie and Tibshirani (2008). The Elements of Statistical
Learning; Chapter 6 

[Cleveland79] Cleveland (1979). Robust Locally Weighted Regression and Smoothing
Scatterplots. J American Statistical Association, 74: 829-836.

"""

# Kernel functions:
def epanechnikov(xx, idx=None):
    """
    The Epanechnikov kernel estimated for xx values at indices idx (zero
    elsewhere) 

    Parameters
    ----------
    xx: float array
        Values of the function on which the kernel is computed. Typically,
        these are Euclidean distances from some point x0 (see do_kernel)

    idx: tuple
        An indexing tuple pointing to the coordinates in xx for which the
        kernel value is estimated. Default: None (all points are used!)  

    Notes
    -----
    This is equation 6.4 in FHT chapter 6        
    
    """        
    ans = np.zeros(xx.shape)
    ans[idx] = 0.75 * (1-xx[idx]**2)
    return ans

def tri_cube(xx, idx):
    """ 
    The tri-cube kernel estimated for xx values at indices idx (zero
    elsewhere) 

    Parameters
    ----------
    xx: float array
        Values of the function on which the kernel is computed. Typically,
        these are Euclidean distances from some point x0 (see do_kernel)

    idx: tuple
        An indexing tuple pointing to the coordinates in xx for which the
        kernel value is estimated. Default: None (all points are used!)  

    Notes
    -----
    This is equation 6.6 in FHT chapter 6        
    """        

    ans = np.zeros(xx.shape)
    ans[idx] = (1-np.abs(xx[idx])**3)**3
    return ans

def do_kernel(x0, x, l=1.0, kernel=epanechnikov):
    """
    Calculate a kernel function on x in the neighborhood of x0

    Parameters
    ----------
    x: float array
       All values of x
    x0: float
       The value of x around which we evaluate the kernel
    l: float or float array (with shape = x.shape)
       Width parameter (metric window size)
    
    """
    # xx is the norm of x-x0. Note that we broadcast on the second axis for the
    # nd case and then sum on the first to get the norm in each value of x:
    xx = np.sum(np.sqrt(np.power(x-x0[:,np.newaxis], 2)), 0)
    idx = np.where(np.abs(xx<=1))
    return kernel(xx,idx)/l


def bi_square(xx, idx=None):
    """
    The bi-square weight function calculated over values of xx

    Parameters
    ----------
    xx: float array

    Notes
    -----
    This is the first equation on page 831 of [Cleveland79].
    """

    ans = np.zeros(xx.shape)
    ans[idx] = (1-xx[idx]**2)**2
    return ans


def lowess(x, w, x0, kernel=epanechnikov, l=1, robust=False):
    """
    Locally linear regression with the LOWESS algorithm.

    Parameters
    ----------
    x: float n-d array  
       Values of x for which f(x) is known (e.g. measured). The shape of this
       is (n, j), where n is the number the dimensions of the and j is the
       number of distinct coordinates sampled.  
    
    w: float array
       The known values of f(x) at these points. This has shape (j,) 

    x0: float or float array.
        Values of x for which we estimate the value of f(x). This is either a
        single scalar value (only possible for the 1d case, in which case f(x0)
        is estimated for just that one value of x), or an array of shape (n, k).

    kernel: callable
        A kernel function. {'epanechnikov', 'tri_cube'}

    l: float or float array with shape = x.shape
       The metric window size for the kernel

    robust: bool
        Whether to apply the robustification procedure from [Cleveland79], page
        831 
        
    Returns
    -------
    The function estimated at x0. 

    Notes
    -----
    The solution to this problem is given by equation 6.8 in Friedman, Hastie
    and Tibshirani (2008). The Elements of Statistical Learning (Chapter 6).

    Example
    -------
    >>> import lowess as lo
    >>> import numpy as np

    # For the 1D case:
    >>> x = np.random.randn(100)
    >>> f = np.cos(x) + 0.2 * np.random.randn(100)
    >>> x0 = np.linspace(-1,1,10)
    >>> f_hat = lo.lowess(x, f, x0)
    >>> import matplotlib.pyplot as plt
    >>> fig,ax = plt.subplots(1)
    >>> ax.scatter(x,f)
    >>> ax.plot(x0,f_hat,'ro')
    >>> plt.show()

    # 2D case (and more...)
    >>> x = np.random.randn(2, 100)
    >>> f = -1 * np.sin(x[0]) + 0.5 * np.cos(x[1]) + 0.2*np.random.randn(100)
    >>> x0 = np.mgrid[-1:1:.1, -1:1:.1]
    >>> x0 = np.vstack([x0[0].ravel(), x0[1].ravel()])
    >>> f_hat = lo.lowess(x, f, x0, kernel=lo.tri_cube)
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.scatter(x[0], x[1], f)
    >>> ax.scatter(x0[0], x0[1], f_hat, color='r')

    """

    if robust:
        # We use the procedure described in 
        # Start by calling this function with robust set to false and the x0
        # input being equal to the x input:
        w_est = lowess(x, w, x, kernel=epanechnikov, l=1, robust=False)
        resid = w_est - w
        median_resid = stats.nanmedian(np.abs(resid))
        robustness_weights = bi_square(resid/(6*median_resid))
        # Calculate the bi-cube function on the
        # Sub-select
        
    # For the case where x0 is provided as a scalar: 
    if not np.iterable(x0):
       x0 = np.asarray([x0])
    ans = np.zeros(x0.shape[-1]) 
    # We only need one design matrix:
    B = np.vstack([np.ones(x.shape[-1]), x]).T
    for idx, this_x0 in enumerate(x0.T):
        # This is necessary in the 1d case (?):
        if not np.iterable(this_x0):
            this_x0 = np.asarray([this_x0])
        # Different weighting kernel for each x0:
        W = np.diag(do_kernel(this_x0, x, l=l, kernel=kernel))

        # XXX It should be possible to calculate W outside the loop, if x0 and
        # x are both sampled in some regular fashion (that is, if W is the same
        # matrix in each iteration). That should save time.

        if robust:
            # We apply the robustness weights to the weighted least-squares
            # procedure:
            robustness_weights[np.isnan(robustness_weights)] = 0
            W = np.dot(W, np.diag(robustness_weights))

        try: 
            # Equation 6.8 in FHT:
            BtWB = np.dot(np.dot(B.T, W), B)
            BtW = np.dot(B.T, W)
            # Get the params:
            beta = np.dot(np.dot(la.inv(BtWB), BtW), w.T)
            # Estimate the answer based on the parameters:
            ans[idx] += beta[0] + np.dot(beta[1:], this_x0)
        # If we are trying to sample far away from where the function is
        # defined, we will be trying to invert a singular matrix. In that case,
        # the regression should not work for you and you should get a nan:
        except la.LinAlgError:
            ans[idx] += np.nan

            

    return ans.T


def ols_matrix(A, norm_func=None):
    """
    Generate the matrix used to solve OLS regression.

    Parameters
    ----------

    A: float array
        The design matrix

    norm: callable, optional
        A normalization function to apply to the matrix, before extracting the
        OLS matrix.

    Notes
    -----

    The matrix needed for OLS regression for the equation:

    ..math ::

        y = A \beta

   is given by:

    ..math ::

        \hat{\beta} = (A' x A)^{-1} A' y

    See also
    --------
    http://en.wikipedia.org/wiki/Ordinary_least_squares#Estimation
    """

    A = np.asarray(A)
    
    if norm_func is not None:
        X = np.matrix(unit_vector(A.copy(), norm_func=norm_func))
    else:
        X = np.matrix(A.copy())

    return la.pinv(X.T * X) * X.T

def l2_norm(arr):
    """
    The l2 norm of an array is is defined as: sqrt(||x||), where ||x|| is the
    dot product of the vector.
    """
    arr = np.asarray(arr)
    return np.sqrt(np.dot(arr.ravel().squeeze(), arr.ravel().squeeze()))


def unit_vector(arr, norm_func=l2_norm):
    """
    Normalize the array by a norm function (defaults to the l2_norm)

    Parameters
    ----------
    arr: ndarray

    norm_func: callable
       A function that is used here to normalize the array.

    Note
    ----
    The norm_func is the thing that defines what a unit vector is for the
    particular application.
    
    """
    return arr/norm_func(arr)

def zero_pad(ts, n_zeros):
    """
    Pad a nitime.TimeSeries class instance with n_zeros before and after the
    data

    Parameters
    ----------
    ts : a nitime.TimeSeries class instance
    
    """
    
    zeros_shape = ts.shape[:-1] + (n_zeros,)
    zzs = np.zeros(zeros_shape)
    # Concatenate along the time-dimension:
    new_data = np.concatenate((zzs, ts.data, zzs), axis=-1)

    return nts.TimeSeries(new_data, sampling_rate=ts.sampling_rate)


def line_broadening(ts, width):
    """
    Apply line-broadening to a time-series

    Parameters
    ----------
    ts : a nitime.TimeSeries class instance

    width : float
        The exponential decay time-constant (in seconds)

    Returns
    -------
    A nitime.TimeSeries class instance with windowed data

    Notes
    -----

    For details, see page 92 of Keeler2005_.

    .. [Keeler2005] Keeler, J (2005). Understanding NMR spectroscopy, second
       edition. Wiley (West Sussex, UK).

    """ 
    # We'll need to broadcast into this many dims:
    n_dims = len(ts.shape)
    # We use the information in the TimeSeries.time attribute to get it:
    win = np.exp(- width * ts.time/np.float(ts.time._conversion_factor))
    
    new_data = ts.data * win
    return nts.TimeSeries(new_data, sampling_rate=ts.sampling_rate)


def freq_to_ppm(f, water_hz=0.0, water_ppm=4.7, hz_per_ppm=127.680):
    """
    Convert a set of numbers from frequenxy in hertz to chemical shift in ppm
    """
    return water_ppm - (f - water_hz)/hz_per_ppm


def ppm_to_freq(ppm, water_hz=0.0, water_ppm=4.7, hz_per_ppm=127.680):
    """
    Convert an array from chemical shift to Hz
    """
    return water_hz + (ppm - water_ppm) * hz_per_ppm

def ppm_idx(f_ppm, lb, ub):
    """
    Create a slice object according to the ppm scale

    Parameters
    ----------
    f_ppm : float array
        The frequency bins (in the ppm scale). Assumed to be descending

    lb,ub : float
        The lower/upper bounds for indexing
        
    Returns
    -------
    idx : slice object which can be used to index 
    """
    idx0 = np.argmin(np.abs(f_ppm - lb))
    idx1 = np.argmin(np.abs(f_ppm - ub))
    return slice(idx1, idx0)    



def phase_correct_zero(spec, phi):
    """
    Correct the phases of a spectrum by phi radians

    Parameters
    ----------
    spec : float array of complex dtype
        The spectrum to be corrected. 
       
    phi : float

    Returns
    -------
    spec : float array
         The phase corrected spectrum

    Notes
    -----
    [Keeler2005] Keeler, J (2005). Understanding NMR Spectroscopy, 2nd
        edition. Wiley. Page 88.  

    """
    c_factor = np.exp(-1j * phi)
    # If it's an array, we need to reshape it and broadcast across the
    # frequency bands in the spectrum. Otherwise, we assume it's a scalar and
    # apply it to all the dimensions of the spec array:
    if hasattr(phi, 'shape'):
        c_factor = c_factor.reshape(c_factor.shape + (1,))

    return spec * c_factor


def phase_correct_first(spec, freq, k):
    """
    First order phase correction.

    Parameters
    ----------
    spec : float array
        The spectrum to be corrected.

    freq : float array
        The frequency axis.

    k : float
        The slope of the phase correction as a function of frequency.

    Returns
    -------
    The phase-corrected spectrum.

    Notes
    -----
    [Keeler2005] Keeler, J (2005). Understanding NMR Spectroscopy, 2nd
        edition. Wiley. Page 88

    """
    c_factor = np.exp(-1j * k * freq)
    c_factor = c_factor.reshape((len(spec.shape) -1) * (1,) + c_factor.shape)
    return spec * c_factor
    
def lorentzian(freq, freq0, area, hwhm, phase, offset, drift):
   """
   Lorentzian line-shape function

   Parameters
   ----------
   freq : float or float array
      The frequencies for which the function is evaluated

   freq0 : float
      The center frequency of the function

   area : float

   hwhm: float
      Half-width at half-max       

   """
   oo2pi = 1/(2*np.pi)
   df = freq - freq0
   absorptive = oo2pi * area * np.ones(freq.shape[0])*(hwhm / (df**2 + hwhm**2))
   dispersive = oo2pi * area * df/(df**2 + hwhm**2)
   return (absorptive * np.cos(phase) + dispersive * np.sin(phase) + offset +
   drift * freq)

def two_lorentzian(freq, freq0_1, freq0_2, area1, area2, hwhm1, hwhm2, phase1,
                   phase2, offset, drift):
   """
   A two-Lorentzian model.

   This is simply the sum of two lorentzian functions in some part of the
   spectrum. Each individual Lorentzian has its own peak frequency, area, hwhm
   and phase, but they share common offset and drift parameters.
   
   """
   return (lorentzian(freq, freq0_1, area1, hwhm1, phase1, offset, drift) +
           lorentzian(freq, freq0_2, area2, hwhm2, phase2, offset, drift))
 

def gaussian(freq, freq0, sigma, amp, offset, drift):
    """
    A Gaussian function with flexible offset, drift and amplitude
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (amp * np.exp(- ((freq - freq0)**2) / (sigma**2) ) +
                drift * freq + offset)

def two_gaussian(freq, freq0_1, freq0_2, sigma1, sigma2, amp1, amp2,
                   offset, drift):
   """
   A two-Gaussian model.

   This is simply the sum of two gaussian functions in some part of the
   spectrum. Each individual gaussian has its own peak frequency, sigma,
   and amp, but they share common offset and drift parameters.

   """
   return (gaussian(freq, freq0_1, sigma1, amp1, offset, drift) +
           gaussian(freq, freq0_2, sigma2, amp2, offset, drift))

def make_idx(f, lb, ub):
    """
    This is a little utility function to replace an oft-called set of
    operations

    Parameters
    ----------
    f : 1d array
        A frequency axis along which we want to slice
       
    lb : float
       Defines the upper bound of slicing

    ub : float
       Defines the lower bound of slicing

    Returns
    -------
    idx : a slice object
    
    """
    
    idx0 = np.argmin(np.abs(f - lb))
    idx1 = np.argmin(np.abs(f - ub))

    # Determine which should be on which side
    if f[0] > f[1]:
        direction = -1
    else:
        direction = 1
        
    if direction == -1:
        idx = slice(idx1, idx0)
    elif direction == 1:
        idx = slice(idx0, idx1)

    return idx


def rmse(arr1, arr2):
    """

    The root of the mean of the squared error between arr1 and arr2

    Parameters
    ----------
    arr1, arr2 : ndarray
       The two arrays to compare

    Returns
    -------
    RMSE : float

    Notes
    -----
    The RMSE will be summarized in one number. That is, even if the two arrays
    are multi-dimensional, the error between them will be averaged over the
    entire array and then the sqrt will be derived.

    """
    return np.sqrt(np.mean((arr1 - arr2)**2))

def detect_outliers(in_arr, thresh=3.0):
    """ 
    Detects outliers more than X standard deviations from mean. 
 
    Parameters 
    ----------   
    in_list: ndarray
        An array of measures for which outliers need to be detected. 
 
    thresh: float (optional)
        Threshold number of standard deviations before a measure is considered
        an outlier. Default = 3.0
 
    Returns 
    ------- 
    outlier_idx: ndarray of boolean  
        An array indicating if a measure is an outlier (True) or not (False). 
    """

    mean = np.nanmean(in_arr)
    std = np.nanstd(in_arr)

    # calculate cutoffs 
    uppthresh = mean + thresh * std
    lowthresh = mean - thresh * std

    return np.logical_or(in_arr < lowthresh, in_arr > uppthresh)


