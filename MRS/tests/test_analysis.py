import os

import numpy as np
import numpy.testing as npt
import matplotlib
matplotlib.use('agg')

import nitime as nt
import nibabel as nib

import MRS
import MRS.utils as ut
import MRS.analysis as ana
import MRS.data as mrd

data_folder = os.path.join(os.path.join(os.path.expanduser('~'), '.mrs_data'))
file_name = os.path.join(data_folder, 'pure_gaba_P64024.nii.gz')

def test_separate_signals():
    """
    Test separation of signals for water-suppressed and other signals
    """
    data = np.transpose(nib.load(file_name).get_data(), [1,2,3,4,5,0]).squeeze()
    w_data, w_supp_data = ana.separate_signals(data)
    # Very simple sanity checks
    npt.assert_equal(w_data.shape[-1], data.shape[-1])
    npt.assert_equal(w_supp_data.shape[-1], data.shape[-1])
    npt.assert_array_equal(data[1], w_data[0])
    
def test_coil_combine():
    """
    Test combining of information from different coils
    """
    data = np.transpose(nib.load(file_name).get_data(), [1,2,3,4,5,0]).squeeze()
    w_data, w_supp_data = ana.coil_combine(data)
    # Make sure that the time-dimension is still correct: 
    npt.assert_equal(w_data.shape[-1], data.shape[-1])
    npt.assert_equal(w_supp_data.shape[-1], data.shape[-1])

    # Check that the phase for the first data point is approximately the
    # same and approximately 0 for all the water-channels:
    npt.assert_array_almost_equal(np.angle(w_data)[:,:,0],
                                  np.zeros_like(w_data[:,:,0]),
                                  decimal=1)  # We're not being awfully strict
                                              # about it.
        
def test_get_spectra():
    """
    Test the function that does the spectral analysis
    """
    data = np.transpose(nib.load(file_name).get_data(), [1,2,3,4,5,0]).squeeze()
    w_data, w_supp_data = ana.coil_combine(data)

    # XXX Just basic smoke-testing for now:
    f_nonw, nonw_sig1 = ana.get_spectra(nt.TimeSeries(w_supp_data,
                                       sampling_rate=5000))

    f_nonw, nonw_sig2 = ana.get_spectra(nt.TimeSeries(w_supp_data,
                                       sampling_rate=5000),
                                       line_broadening=5)

    f_nonw, nonw_sig3 = ana.get_spectra(nt.TimeSeries(w_supp_data,
                                       sampling_rate=5000),
                                       line_broadening=5,
                                       zerofill=1000)
    
def test_bootstrap_stat():
    """
    Test simple bootstrapping statistics
    """

    rand_array = np.random.randn(100, 1000)
    arr_mean, mean_ci = ana.bootstrap_stat(rand_array)
    npt.assert_array_equal(np.mean(rand_array, 0), arr_mean)

    arr_var, var_ci = ana.bootstrap_stat(rand_array, stat=np.var)
    npt.assert_array_equal(np.var(rand_array, 0), arr_var)
