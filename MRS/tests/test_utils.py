import numpy as np
import numpy.testing as npt
import MRS.utils as ut
import nitime as nts

def test_phase_correct_zero():
    """
    Test that phase correction works 
    """
    # Make some complex numbers, with the same phase (pi):
    arr = np.random.rand(10,10,10,1024) * np.exp(1j * np.pi)
    corr_arr = ut.phase_correct_zero(arr, -np.pi)
    npt.assert_array_almost_equal(np.angle(corr_arr), 0)
    
def test_phase_correct_first():
    """
    Test that phase correction works
    """
    # Make some complex numbers, all with the same phase (pi):
    freqs = np.linspace(1, np.pi, 1024)
    arr = np.random.rand(10,10,10,1024) * np.exp(1j * np.pi)
    corr_arr = ut.phase_correct_first(arr, freqs, 1)

    
def test_line_broadening():
    # Complex time series:
    arr = np.random.rand(10,10,10,1024) * np.exp(1j * np.pi)
    ts = nts.TimeSeries(data=arr, sampling_rate=5000.)
    lbr = 1000. 
    new_ts = ut.line_broadening(ts, lbr)
    npt.assert_equal(new_ts.shape, ts.shape)


def test_lorentzian():
    """
    Test the lorentzian function
    """
    freq = np.linspace(0,6,100)
    freq0 = 3
    area = 1
    hwhm = 1
    phase = np.pi
    baseline0 = 1
    baseline1 = 0
    ut.lorentzian(freq, freq0, area, hwhm, phase, baseline0, baseline1)
    
def test_detect_outliers():
    """
    Test that outlier detection works.
    """
    # Make random numbers, with at least two outliers
    b = np.random.rand(1000)
    st = np.std(b)
    b[0] = b[0] + 5 * st
    b[1] = b[1] - 5 * st
    thresh = 2.5
    idx = ut.detect_outliers(b, thresh)
    
    npt.assert_(sum(idx) >= 2)
     
