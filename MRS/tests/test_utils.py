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
    
    
def test_apodize():
    # Complex time series:
    arr = np.random.rand(10,10,10,1024) * np.exp(1j * np.pi)
    ts = nts.TimeSeries(data=arr, sampling_rate=5000.)

    lbr = 1000. 

    for n in [None, 256]:
        new_ts = ut.apodize(ts, lbr, n)
        npt.assert_equal(new_ts.shape, ts.shape)
    
