import numpy.testing as npt
import scipy.io as sio

import mrs

file_name = 'mrs/pure_gaba_P64024.7'

def test_get_header():
    """
    Test that the header we get has the same information as was read by the
    matlab code.
    """
    mat_header = sio.loadmat('header.mat', squeeze_me=True)['header']
    hdr = mrs.get_header(file_name)

    for x in hdr:
        npt.assert_(mat_header[x]==hdr[x], "hdr['%s'] = %s, not %s "%(
            x, hdr[x], mat_header[x]))

def test_get_data():
    """
    Test that the data is the same as the one read using Matlab tools.

    """ 
    matlab_data = sio.loadmat('data.mat', squeeze_me=True)['data']
    mrs_data = mrs.get_data(file_name)

    # There are differences in the data because the numbers are very large, so
    # we check that the differences are minscule in relative terms:
    mat_dat_e8 = matlab_data.squeeze() / 1e8
    mrs_dat_e8 = mrs_data.squeeze() / 1e8
    npt.assert_almost_equal(mrs_dat_e8, mat_dat_e8)
    
