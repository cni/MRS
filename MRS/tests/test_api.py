import os
import MRS
import MRS.api as api

test_path = os.path.join(MRS.__path__[0], 'tests')
file_name = os.path.join(test_path, 'pure_gaba_P64024.nii.gz')



def test_fitting():
    """
    Test fitting of Gaussian function to creatine peak
    """
    G = api.GABA(file_name)
    G.fit_gaba()
