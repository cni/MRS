import os
import MRS
import MRS.api as api
import os

data_folder = os.path.join(os.path.join(os.path.expanduser('~'), '.mrs_data'))
file_name = os.path.join(data_folder, 'pure_gaba_P64024.nii.gz')

def test_fitting():
    """
    Test fitting functions. This should exercise a lot of the code in the api
    module.
    """
    G = api.GABA(file_name)
    G.fit_creatine()
    G.fit_gaba()
    G.fit_glx()
    G.fit_water()
