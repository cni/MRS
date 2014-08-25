import MRS
import MRS.qc as qc
import os
from nibabel.tmpdirs import InTemporaryDirectory


data_folder = os.path.join(os.path.join(os.path.expanduser('~'), '.mrs_data'))
ref_file = os.path.join(data_folder, '5182_1_1.nii.gz')
end_file = os.path.join(data_folder, '5182_15_1.nii.gz')


def test_motion():
    """
    test motion detection
    """
    with InTemporaryDirectory() as tmpdir:
        qcres=qc.motioncheck(ref_file, end_file, out_path='.', thres=5.0)


