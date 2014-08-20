import MRS
import MRS.qc as qc
import os
from nibabel.tmpdirs import InTemporaryDirectory

test_path = os.path.join(MRS.__path__[0], 'tests')
ref_file = os.path.join(test_path, '5182_1_1.nii.gz')
end_file = os.path.join(test_path, '5182_15_1.nii.gz')


def test_motion():
    """
    test motion detection
    """
    with InTemporaryDirectory() as tmpdir:
        qcres=qc.motioncheck(ref_file, end_file, out_path='.', thres=5.0)


