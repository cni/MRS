import os
import MRS
import MRS.qc as qc

test_path = os.path.join(MRS.__path__[0], 'tests')
ref_file = os.path.join(test_path, '5182_1_1.nii.gz')
end_file = os.path.join(test_path, '5182_15_1.nii.gz')

def test_motion():
    """
    test motion detection
    """
    qcres=qc.motioncheck(ref_file,end_file, thres=5.0)
