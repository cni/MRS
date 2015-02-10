"""
quality control for MRS data
""" 
import os
import os.path as op
import nibabel as nib
import numpy as np
import nipype.pipeline.engine as pe
from nipype.interfaces import fsl

def motioncheck(ref_file, end_file, out_path=None, thres=5.0):
    """
    Checks motion between structural scans of the same modality. 
    Ideally obtained at the beginning and end of a scanning session.

    Parameters
    ----------
    ref_file: nifti file 
        Nifti file of first localizer acquired at the beginning of the session

    end_file: nifti
        nifti file of the localizer acquired at the end of the session	

    thres: float
        threshold in mm of maximum allowed motion. Default 5mm

    Returns
    -------
    rms : float
        root mean square of xyz translation

    passed: boolean
        indicates if motion passed threshold: 1 if passed, 0 if failed.
    """        

    ref = nib.load(ref_file)
    end = nib.load(end_file)
    ref_data = ref.get_data()
    end_data = end.get_data()    

    # Check if same affine space. modality must be the same to use realign, 
    # and prescription must be the same to deduce motion
    ref_aff=ref.get_affine()
    end_aff=end.get_affine()

    if np.array_equal(ref_aff, end_aff):
        print('affines match')
    else:
        raise ValueError("Affines of start and end images do not match")

    # save only axials
    refax = ref_data[:, :, :, 0, np.newaxis]
    endax = end_data[:, :, :, 0, np.newaxis]

    if out_path is None:
        path = os.path.dirname(ref_file)
    
    refax_img = nib.Nifti1Image(refax, ref_aff)
    nib.save(refax_img, op.join(out_path, 'refax.nii.gz'))
    endax_img = nib.Nifti1Image(endax, ref_aff)
    nib.save(endax_img, op.join(out_path, 'endax.nii.gz'))

    # realignment
    ref_file = op.join(out_path, 'refax.nii.gz')
    in_file = op.join(out_path, 'endax.nii.gz')
    mat_file = op.join(out_path, 'mat.nii.gz')

    mcflt = fsl.MCFLIRT(in_file=in_file, ref_file=ref_file, save_mats=True,
                        cost='mutualinfo')
    res = mcflt.run() 

    print('realignment affine matrix saved in mat_file: %s'
          %res.outputs.mat_file)

    aff_file=res.outputs.mat_file
    aff = np.loadtxt(aff_file, dtype=float)

    # compute RMS as indicator of motion
    rel=aff[0:3, 3]
    rms = np.sqrt(np.mean(rel**2))
  
    if rms>=thres:
        passed=False
    else:
        passed=True


    return rms, passed

