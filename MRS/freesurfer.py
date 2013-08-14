"""
freesurfer tools
""" 

import os
import re
import numpy as np
import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl   

def reconall(subjfile,subjID=None,subjdir=None, runreconall=True): 
    """
    Carries out Freesurfer's reconall on T1 nifti file
    
    WARNING: Reconall takes very long to run!!

    http://nipy.sourceforge.net/nipype/users/examples/smri_freesurfer.html

    Parameters
    ----------
    subjfile: nifti file
        Path to subject's T1 nifti file
    
    subjID: string
        optional name for subject's output folder

    subjdir: the directory to where segmentation results should be saved. Defaults to same directory as subjfile.  
    runreconall: boolean
        If set to true, runs reconall, otherwise just converts assorted mgz files to nii
    """  

    T1dir = os.path.dirname(subjfile)
    filename = os.path.basename(subjfile)

    # Tell freesurfer what subjects directory to use
    if subjdir==None:
        subjdir=T1dir
    fs.FSCommand.set_default_subjects_dir(subjdir)
    segdir=subjdir+'/'+subjID+'/'
    print 'saving to ' + subjdir

    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_',subjfile)
        subjID=m.group(0) + 'seg'        
        

    # check if file exists
    if os.path.isfile(subjfile):
        print 'running recon-all on ' + filename
    else:
        raise ValueError("File: %s does not exist!"%filename)

    # check if nifti format
    ext = os.path.splitext(filename)[-1].lower()
    if ext != ".nii":
        raise ValueError("File: %s is not a nifti file!"%filename)

    wf = pe.Workflow(name="segment")
    wf.base_dir = T1dir

    if runreconall:
        # run recon-all
        reconall = pe.Node(interface=fs.ReconAll(), name='reconall')
        reconall.inputs.subject_id = subjID 
        reconall.inputs.directive = 'all'
        reconall.inputs.subjects_dir = subjdir
        reconall.inputs.T1_files = subjfile
    
        wf.add_nodes([reconall])
        result = wf.run()

    wf2 = pe.Workflow(name="convertmgz")
    wf2.base_dir = T1dir

    # convert ribbon.mgz to nii
    convertmgz = pe.Node(interface=fs.MRIConvert(), name='convertmgz')
    convertmgz.inputs.in_file = segdir+'mri/ribbon.mgz'
    convertmgz.inputs.out_orientation='LPS'
    convertmgz.inputs.resample_type= 'nearest'
    convertmgz.inputs.reslice_like= subjfile
    convertmgz.inputs.out_file=segdir+subjID+'_gmwm.nii.gz'

    wf2.add_nodes([convertmgz])
    result2 = wf2.run()

    wf3 = pe.Workflow(name="convertmgz2")
    wf3.base_dir = T1dir

    convertmgz2 = pe.Node(interface=fs.MRIConvert(), name='convertmgz2')
    convertmgz2.inputs.in_file = segdir+'mri/aseg.auto.mgz'
    convertmgz2.inputs.out_orientation='LPS'
    convertmgz2.inputs.resample_type= 'nearest'
    convertmgz2.inputs.reslice_like= subjfile
    convertmgz2.inputs.out_file=segdir+subjID+'_aseg.nii.gz'

    wf3.add_nodes([convertmgz2])
    result3 = wf3.run()
    if runreconall:
        return (result, result2, result3)
    else:
        return (result2,result3)

def MRSvoxelStats(segfile, pfile=None, center=None, dim=None, subjID=None, gareas=[3,42,11,12,13,26,50,51,52,58,9,10,48,49],wareas=[2,41],csfareas=[4,5,14,15,24,43,44,72]):
    """
    returns grey/white/CSF content within MRS voxel

    Parameters
    ----------
    segfile:  nifti file
        path to segmentation file with grey/white matter labels.

    pfile: nifti file
        path to pfile of MRS voxel. provide either this or center + dim

    center : integer array
        [x,y,z] where x, y and z are the coordinates of the point of interest. Provide either pfile or center+dim
    
    dim : float array
        dimensions of voxel in mm. Provide either pfile or center+dim

    subjID: string
        optional subject identifier. Defaults to nims scan number

    gareas, wareas, csfareas: arrays of integers
        arrays of freesurfer labels for gray, white, and csf areas respectively.
    
    See http://miykael.github.io/nipype-beginner-s-guide/regionOfInterest.html

    """
    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_',segfile)
        subjID=m.group(0)[:-1] 

    # get segmentation file
    gmwm = nib.load(segfile)
    gmwm_data = gmwm.get_data().squeeze()
    gmwm_aff = gmwm.get_affine()
    segdir = os.path.dirname(segfile)

    # get pfile of MRS voxel if one is provided
    if pfile!=None:
        if center != None or dim != None:
            raise ValueError('provide EITHER pfile OR center and dim, not both!')
        mrs = nib.load(pfile)
        mrs_data = mrs.get_data().squeeze()
        mrs_aff = mrs.get_affine()
        tmp=mrs_aff.copy()
        mrs_aff[0,3]=tmp[1,3]
        mrs_aff[1,3]=-1.0*tmp[0,3]
        center = np.round(np.dot(np.dot(np.linalg.pinv(gmwm_aff), mrs_aff), [0,0,0,1]))[:3].astype(int)

        dim=np.diagonal(mrs_aff)[:3]
    else: # no pfile
        if center==None or dim==None:
            raise ValueError('if no pfile is provided, provide center and dimensions of voxel')    

    # calculate beginning corner of MRS voxel
    print 'Creating mask with center '+str(center[0])+', '+str(center[1])+', '+str(center[2])+', dimensions '+str(dim[0]) + ', '+str(dim[1]) +', '+str(dim[2]) +'mm.'

    corner = [center[0]-dim[0]/2,
              center[1]-dim[1]/2,
              center[2]-dim[2]/2]

    ROImask = pe.Node(interface=fsl.ImageMaths(),name="ROImask")
    ROIValues = (corner[0],dim[0],corner[1],dim[1],corner[2],dim[2])
    ROImask.inputs.op_string = '-mul 0 -add 1 -roi %d %d %d %d %d %d 0 1'%ROIValues
    ROImask.inputs.out_data_type = 'float'
    ROImask.inputs.in_file=segfile	
    ROImask.inputs.out_file=segdir+'/'+subjID+'_ROIMask.nii.gz'
    
    mask_wf=pe.Workflow(name="ROImask")
    mask_wf.add_nodes([ROImask])
    mask_wf.run()
    
    # multiply voxel ROI with segfile
    roifile=segdir+'/'+subjID+'_ROIMask.nii.gz'
    roi = nib.load(roifile)
    roi_data = roi.get_data().squeeze()

    masked=np.multiply(roi_data,gmwm_data)

    # extract stats from a given segmentation
    total = np.size(np.nonzero(roi_data==1))
    white = np.size(np.nonzero(masked==42)) + np.size(np.nonzero(masked==3))
    grey = np.size(np.nonzero(masked==41)) + np.size(np.nonzero(masked==2))
    other = total - white - grey
    
    # proportions
    pWhite = 1.0*white/total
    pGrey = 1.0*grey/total
    pOther = 1.0*other/total

    return (total, grey, white, other, pGrey, pWhite, pOther)

    
