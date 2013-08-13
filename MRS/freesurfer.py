"""
freesurfer tools
""" 

import os
import re
import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl   

def reconall(subjfile,subjID=None,subjdir=None): 
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
    return (result, result2)

	
def cubicMaskStats(segfile, center, length=25.0, subjID=None):
    """
    returns grey/white/CSF content within cubic mask

    Parameters
    ----------
    segfile:  nifti file
        path to segmentation file with grey/white matter labels.

    center : integer array
        [x,y,z] where x, y and z are the coordinates of the point of interest
    
    length : float
        length of cube in mm

    subjID: string
        optional subject identifier. Defaults to nims scan number

   
    
    See http://miykael.github.io/nipype-beginner-s-guide/regionOfInterest.html

    """
    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_',subjT1)
        subjID=m.group(0)[:-1] 

    # get segmentation file
    gmwm = nib.load(segfile)
    gmwm_data = gmwm.get_data().squeeze()
    segdir = os.path.dirname(segfile)

    # calculate beginning corner of cubic ROI
    print 'Creating cubic mask with center '+str(center[0])+', '+str(center[1])+', '+str(center[2])+', length '+str(length) + 'mm.'
    halflen = length/2
    corner = [center[0]-halflen,
              center[1]-halflen,
              center[2]-halflen]

    cubemask = pe.Node(interface=fsl.ImageMaths(),name="cubemask")
    cubeValues = (corner[0],length,corner[1],length,corner[2],length)
    cubemask.inputs.op_string = '-mul 0 -add 1 -roi %d %d %d %d %d %d 0 1'%cubeValues
    cubemask.inputs.out_data_type = 'float'
    cubemask.inputs.in_file=segfile	
    cubemask.inputs.out_file=segdir+subjID+'_cubeMask.nii.gz'
    
    mask_wf=pe.Workflow(name="cubemask")
    mask_wf.add_nodes([cubemask])
    mask_wf.run()
    
    # multiply cubic ROI with segfile
    masked=np.multiply(cube_data,gmwm_data)

    # extract stats from a given segmentation
    total = size(nonzero(cube_data==1))
    white = size(nonzero(masked==42)) + size(nonzero(masked==3))
    grey = size(nonzero(masked==41)) + size(nonzero(masked==2))
    other = total - white - grey
    
    # proportions
    pWhite = white/total
    pGrey = grey/total
    pOther = other/total

    return (total, grey, white, other, pGrey, pWhite, pOther)

    
