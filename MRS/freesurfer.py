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


def reconall(subjfile, subjID=None, subjdir=None, runreconall=True): 
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

    subjdir: string

        The directory to where segmentation results should be saved. Defaults
        to same directory as subjfile.

    runreconall: boolean
        If set to true, runs reconall, otherwise just converts assorted mgz
        files to nii 
    """  
    T1dir = os.path.dirname(subjfile)
    filename = os.path.basename(subjfile)

    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_', subjfile)
        subjID=m.group(0) + 'seg'        

    # Tell freesurfer what subjects directory to use
    if subjdir==None:
        subjdir=T1dir
    fs.FSCommand.set_default_subjects_dir(subjdir)
    segdir=subjdir+'/'+subjID+'/'
    print('saving to ' + subjdir)

    # check if file exists
    if os.path.isfile(subjfile):
        print('running recon-all on ' + filename)
    else:
        raise ValueError("File: %s does not exist!"%filename)

    # check if nifti format
    ext=filename.split('.')[1].lower()
    if ext != "nii":
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

    # convert mgz to nii
    wf2 = pe.Workflow(name="convertmgz")
    wf2.base_dir = T1dir

    convertmgz = pe.Node(interface=fs.MRIConvert(), name='convertmgz')
    convertmgz.inputs.in_file = segdir+'mri/aseg.auto.mgz'
    convertmgz.inputs.out_orientation='LPS'
    convertmgz.inputs.resample_type= 'nearest'
    convertmgz.inputs.reslice_like= subjfile
    convertmgz.inputs.out_file=segdir+subjID+'_aseg.nii.gz'

    wf2.add_nodes([convertmgz])
    result2 = wf2.run()
    if runreconall:
        return (result, result2)
    else:
        return (result2)


GAREAS = [3,8,42,17,18,53,54,11,12,13,26,50,51,52,58,9,10,47,48,49,16,28,60]
WAREAS = [2,7,41,46]
CSFAREAS = [4,5,14,15,24,43,44,72]


def MRSvoxelStats(segfile, MRSfile=None, center=None, dim=None, subjID=None,
                  gareas=GAREAS,wareas=WAREAS,csfareas=CSFAREAS):
    """
    returns grey/white/CSF content within MRS voxel

    Parameters
    ----------
    segfile:  nifti file
        path to segmentation file with grey/white matter labels (freesurfer
        aseg file converted from mgz). 

    MRSfile: nifti file
        path to MRSfile of MRS voxel. provide either this or center + dim

    center : integer array
        [x,y,z] where x, y and z are the coordinates of the point of
        interest. Provide either MRSfile or center+dim 
    
    dim : float array
        dimensions of voxel in mm. Provide either MRSfile or center+dim

    subjID: string
        optional subject identifier. Defaults to nims scan number

    gareas, wareas, csfareas: arrays of integers
        arrays of freesurfer labels for gray, white, and csf areas
        respectively. Determines which areas are considered
        grey/white/csf. Some areas like brainstem were lumped with grey
        matter. Did not include hypointensities. 
   
    Returns
    -------

    raw number of voxels and proportions of grey, white, csf and
    non-grey-or-white-matter 

    Notes
    -----
    See http://miykael.github.io/nipype-beginner-s-guide/regionOfInterest.html

    """
    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_',segfile)
        subjID=m.group(0)[:-1] 

    # get segmentation file
    aseg = nib.load(segfile)
    aseg_data = aseg.get_data().squeeze()
    aseg_aff = aseg.get_affine()
    segdir = os.path.dirname(segfile)
    segvoxdim=np.diagonal(aseg_aff)[:3]

    # get nifti file of MRS voxel if one is provided
    if MRSfile is not None:
        if center is not None or dim is not None:
            msg = 'provide EITHER MRSfile OR center and dim, not both!'
            raise ValueError(msg)

        mrs = nib.load(MRSfile)
        mrs_data = mrs.get_data().squeeze()
        mrs_aff = mrs.get_affine()        
        # This applies the concatenation of the transforms from mrs space to
        # the T1 space. [0,0,0] is the center of the MRS voxel:
        center = np.round(np.dot(np.dot(np.linalg.pinv(aseg_aff), mrs_aff),
                                 [0,0,0,1]))[:3].astype(int)
        
        dim=np.diagonal(mrs_aff)[:3]
    else: # no MRSfile
        if center==None or dim==None:
            msg = 'if no MRSfile is provided, provide center and '
            msg += 'dimensions of voxel'
            raise ValueError()    

    # calculate beginning corner of MRS voxel
    print('Creating mask with center: [%s, %s, %s]'%(center[0],
                                                    center[1],
                                                    str(center[2])))

    print('and dimensions: [%s, %s, %s] mm'%(dim[0], dim[1], dim[2]))
    
    # calculate roi mask with numpy
    # round up to nearest number of voxel units
    voxdim=np.zeros(3)
    voxdim[0]=np.ceil(np.abs(dim[0] / segvoxdim[0]))
    voxdim[1]=np.ceil(np.abs(dim[1] / segvoxdim[1]))
    voxdim[2]=np.ceil(np.abs(dim[2] / segvoxdim[2]))

    print ('MRS voxel dimensions in T1 voxel units: [%s, %s, %s]'%(voxdim[0],
                                                                  voxdim[1],
                                                                  voxdim[2]))

    # corners
    lcorner = [center[0] - voxdim[0] / 2,
               center[1] - voxdim[1] / 2,
               center[2] - voxdim[2] / 2]

    ucorner = [center[0] + voxdim[0] / 2,
               center[1] + voxdim[1] / 2,
               center[2] + voxdim[2] / 2]
    # create mask
    mdata = np.zeros(aseg_data.shape)
    for i in range(int(lcorner[0]),int(ucorner[0])):
        for j in range(int(lcorner[1]),int(ucorner[1])):
            for k in range(int(lcorner[2]),int(ucorner[2])):
                mdata[i,j,k]=1


    # calculate grey/white/csf from freesurfer labels
    gdata = np.zeros(aseg_data.shape)
    wdata = np.zeros(aseg_data.shape)
    csfdata = np.zeros(aseg_data.shape)

    for data, areas in zip([gdata, wdata, csfdata], [gareas, wareas, csfareas]):
        for area in areas:
            data[np.where(aseg_data==area)] = 1

    # multiply voxel ROI with seg data
    gmasked= mdata * gdata
    wmasked= mdata * wdata
    csfmasked= mdata * csfdata
    
    # extract stats from a given segmentation
    total = np.sum(mdata)
    white = np.sum(wmasked)
    grey = np.sum(gmasked)
    csf = np.sum(csfmasked)
    nongmwm = total - grey - white

    # proportions
    pWhite = float(white) / total
    pGrey = float(grey) / total
    pCSF = float(csf) / total 
    pNongmwm = float(nongmwm) / total

    return (total, grey, white, csf,nongmwm, pGrey, pWhite,pCSF, pNongmwm)
