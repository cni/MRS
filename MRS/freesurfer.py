"""
freesurfer tools
""" 

import os
import re
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl   

def reconall(subjfile,subjID=None,subjdir=None): 
    """
    Carries out Freesurfer's reconall on T1 nifti file
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
    #result = wf.run()

    # convert ribbon.mgz to nii
    convertmgz = pe.Node(interface=fs.MRIConvert(), name='convertmgz')
    convertmgz.inputs.in_file = segdir+'mri/ribbon.mgz'
    convertmgz.inputs.out_orientation='LPS'
    convertmgz.inputs.resample_type= 'nearest'
    convertmgz.inputs.reslice_like= subjfile
    convertmgz.inputs.out_file=segdir+subjID+'_gmwm.nii.gz'
    
    
    wf2 = pe.Workflow(name="convertmgz")
    wf2.base_dir = T1dir

    wf2.add_nodes([convertmgz])
    result2 = wf2.run()
    #return (result, result2)
