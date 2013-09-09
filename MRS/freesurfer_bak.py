"""
freesurfer tools
""" 

import os
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl   

def reconall(subjfile):
    """
    Carries out Freesurfer's reconall on T1 nifti file
    http://nipy.sourceforge.net/nipype/users/examples/smri_freesurfer.html

    Parameters
    ----------
    subjfile: nifti file
        Path to subject's T1 nifti file
  
        http://nipy.sourceforge.net/nipype/interfaces/generated/nipype.interfaces.freesurfer.preprocess.html#reconall
    """  
    parser = argparse.ArgumentParser()
	
    T1dir = os.path.dirname(subjfile)
    filename = os.path.basename(subjfile)

    # Tell freesurfer what subjects directory to use
    subjects_dir = T1dir
    fs.FSCommand.set_default_subjects_dir(subjects_dir)

    # check if file exists
    if os.path.isfile(subjfile):
        print 'running recon-all on ' + filename
    else:
        parser.error('File doesn\'t exist')

    # check if nifti format
    ext = os.path.splitext(filename)[-1].lower()
    if ext != ".nii":
        parser.error('File needs to be nifti format.')
	
    wf = pe.Workflow(name="segment")
    wf.base_dir = T1dir
	
    # run recon-all
    reconall = pe.Node(interface=fs.ReconAll(), name='reconall')
    reconall.inputs.subject_id = filename[:4]
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = T1dir
    reconall.inputs.T1_files = subjfile

    wf.add_nodes([reconall])
    result = wf.run()

	
def cubicMask(center, length)
    """
    define a cubic mask
    
    Parameters
    ----------
    center : [x,y,z] where x, y and z are the coordinates of the point of interest
    length : length of cube
    """	

    # calculate beginning corner of cubic ROI
    halflen = length/2
    corner = [center[0]-halflen,
	      center[1]-halflen,
	      center[2]-halflen]

    cubemask = pe.Node(interface=fsl.ImageMaths(),name="cubemask")
    cubeValues = (corner[0],length,corner[1],length,corner[2],length)
    cubemask.inputs.op_string = '-mul 0 -add 1 -roi %d %d %d %d %d %d 0 1'%cubeValues
    cubemask.inputs.out_data_type = 'float'
