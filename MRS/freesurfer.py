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

def reconall(subjfile,*args): 
    """
    Carries out Freesurfer's reconall on T1 nifti file
    http://nipy.sourceforge.net/nipype/users/examples/smri_freesurfer.html

    Parameters
    ----------
    subjfile: nifti file
        Path to subject's T1 nifti file
    
    args: the directory to where segmentation results should be saved can be enetered as an optional second input parameter. Defaults to same directory as subjfile.  
    """  
    parser = argparse.ArgumentParser()

    T1dir = os.path.dirname(subjfile)
    filename = os.path.basename(subjfile)

    # Tell freesurfer what subjects directory to use
    if len(args)>0:
        subjdir=args[0]
    else:
        subjdir=T1dir
    fs.FSCommand.set_default_subjects_dir(subjdir)
    print 'saving to ' + subjdir

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
    reconall.inputs.subjects_dir = subjdir
    reconall.inputs.T1_files = subjfile

    wf.add_nodes([reconall])
    result = wf.run()


	
def cubicMaskStats(subjectID, center, length, basedir):
    """
    define a cubic mask, returns grey/white/CSF content within mask

    Parameters
    ----------
    subjectID: string
        subject identifier

    center : integer array
        [x,y,z] where x, y and z are the coordinates of the point of interest
    
    length : float
        length of cube in mm

    basedir: directory 
        where segmentation results are, and where results will be stored
    
    See http://miykael.github.io/nipype-beginner-s-guide/regionOfInterest.html

    """

    # get data
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id']),name='datasource')
    datasource.inputs.base_directory=basedir
    

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
    cubemask.inputs.in_file=subjectID+'_*_.nii'	

    # extract stats from a given segmentation
    segstat = pe.Node(interface=fs.SegStats(),name='segstat')

    #Create a datasink node to store important outputs
    datasink = pe.Node(interface=nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = experiment_dir + '/results'
    datasink.inputs.container = fROIOutput
