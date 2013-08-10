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

	
def cubicMaskStats(subjT1, center, length=25.0, subjID=None, segdir=None):
    """
    define a cubic mask, returns grey/white/CSF content within mask

    Parameters
    ----------
    subjT1: nifti file
        path to subject's T1 file

    center : integer array
        [x,y,z] where x, y and z are the coordinates of the point of interest
    
    length : float
        length of cube in mm

    subjID: string
        optional subject identifier. Defaults to nims scan number

    segdir: directory 
        where segmentation results are, and where results will be stored. Defaults to directory where T1 is.
    
    See http://miykael.github.io/nipype-beginner-s-guide/regionOfInterest.html

    """
    # subject ID
    if subjID==None:
        m=re.search('(\w+?)_*_',subjT1)
        subjID=m.group(0)[:-1] 


    # get data
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id']),name='datasource')
    datasource.inputs.base_directory=segdir
    

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
    cubemask.inputs.in_file=subjT1	
    cubemask.inputs.out_file=segdir+subjID+'_cubeMask.nii.gz'
    
    mask_wf=pe.Workflow(name="cubemask")
    mask_wf.add_nodes([cubemask])
    mask_wf.run()
    
    # mask the ROI with a subject specific T-map, create seg file for seg stats
#    tmapmask = pe.Node(interface=fsl.ImageMaths(),name="tmapmask")
#    tmapmask.inputs.out_data_type = 'float'
#    tmapmask.inputs.in_file = segdir+subjID+'_cubeMask.nii.gz'
#    tmapmask.inputs.out_file = segdir+subjID+'_tmap.nii.gz'
#
#    tmap_wf=pe.Workflow(name="tmapmask")
#    tmap_wf.add_nodes([tmapmask])
#    tmap_wf.run()

    


#    # extract stats from a given segmentation
#    segstat = pe.Node(interface=fs.SegStats(),name='segstat')
#    segstat.inputs.in_file=subjT1 # use segmentation to report stats on this volume
#    segstat.inputs.brain_vol='brainmask'
#    segstat.inputs.mask_file=segdir+subjID+'_cubeMask.nii.gz'
#    segstat.inputs.summary_file=segdir+subjID+'_segStatsSummary.stats'
#    segstat.inputs.segmentation_file=segdir + subjID+'_tmap.nii.gz'
#    #segstat.inputs.args='--surf-ctxgmwm' # gray and white matter volumes?
## aseg.stats has info on gray/white matter volume
#
#    seg_wf=pe.Workflow(name="seg")
#    seg_wf.add_nodes([segstat])
#    seg_wf.run()

    #Create a datasink node to store important outputs
  #  datasink = pe.Node(interface=nio.DataSink(), name="datasink")
  #  datasink.inputs.base_directory = segdir + '/maskStats'
    #datasink.inputs.container = subjID+'_maskStats'

    # begin workflow
  #  wf = pe.Workflow(name='ROIflow')
  #  wf.base_dir = segdir + '/maskStats'

  #  wf.connect([(cubemask,segstat,['out_file','segmentation_file']),
   #             (segstat,datasink,['summary_file','statistic']),
    #           ])
 #   results=wf.run()

#    return results
