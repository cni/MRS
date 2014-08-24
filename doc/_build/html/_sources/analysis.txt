Data analysis in SMAL
---------------------

Programming with SMAL:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The simplest form of the analysis is in a python session (or script)::

    import MRS.api as mrs
    G = mrs.GABA(file_name)
    G.fit_gaba()

Where `file_name` is a variable which includes the full path to a nifti file
containing data organized as specified in ':ref:`data`'. Once these lines are
executed, the object `G` will now contain several attributes that quantify the
relative abundance of GABA and creatine, which can then be used in further
analysis.

We provide several examples of data analysis as  `IPython`_ notebooks. These
can be viewed and downloaded `here`__.

__ nbviewer_


Command line interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A command line interface (CLI) is available to conduct basic analysis of MRS
data stored in a nifti files following the specification in ':ref:`data`'.

To run this interface run the following line in a shell session::

    mrs-analyze.py ~/.mrs_data/12_1_PROBE_MEGA_L_Occ.nii.gz --plot True --out_file ~/tmp/mrs.csv 


Where you can replace the full-path to the input file with the file you are
analyzing. This command will analyze the data, produce a plot with the sum
spectra (echo on + echo off), the difference spectra (echo on - echo off), and
the fit of the creatine and GABA models to these data, together with their
calculated areas under the curves. If you do not want to produce plots, exclude
the `--plot True`.  In addition, it would save a file under `~/tmp/mrs.csv`
with the frequency bands (in ppm) and the spectra (echo on, echo off and
difference) as a function of frequency band. You can use this file for further
analysis (e.g. using other modeling techniques).

To explore all the options in the CLI run the following line
in a shell session::

    mrs-analyze.py
    
This will print a list of the input options to the CLI.

.. include:: links_names.inc
