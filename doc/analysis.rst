Data analysis in SMAL
---------------------

We provide several examples of data analysis as  `IPython`_ notebooks. These
can be viewed and downloaded `here`__.

__ nbviewer_


The simplest form of the analysis is::

    import MRS.api as mrs
    G = mrs.GABA(file_name)
    G.fit_gaba()

Where `file_name` is a variable which includes the full path to a nifti file
containing data organized as specified in ':ref:`data`'. Once these lines are
executed, the object `G` will now contain several attributes that quantify the
relative abundance of GABA and creatine, which can then be used in further
analysis.


.. include:: links_names.inc
