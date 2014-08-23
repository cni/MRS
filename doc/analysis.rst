Data analysis in SMAL
---------------------

We provide several examples of data analysis as  `IPython`__ notebooks. These
can be viewed and downloaded `here`__.

__ nbviewer

The simplest form of the analysis is::

    import MRS.api as mrs
    G = mrs.GABA(file_name)
    G.fit_gaba()

Once these lines are executed, the object `G` will now contain several attributes
that quantify the relative abundance of GABA and creatine, which can then be
used in further analysis.
    
.. include:: links_names.inc
