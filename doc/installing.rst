
Installing SMAL
---------------------

Dependencies
~~~~~~~~~~~~~~~~~~~~~

To install `SMAL` and start using it, you will first need to make sure that you have the
dependencies installed.

`SMAL` was developed using Python 2.7. In addition, it relies on code from several
other libraries:

- Scientific Python libraries:

    - Scipy_
    - Numpy_
    - Matplotlib_

 The easiest way to get all of these installed is using the Anaconda_ or
 Canopy_ installers

- Neuroimaging libraries:

    - Nibabel_
    - Nipy_
    - Nipype_
    - Nitime_

Follow the instructions on the respective websites for installation instructions.

Getting the `SMAL` code
~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have downloaded and installed these dependencies, you can go ahead and
install `SMAL`. The source code can be downloaded from our `github`__
repository. The most recent cutting edge version of the code can be downloaded
in `zip`__ or `tar`__ formats. A stable release version is availabe to dowload `here`__  

__ mrs-github_
__ download-zip_
__ download-tar_
__ github-releases


Installing
~~~~~~~~~~~~~~~~~~~~~

Either way, once the code is downloaded, it can be installed issuing the
following command on the shell command-line in the top level directory of the
source-code download::

    python setup.py install

    
.. include:: links_names.inc
