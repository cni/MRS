# The Stanford MRS Analysis Library (SMAL)

Analysis of data from MR spectroscopy experiments.

For usage, please refer to the [online documentation](http://cni.github.io/MRS). 


## Installation

### Released versions
Using SMAL requires the following scientific python software packages: numpy,
scipy, matplotlib. The easiest way to install these for most platforms is using
the free [Anaconda](http://continuum.io/downloads) or
[Canopy](https://www.enthought.com/products/canopy/) software distributions.

In addition, the software requires the nibabel and nitime neuroimaging
libraries. These are most easily installed using `pip`. As in:

    pip install nibabel
    pip install nipype

Additional, optional, requirements (used in some functions) are nipype,
together with FSL and Freesurfer, as well as IPython (which is used to run the
examples). 

### From source

To install from source make sure that you have all the dependencies installed,
then clone this repository and run:

    python setup.py install

## Usage

mrs-analysis: This is a little command-line utility that takes nifti files as
input and produces a `csv` file as output, with the ppm scale and the spectra
from the two measurement echos. See `mrs-analyze --help` for help. 

## License

See `LICENSE` for details. 

(C) Ariel Rokem, Grace Tang, Center for Cognitive and Neurobiological Imaging,
Stanford University 2013
