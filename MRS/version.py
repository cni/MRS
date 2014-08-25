"""MRS version/release information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "MRS analysis software"

long_description = """

The Stanford CNI MRS Analysis Library
=====================================
This library provides algorithms and methods to read and analyze data from
Magnetic Resonance Spectroscopy (MRS) experiments. It provides an API for
fitting models of the spectral line-widths of several different molecular
species, and quantify their relative abundance in human brain tissue.


Copyright (c) 2013-, Ariel Rokem, Grace Tang.

The Center for Neurobiological and Cognitive Imaging, Stanford University.

All rights reserved.

"""

NAME = "MRS"
MAINTAINER = "Ariel Rokem"
MAINTAINER_EMAIL = "arokem@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/arokem/MRS"
DOWNLOAD_URL = "http://github.com/arokem/MRS"
LICENSE = "GPL"
AUTHOR = "Ariel Rokem, Grace Tang"
AUTHOR_EMAIL = "arokem@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['MRS',
            'MRS.api',
            'MRS.analysis',
            'MRS.utils',
            'MRS.freesurfer',
            'MRS.qc']
            
PACKAGE_DATA = {"MRS": ["LICENSE"]}

REQUIRES = ["numpy", "matplotlib", "scipy"]
