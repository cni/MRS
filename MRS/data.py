from __future__ import division, print_function, absolute_import

import os
import sys
import textwrap
import contextlib

if sys.version_info[0] < 3:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen

from os.path import join as pjoin
from hashlib import md5
from shutil import copyfileobj

import numpy as np
import nibabel as nib

import zipfile

class FetcherError(Exception):
    pass

import MRS

data_folder = os.path.join(pjoin(os.path.expanduser('~'), '.mrs_data'))

def _log(msg):
    print(msg)


def fetch_data(files, folder):
    """Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files : dictionary
        For each file in `files` the value should be (url, md5). The file will
        be downloaded from url if the file does not already exist or if the
        file exists but the md5 checksum does not match.
    folder : str
        The directory where to save the file, the directory will be created if
        it does not already exist.

    Raises
    ------
    FetcherError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.

    """
    if not os.path.exists(folder):
        _log("Creating new folder %s" % (folder))
        os.makedirs(folder)

    all_skip = True
    for f in files:
        url, md5 = files[f]
        fullpath = pjoin(folder, f)
        if os.path.exists(fullpath) and (_get_file_md5(fullpath) == md5):
            continue
        all_skip = False
        _log('Downloading "%s" to %s' % (f, folder))
        _get_file_data(fullpath, url)
        if _get_file_md5(fullpath) != md5:
            msg = """The downloaded file, %s, does not have the expected md5
checksum of "%s". This could mean that that something is wrong with the file or
that the upstream file has been updated.""" % (fullpath, md5)
            msg = textwrap.fill(msg)
            raise FetcherError(msg)

    if all_skip:
        _log("All files already in %s." % (folder))
    else:
        _log("Files successfully downloaded to %s" % (folder))

def _get_file_md5(filename):
    """Compute the md5 checksum of a file"""
    md5_data = md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128*md5_data.block_size), b''):
            md5_data.update(chunk)
    return md5_data.hexdigest()


def check_md5(filename, stored_md5):
    """
    Computes the md5 of filename and check if it matches with the supplied
    string md5 

    Input
    -----
    filename : string
        Path to a file.
    md5 : string
        Known md5 of filename to check against.

    """
    computed_md5 = _get_file_md5(filename)
    if stored_md5 != computed_md5:
        print ("MD5 checksum of filename", filename,
               "failed. Expected MD5 was", stored_md5,
               "but computed MD5 was", computed_md5, '\n',
               "Please check if the data has been downloaded correctly or if the upstream data has changed.")

def _get_file_data(fname, url):
    with contextlib.closing(urlopen(url)) as opener:
        with open(fname, 'wb') as data:
            copyfileobj(opener, data)


def fetch_from_sdr(folder=data_folder, data='test'):
    """
    Download MRS data from SDR

    Parameters
    ----------
    folder : str
        Full path to a location in which to place the data. Per default this
    will be a directory under the user's home `.mrs_data`.

    data : str
       Which data to download. Either 'test', which is data required for
       testing, or 'example', which is data needed for the example notebooks.
       
    """
    url = "https://stacks.stanford.edu/file/druid:fn662rv4961/"

    if data == 'test':
        md5_dict = {'5182_1_1.nii.gz': '0656e59818538baa7d45311f2581bb4e',
                '5182_15_1.nii.gz': 'a5a307b581620184baf868cd0df81f89',
                'data.mat': 'a6275698f2220c65994354d412e6d82e',
                'pure_gaba_P64024.nii.gz': 'f3e09ec0f00bd9a03910b19bfe731afb'}

    elif data == 'example':
        md5_dict = {'12_1_PROBE_MEGA_L_Occ.nii.gz':
                    'a0571606c1caa16a9d9b00847771bc94',
                     '5062_2_1.nii.gz':
                    '6f77fb5134bc2841bdfc954390f0f4a4'}
        
    if not os.path.exists(folder):
        print('Creating new directory %s' % folder)
        os.makedirs(folder)
        
    for k, v in md5_dict.items():
        fname = pjoin(folder, k)
        if not os.path.exists(fname):
            print('Downloading %s from SDR ...'%k)
            _get_file_data(fname, url + k)
            check_md5(fname, v)
        else:
            print('File %s is already in place. If you want to fetch it again, please first remove it from the folder %s ' % (fname, folder))

    print('Done.')
    print('Files copied in folder %s' % folder)
        



