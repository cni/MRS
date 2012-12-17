import numpy as np

import pfile
import pfheader

import nitime as nt
import nitime.analysis as nta


def get_header(file_name):
    """
    Read the header of a P file.

    Parameters
    ----------
    file_name : str
       Full path to the P file.


    Returns
    -------
    hdr : dict
       The fields of the dict contain the information in the file header.
    
    """
    f_obj = file(file_name, 'r')
    hdr_offset = dict(version = [0, np.float32],
                      run = [4, np.uint32],
                      npasses = [64, np.uint16],
                      frsize = [80, np.uint16],
                      nframes = [74, np.uint16],
                      nslices = [68, np.uint16],
                      nechoes = [70, np.uint16],
                      rawhdrsize = [1468, np.uint32],
                      hnover = [78, np.uint16],
                      ptsize = [82, np.uint16],
                      nex = [72, np.uint16],
                      startrcvr = [200, np.uint16],
                      endrcvr = [202, np.uint16],
                      rhimsize = [106, np.uint16],
                      rhrecon = [60, np.uint16],
                      rhtype = [56, np.uint16],
                      rhdayres = [104, np.uint16],
                      te = [1212, np.uint32],
                      te2 = [1216, np.uint32])

    hdr = {}
    for x in hdr_offset:
        f_obj.seek(hdr_offset[x][0])
        hdr[x] = np.fromfile(f_obj, hdr_offset[x][1], 1)[0]

    # For some versions of these files, you need to read in more stuff:
    if hdr['version'] >= 20:
        more_offset = dict(rawsize = [1660, np.uint64],
                           exam = [148712, np.uint16],
                           series = [148724, np.uint16],
                           image = [148726, np.uint16],
                           ileaves = [914, np.uint16])

    elif hdr['version'] >= 14.3:
        more_offset = dict(rawsize = [116, np.uint32],
                           exam = [144884, np.uint16],
                           series = [144896, np.uint16],
                           image = [144898, np.uint16])


    elif hdr['version'] >= 14.0:
         more_offset = dict(rawsize = [116, np.uint32],
                           exam = [143384, np.uint16],
                           series = [143396, np.uint16],
                           image = [143398, np.uint16])

    # This is apparently the case for the oldest files:
    else: 
         more_offset = dict(rawsize = [116, np.uint32],
                           exam = [65200, np.uint16],
                           series = [65212, np.uint16],
                           image = [65214, np.uint16])

    for x in more_offset:
        f_obj.seek(more_offset[x][0])
        hdr[x] = np.fromfile(f_obj, more_offset[x][1], 1)

    # Calculate the number of coils, which is not explicitly stored:
    hdr['ncoils'] = hdr['endrcvr'] - hdr['startrcvr'] + 1

    return hdr

def get_data(file_name, header=None):
    """
    Get the data from a Pfile

    Parameters
    ----------
    file_name : str
       The full path to a P file containing data
    """

    if header is None:
        hdr = get_header(file_name)

    f_obj = file(file_name, 'r')
    
    n_frames = hdr['nframes'] + hdr['hnover']
    n_echoes = hdr['nechoes']
    n_slices = hdr['nslices']/hdr['npasses']
    n_coils = hdr['ncoils']
    n_passes = hdr['npasses']

    # Preallocate:
    data = np.empty((hdr['frsize'],
                     n_frames,
                     n_echoes,
                     n_slices,
                     n_coils,
                     n_passes), dtype=np.complex)


    # Size (in bytes) of different entities in the data:
    ptsize = hdr['ptsize']  # The size of each sample
    data_type = [np.int16, np.int32][ptsize/2-1]
    
    # This is double the size as above, because the data is complex:
    framesize = 2*ptsize*hdr['frsize']

    echosize = framesize*(1+hdr['nframes']+hdr['hnover'])
    slicesize = echosize*hdr['nechoes']
    coilsize = slicesize*hdr['nslices']/hdr['npasses']
    passsize = coilsize*hdr['ncoils']

    
    # Jump past the header:
    bytes_to_skip = hdr['rawhdrsize']

    for passidx in range(n_passes):
        for coilidx in range(n_coils):
            for sliceidx in range(n_slices):
                for echoidx in range(n_echoes):
                    for frameidx in range(1, n_frames+1):
                        bytes_to_skip = (passidx * passsize +
                                          coilidx * coilsize +
                                          sliceidx * slicesize +
                                          echoidx * echosize +
                                        frameidx * framesize) + hdr['rawhdrsize']
                        f_obj.seek(bytes_to_skip)
                        dr = np.fromfile(f_obj, data_type, framesize/ptsize)
                        dr = np.reshape(dr, (-1, 2)).T
                        data[:,
                             frameidx-1,
                             echoidx,
                             sliceidx,
                             coilidx,
                             passidx] = dr[0] + dr[1] * 1j


    return data
