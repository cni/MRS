.. _data:


Organizing your data
---------------------

For its analysis, `SMAL` uses the raw MRS data, acquired with a phased array
coil. To achieve proper coil combination, we assume that the raw data from each
coil is provided as as a separate channel in the data. There are defaults set
for the number of non-water-suppressed transients, and sampling rate, but these
can be set as optional inputs to the user interface. 

The main :mod:`MRS.api` user interface assumes that the data is organized as an
array with shape: `(transients, echos, channels, timepoints)`. For example,
typical data might have the shape: `(68, 2, 32, 4096)`, where 68 transients are
acquired, each with two echos (GABA-edited and not GABA-edited), with a
32-channel phased array coil, acquiring 4096 samples in each FID curve at 5 kHz.

Alternatively, we can accept a full path to the data saved as a nifti
file. Following the nifti convention, the data array in the file needs to be
saved with the shape:  `(transients, 1, 1, echos, channels, timepoints)`. That
is, for this data, the data array in the nifti file will have the shape:
`(4096, 1, 1, 68, 2, 32)`. Data acquired at the `Stanford CNI`__, will be saved
as a nifti file in this convention (see also the example data provided at the
`Stanford Data Repository`__).

__ cni_
__ sdr-purl_

.. include:: links_names.inc

