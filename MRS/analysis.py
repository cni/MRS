import os

import numpy as np
import matplotlib.pyplot as plt

import files as io
import nitime as nt
import nitime.analysis as nta

def coil_combine(data, w_idx=[1,2,3], sampling_rate=5000.0, NFFT=640):
    """
    Combine data across coils based on the amplitude of the water peak,
    according to:

    .. math::
        
        signal = sum_{i}(w(i)sig(i))

        w(i) = s(i)/(sqrt(sum(s(i)))
    
    where s(i) is the amplitude of the water peak in each coil 

    Parameters
    ----------
    data : float array
       The data as it comes from the scanner (read using the functions in
       files.py), with dimensions time x transients x off-/on-resonance x coils
    
    w_idx : tuple
       The indices to the non-water-suppressed transients. Per default we take
        the 2nd-4th transients. We dump the first one, because it seems to be
        quite different than the rest of them...
    
    sampling rate : float
        The sampling rate in Hz
    """
    # The transients are the second dimension in the data
    idxes_w = np.zeros(data.shape[1], dtype=bool)
    idxes_w[w_idx] = True
    # Data with water unsuppressed (first four transients - we throw away the
    # first one which is probably crap):
    w_data = data[:,np.where(idxes_w),:,:]
    # Data with water suppressed (the rest of the transients):
    idxes_nonw = np.zeros(data.shape[1], dtype=bool)
    idxes_nonw[np.where(~idxes_w)] = True
    idxes_nonw[0] = False
    w_supp_data = data[:,np.where(idxes_nonw),:,:]
    
    # Initialize time-series with this stuff
    ts_w = nt.TimeSeries(w_data.T, sampling_rate=sampling_rate)
    ts_w_supp = nt.TimeSeries(w_supp_data.T, sampling_rate=sampling_rate)
        
    # Push them through into spectral analysis. We'll use the Welch method
    S_w = nta.SpectralAnalyzer(ts_w, method=dict(NFFT=NFFT))    
    f_w, psd_w = S_w.spectrum_multi_taper

    # Average across repeats:
    max_psd = np.max(psd_w.squeeze(), -1)
    max_psd = np.mean(max_psd, 2)
    # Average across off-/on-resonance (they're very similar anyway):
    max_psd = np.mean(max_psd, -1) 
    w = np.power(max_psd/np.sqrt(np.sum(max_psd**2)),2)
    w = max_psd/np.sqrt(np.sum(max_psd))
    # Shorthand:
    na = np.newaxis
    # reshape to the data dimensions, so that you can broadcast
    w = w[:,na,na,na] #* 8 
    # Making sure the coil dimension is now first:
    weighted_w_data = np.sum(w * w_data.squeeze().T, 0)
    weighted_w_supp_data = np.sum(w * w_supp_data.squeeze().T, 0)
    return weighted_w_data, weighted_w_supp_data

if "__name__" == "__main__":

    data_files = ['mrs_P71680.7']#, 'mrs_P76288.7', 'pure_gaba_P64024.7']
    data_dir = '/Users/arokem/projects/MRS/data/' 
    sampling_rate = 128

    fig, ax = plt.subplots(1)


    for file_name in data_files:
        data = io.get_data(os.path.join(data_dir, file_name))
        data = data.squeeze()
        # Average over repeats.
        data = np.mean(data, 1)
        ft = np.fft.fft(data)

        #data = np.mean(data[0])
        #ts = nt.TimeSeries(data=data.T, sampling_rate=sampling_rate)

        #S = nta.SpectralAnalyzer(ts)
        #f,c = S.spectrum_fourier
        # ax.plot(f[:500], np.abs((c[0] - c[1]))[:500], label=file_name)
        #ax.plot(f, np.abs((c[0] - c[1])))
        #ax.plot(f, np.abs(np.fft.fftshift(c[0])))
        #ax.plot(np.abs(ft))
        #ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (TE1 - TE2)')

    fig.set_size_inches([10,10])
    plt.legend()
    plt.show()

