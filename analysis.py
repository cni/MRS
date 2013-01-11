import os

import numpy as np
import matplotlib.pyplot as plt

import files as io
import nitime as nt
import nitime.analysis as nta

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

