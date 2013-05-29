import numpy as np
import MRS.analysis as ana
import MRS.utils as ut
import MRS.files as io


class GABA(object):
    """
    Class for analysis of GABA MRS.
    
    """

    def __init__(self, in_file, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-0.7, max_ppm=4.3):
        """
        Parameters
        ----------

        in_file : str
            Path to a P file containing MRS data.
        
        """
        self.raw_data =  io.get_data(in_file)
        w_data, w_supp_data = ana.coil_combine(self.raw_data)
        # We keep these around for reference, as private attrs
        self._water_data = w_data
        self._w_supp_data = w_supp_data
        # This is the time-domain signal of interest, combined over coils:
        self.data = ana.subtract_water(w_data, w_supp_data)

        f_hz, spectra = ana.get_spectra(self.data,
                                           line_broadening=line_broadening,
                                           zerofill=zerofill,
                                           filt_method=filt_method)
                                           
        self.f_hz = f_hz
        # Convert from Hz to ppm and extract the part you are interested in.
        f_ppm = ut.freq_to_ppm(self.f_hz)
        idx0 = np.argmin(np.abs(f_ppm - min_ppm))
        idx1 = np.argmin(np.abs(f_ppm - max_ppm))
        self.idx = slice(idx1, idx0)
        self.f_ppm = f_ppm[self.idx]
    
        # The first echo (off-resonance) is in the first output 
        self.echo1 = spectra[:,0][:,self.idx]
        # The on-resonance is in the second:
        self.echo2 = spectra[:,1][:,self.idx]

        

    
