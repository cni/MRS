import numpy as np
import MRS.analysis as ana
import MRS.utils as ut
import MRS.files as io


class GABA(object):
    """
    Class for analysis of GABA MRS.
    
    """

    def __init__(self, in_file, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-0.7, max_ppm=4.3,
                 fit_lb=2.8, fit_ub=3.2):
        """
        Parameters
        ----------

        in_file : str
            Path to a P file containing MRS data.

        line_broadening : float
           How much to broaden the spectral line-widths (Hz)
           
        zerofill : int
           How many zeros to add to the spectrum for additional spectral
           resolution

        min_ppm, max_ppm : float
           The limits of the spectra that are represented

        fit_lb, fit_ub : float
           The limits for the part of the spectrum for which we fit the
           creatine and GABA peaks. 
        
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

        # Calculate sum and difference:
        self.diff_spectra = self.echo2 - self.echo1
        self.sum_spectra = self.echo2 + self.echo1
        self.fit_lb=fit_lb
        self.fit_ub=fit_ub  


    def fit_creatine(self, reject_outliers=3.0):
        """
        Fit a model to the portion of the summed spectra containing the
        creatine signal.

        Parameters
        ----------
        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (2.6, 3.6)

        reject_outliers: float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection
           

        """
        model, signal, params, fit_idx = ana.fit_lorentzian(self.sum_spectra,
                                                            self.f_ppm,
                                                            lb=self.fit_lb,
                                                            ub=self.fit_ub)

        # Reject outliers:
        if reject_outliers:
            # Z score across repetitions:
            z_score = (params - np.mean(params, 0))/np.std(params, 0)
            outlier_idx = np.unique(np.where(np.abs(z_score)>3.0)[0])
            # Use an array of ones to index everything but the outliers:
            ii = np.ones(signal.shape[0], dtype=bool)
            ii[outlier_idx] = 0
            # We'll keep around a private attribute to tell us which transients
            # were good:
            self._fit_transients = np.where(ii)
            model = model[self._fit_transients]
            signal = signal[self._fit_transients]
            params = params[self._fit_transients]
            
        self.creatine_model = model
        self.creatine_signal = signal
        self.creatine_params = params
        self.fit_idx = fit_idx
        
    def fit_gaba(self):
        """
        Fit a Gaussian function to the GABA peak at ~ 3 ppm.
        """
        # We need to fit the creatine, so that we know which transients to
        # exclude in fitting the GABA peak:
        if not hasattr(self, 'creatine_params'):
            self.fit_creatine()

        fit_spectra = self.diff_spectra[self._fit_transients]
        # fit_idx should already be set from fitting the creatine params:
        model, signal, params, _ = ana.fit_gaussian(fit_spectra,
                                                    self.f_ppm,
                                                    lb=self.fit_lb,
                                                    ub=self.fit_ub)

        self.gaba_model = model
        self.gaba_signal = signal
        self.gaba_params = params
