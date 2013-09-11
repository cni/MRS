import numpy as np
import scipy.stats as stats
import nibabel as nib

import MRS.analysis as ana
import MRS.utils as ut
import MRS.freesurfer as fs


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
            Path to a nifti file containing MRS data.

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
        # The nifti files follow the strange nifti convention, but we want to
        # use our own logic, which is transients on dim 0 and time on dim -1:
        self.raw_data = np.transpose(nib.load(in_file).get_data(),
                                     [1,2,3,4,5,0]).squeeze()

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

    def naa_correct(self):

        """
        This function resets the fits and corrects shifts in the spectra.
        It uses uses the NAA peak at 2.0ppm as a guide to replaces the existing
        f_ppm values! 
        """
        self.reset_fits()

        # calculate diff
        diff = np.mean(self.diff_spectra, 0)
        # find index of NAA peak in diff spectrum
        idx = np.argmin(diff)
        NAA_ppm = np.max(self.f_ppm)-(float(idx)/len(diff))*(np.max(self.f_ppm)-np.min(self.f_ppm))
        
        # determine how far spectrum is shifted
        NAA_shift = 2.0-NAA_ppm
        
        # correct
        self.f_ppm = self.f_ppm + NAA_shift
        
    def reset_fits(self):
        """
        This is used to restore the original state of the fits.
        """
        for attr in ['creatine_params', 'creatine_model', 'creatine_signal',
                     'cr_idx', 'creatine_auc', 'gaba_params', 'gaba_model',
                     'gaba_signal', 'gaba_idx', 'gaba_auc', 'glx_params',
                     'glx_model', 'glx_signal', 'glx_idx', 'glx_auc' ]:
            if hasattr(self, attr):
                self.__delattr__(attr)


    def fit_water(self, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-0.7, max_ppm=5.0):
        """

        """
        # Get the water spectrum as well:
        f_hz, w_spectra = ana.get_spectra(self._water_data,
                                          line_broadening=line_broadening,
                                          zerofill=zerofill,
                                          filt_method=filt_method)

        f_ppm = ut.freq_to_ppm(f_hz)

        # We use different limits for the water part:
        idx0 = np.argmin(np.abs(f_ppm - min_ppm))
        idx1 = np.argmin(np.abs(f_ppm - max_ppm))
        idx = slice(idx1, idx0)
        f_ppm = f_ppm[idx]
        self.water_spectra = np.mean(w_spectra, 1)[:, idx]
        model, signal, params, fit_idx = ana.fit_lorentzian(self.water_spectra,
                                                            f_ppm,
                                                            lb=min_ppm,
                                                            ub=max_ppm)

        # Store the params:
        self.water_model = model
        self.water_signal = signal
        self.water_params = params

        mean_params = stats.nanmean(params, 0)
        self.water_auc = ana.integrate(ut.lorentzian,
                                          f_ppm[idx],
                                          tuple(mean_params),
                                          offset = mean_params[-2],
                                          drift = mean_params[-1])


    def fit_creatine(self, reject_outliers=3.0, fit_lb=2.7, fit_ub=3.2):
        """
        Fit a model to the portion of the summed spectra containing the
        creatine signal.

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (2.7, 3.1)

        Note
        ----
        We use upper and lower bounds that are a variation on the bounds
        mentioned on the GANNET ISMRM2013 poster [1]_.

        [1] RAE Edden et al (2013). Gannet GABA analysis toolkit. ISMRM
        conference poster.

        """
        model, signal, params, fit_idx = ana.fit_lorentzian(self.echo1,
                                                            self.f_ppm,
                                                            lb=fit_lb,
                                                            ub=fit_ub)

        # Use an array of ones to index everything but the outliers and nans:
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            # Z score across repetitions:
            z_score = (params - np.mean(params, 0))/np.std(params, 0)
            outlier_idx = np.where(np.abs(z_score)>3.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            ii[outlier_idx] = 0
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        # We'll keep around a private attribute to tell us which transients
        # were good:
        self._cr_transients = np.where(ii)
        self.creatine_model = model
        self.creatine_signal = signal
        self.creatine_params = params
        self.cr_idx = fit_idx
        mean_params = stats.nanmean(params, 0)
        self.creatine_auc = ana.integrate(ut.lorentzian,
                                          self.f_ppm[self.idx],
                                          tuple(mean_params),
                                          offset = mean_params[-2],
                                          drift = mean_params[-1])


    def _gaussian_helper(self, reject_outliers, fit_lb, fit_ub, phase_correct):
        """
        This is a helper function for fitting different segments of the spectrum
        with Gaussian functions (GLX and GABA)

        Parameters
        ----------
        reject_outliers : float or bool
            Z score for outlier rejection. If set to `False`, not outlier
            rejection.

        fit_lb : float
            The lower bound of the part of the ppm scale for which the Gaussian
            is fit.

        fit_ub : float
            The upper bound of the part of the scale fit.

        phase_correct : bool
            Whether to perform first-order phase correction by the paramters
            of the creatine Lorentzian fit.

        Returns
        -------
        choose_transients : tuple
            Indices into the original data's transients dimension to select

        non-outlier transients. If reject_outliers is set to `False`, this is
            all the  transients

        model : ndarray
            The model predicition in each transient, based on the fit.

        signal : ndarray
            The original signal in this part of the difference spectrum.

        params : ndarray
            The Gaussian parameters in each transient as fit.

        this_idx : slice object
            A slice into the part of the spectrum that is fit
        """
        # We need to fit the creatine, so that we know which transients to
        # exclude in fitting this peak:
        if not hasattr(self, 'creatine_params'):
            self.fit_creatine()

        fit_spectra = np.ones(self.diff_spectra.shape) * np.nan
        fit_spectra[self._cr_transients] =\
                    self.diff_spectra[self._cr_transients].copy()

        if phase_correct: 
            for ii, this_spec in enumerate(fit_spectra):
                fit_spectra[ii] = ut.phase_correct_zero(this_spec,
                                        self.creatine_params[ii, 3])

        # fit_idx should already be set from fitting the creatine params:
        model, signal, params, this_idx = ana.fit_gaussian(fit_spectra,
                                                           self.f_ppm,
                                                           lb=fit_lb,
                                                           ub=fit_ub)

        # We'll use these indices to reject outliers (or not):
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            # Z score across repetitions:
            z_score = (params - np.mean(params, 0))/np.std(params, 0)
            outlier_idx = np.where(np.abs(z_score)>3.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            # Use an array of ones to index everything but the outliers and nans:
            ii[outlier_idx] = 0
            # Set the outlier transients to nan:
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        choose_transients = np.where(ii)
        return choose_transients, model, signal, params, this_idx


    def fit_gaba(self, reject_outliers=3.0, fit_lb=2.8, fit_ub=3.4,
                 phase_correct=True):
        """
        Fit a Gaussian function to the GABA peak at ~ 3 ppm.
        """
        choose_transients, model, signal, params, this_idx=self._gaussian_helper(
            reject_outliers, fit_lb, fit_ub, phase_correct)

        self._gaba_transients = choose_transients
        self.gaba_model = model
        self.gaba_signal = signal
        self.gaba_params = params
        self.gaba_idx = this_idx
        mean_params = stats.nanmean(params, 0)
        # Calculate AUC over the entire domain:
        self.gaba_auc = ana.integrate(ut.gaussian,
                                      self.f_ppm[self.idx],
                                      tuple(mean_params),
                                      offset = mean_params[-2],
                                      drift = mean_params[-1])


    def est_gaba_conc(self):
	"""
	Estimate gaba concentration based on equation adapted from Sanacora
	1999, p1045

	Ref: Sanacora, G., Mason, G. F., Rothman, D. L., Behar, K. L., Hyder,
	F., Petroff, O. A., ... & Krystal, J. H. (1999). Reduced cortical
	$\gamma$-aminobutyric acid levels in depressed patients determined by
	proton magnetic resonance spectroscopy. Archives of general psychiatry,
	56(11), 1043.

	"""
	# need gaba_auc and creatine_auc
	if not hasattr(self, 'gaba_params'):
	    self.fit_gaba()

	# estimate [GABA] according to equation9
	gaba_conc_est = self.gaba_auc / self.creatine_auc * 1.5 * 9.0
	
	self.gaba_conc_est = gaba_conc_est


    def voxel_seg(self, segfile, MRSfile):
        """
        add voxel segmentation info
        
        Parameters
        ----------
        
        segfile : str
            Path to nifti file with segmentation info (e.g. XXXX_aseg.nii.gz)
        
        MRSfile : str
            Path to MRS nifti file 
        """
        total, grey, white, csf, nongmwm, pGrey, pWhite, pCSF, pNongmwm =\
            fs.MRSvoxelStats(segfile, MRSfile)
        
        self.pGrey = pGrey
        self.pWhite = pWhite
        self.pCSF = pCSF
        self.pNongmwm = pNongmwm

    def fit_glx(self, reject_outliers=3.0, fit_lb=3.5, fit_ub=4.5,
                 phase_correct=True):
        """
        Fit a Gaussian function to the Glu/Gln (GLX) peak at ~ 4 ppm).
        """
        choose_transients, model, signal, params, this_idx=self._gaussian_helper(
            reject_outliers, fit_lb, fit_ub, phase_correct)

        self._glx_transients = choose_transients
        self.glx_model = model
        self.glx_signal = signal
        self.glx_params = params
        self.glx_idx = this_idx
        mean_params = stats.nanmean(params, 0)
        # Calculate AUC over the entire domain:
        self.glx_auc = ana.integrate(ut.gaussian,
                                      self.f_ppm[self.idx],
                                      tuple(mean_params),
                                      offset = mean_params[-2],
                                      drift = mean_params[-1])


class SingleVoxel(object):
    """
    Class for representation and analysis of single voxel (SV) -PROBE
    experiments.
    """

    def __init__(self, in_file, line_broadening=5, zerofill=100,
                 filt_method=None, min_ppm=-0.7, max_ppm=4.3):
        """
        Parameters
        ----------

        in_file : str
            Path to a nifti file with SV-PROBE MRS data.

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
        self.raw_data = np.transpose(nib.load(in_file).get_data(),
                                     [1,2,3,4,5,0]).squeeze()

        w_data, w_supp_data = ana.coil_combine(self.raw_data, w_idx = range(8),
                                               coil_dim=1)
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
        self.spectra = spectra[:,self.idx]
    
