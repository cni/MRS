import numpy as np
import scipy.stats as stats
import nibabel as nib
import warnings

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
        f_hz, w_supp_spectra = ana.get_spectra(w_supp_data,
                                           line_broadening=line_broadening,
                                           zerofill=zerofill,
                                           filt_method=filt_method)

        self.w_supp_spectra = w_supp_spectra

        # Often, there will be some small offset from the on-resonance
        # frequency, which we can correct for. We fit a Lorentzian to each of
        # the spectra from the water-suppressed data, so that we can get a
        # phase-corrected estimate of the frequency shift, instead of just
        # relying on the frequency of the maximum:
        self.w_supp_lorentz = np.zeros(w_supp_spectra.shape[:-1] + (6,))
        for ii in range(self.w_supp_lorentz.shape[0]):
            for jj in range(self.w_supp_lorentz.shape[1]):
                self.w_supp_lorentz[ii,jj]=\
                    ana._do_lorentzian_fit(f_hz, w_supp_spectra[ii,jj])

        # We store the frequency offset for each transient/echo:
        self.freq_offset = self.w_supp_lorentz[..., 0]

        # But for now, we average over all the transients/echos for the
        # correction: 
        mean_freq_offset = np.mean(self.w_supp_lorentz[..., 0])
        f_hz = f_hz - mean_freq_offset
    
        self.water_fid = w_data
        self.w_supp_fid = w_supp_data
        # This is the time-domain signal of interest, combined over coils:
        self.data = ana.subtract_water(w_data, w_supp_data)

        _, spectra = ana.get_spectra(self.data,
                                     line_broadening=line_broadening,
                                     zerofill=zerofill,
                                     filt_method=filt_method)

        self.f_hz = f_hz
        # Convert from Hz to ppm and extract the part you are interested in.
        f_ppm = ut.freq_to_ppm(self.f_hz)
        idx0 = np.argmin(np.abs(f_ppm - min_ppm))
        idx1 = np.argmin(np.abs(f_ppm - max_ppm))
        self.idx = slice(idx1, idx0)
        self.f_ppm = f_ppm
    
        self.echo_off = spectra[:, 1]
        self.echo_on = spectra[:, 0]

        # Calculate sum and difference:
        self.diff_spectra = self.echo_on - self.echo_off
        self.sum_spectra = self.echo_off + self.echo_on

        
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
                 filt_method=None, min_ppm=-5.0, max_ppm=5.0):
        """

        """
        # Get the water spectrum as well:
        f_hz, w_spectra = ana.get_spectra(self.water_fid,
                                          line_broadening=line_broadening,
                                          zerofill=zerofill,
                                          filt_method=filt_method)

        f_ppm = ut.freq_to_ppm(f_hz)
        # Averaging across echos:
        self.water_spectra = np.mean(w_spectra, 1)
        model, signal, params = ana.fit_lorentzian(self.water_spectra,
                                                   self.f_ppm,
                                                   lb=min_ppm,
                                                   ub=max_ppm)

        # Store the params:
        self.water_model = model
        self.water_signal = signal
        self.water_params = params
        self.water_idx = ut.make_idx(self.f_ppm, min_ppm, max_ppm)
        mean_params = stats.nanmean(params, 0)
        self.water_auc = self._calc_auc(ut.lorentzian, params, self.water_idx)


    def _calc_auc(self, model, params, idx):
        """
        Helper function to calculate the area under the curve of a model

        Parameters
        ----------
        model : callable
            Probably either ut.lorentzian or ut.gaussian, but any function will
            do, as long as its first parameter is an array of frequencies and
            the third parameter controls its amplitude.

        params : ndarray
            Each row of these should contain exactly the number of params that
            the model function expects after the first (frequency)
            parameter. The second column should control the amplitude of the
            function.

        idx :
           Indices to the part of the spectrum over which AUC will be
           calculated.

        """


        # Here's what we are going to do: For each transient, we generate
        # the spectrum for two distinct sets of parameters: one is exactly as
        # fit to the data, the other is the same expect with amplitude set to
        # 0. To calculate AUC, we take the difference between them:
        auc = np.zeros(params.shape[0])
        delta_f = np.abs(self.f_ppm[1]-self.f_ppm[0])
        p = np.copy(params)
        for t in range(auc.shape[0]):
            model1 = model(self.f_ppm[idx], *p[t])
            # This controls the amplitude in both the Gaussian and the
            # Lorentzian: 
            p[t, 1] = 0
            model0 = model(self.f_ppm[idx], *p[t])
            auc[t] = np.sum((model1 - model0) * delta_f)
        return auc

    def _outlier_rejection(self, params, model, signal, ii):
        """
        Helper function to reject outliers

        DRY!
        
        """
        # Z score across repetitions:
        z_score = (params - np.mean(params, 0))/np.std(params, 0)
        # Silence warnings: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outlier_idx = np.where(np.abs(z_score)>3.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            ii[outlier_idx] = 0
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        return model, signal, params, ii



    def fit_creatine(self, reject_outliers=3.0, fit_lb=2.7, fit_ub=3.5):
        """
        Fit a model to the portion of the summed spectra containing the
        creatine and choline signals.

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (2.7, 3.5)

        Note
        ----
        We use upper and lower bounds that are a variation on the bounds
        mentioned on the GANNET ISMRM2013 poster [1]_.

        [1] RAE Edden et al (2013). Gannet GABA analysis toolkit. ISMRM
        conference poster.

        """
        # We fit a two-lorentz function to this entire chunk of the spectrum,
        # to catch both choline and creatine
        model, signal, params = ana.fit_two_lorentzian(self.sum_spectra,
                                                       self.f_ppm,
                                                       lb=fit_lb,
                                                       ub=fit_ub)

        # Use an array of ones to index everything but the outliers and nans:
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)
            
        # We'll keep around a private attribute to tell us which transients
        # were good (this is for both creatine and choline):
        self._cr_transients = np.where(ii)
        
        # Now we separate choline and creatine params from each other (remember
        # that they both share offset and drift!):
        self.choline_params = params[:, (0,2,4,6,8,9)]
        self.creatine_params = params[:, (1,3,5,7,8,9)]
        
        self.cr_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        # We'll need to generate the model predictions from these parameters,
        # because what we're holding in 'model' is for both together:
        self.choline_model = np.zeros((self.creatine_params.shape[0],
                                    np.abs(self.cr_idx.stop-self.cr_idx.start)))

        self.creatine_model = np.zeros((self.choline_params.shape[0],
                                    np.abs(self.cr_idx.stop-self.cr_idx.start)))
        
        for idx in range(self.creatine_params.shape[0]):
            self.creatine_model[idx] = ut.lorentzian(self.f_ppm[self.cr_idx],
                                                     *self.creatine_params[idx])
            self.choline_model[idx] = ut.lorentzian(self.f_ppm[self.cr_idx],
                                                    *self.choline_params[idx])
        self.creatine_signal = signal
        self.creatine_auc = self._calc_auc(ut.lorentzian,
                                           self.creatine_params,
                                           self.cr_idx)
        self.choline_auc = self._calc_auc(ut.lorentzian,
                                          self.choline_params,
                                          self.cr_idx)

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
        # Silence warnings: 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_spectra[self._cr_transients] =\
                self.diff_spectra[self._cr_transients].copy()

        if phase_correct: 
            for ii, this_spec in enumerate(fit_spectra):
                # Silence warnings: 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_spectra[ii] = ut.phase_correct_zero(this_spec,
                                        self.creatine_params[ii, 3])

        # fit_idx should already be set from fitting the creatine params:
        model, signal, params = ana.fit_gaussian(fit_spectra,
                                                 self.f_ppm,
                                                 lb=fit_lb,
                                                 ub=fit_ub)

        # We'll use these indices to reject outliers (or not):
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)

        choose_transients = np.where(ii)
        this_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        return choose_transients, model, signal, params, this_idx


    def fit_gaba(self, reject_outliers=3.0, fit_lb=2.8, fit_ub=3.4,
                 phase_correct=True):
        """
        Fit a Gaussian function to the GABA peak at ~ 3 ppm.
        """
        choose_transients, model, signal, params,this_idx= self._gaussian_helper(
            reject_outliers, fit_lb, fit_ub, phase_correct)

        self._gaba_transients = choose_transients
        self.gaba_model = model
        self.gaba_signal = signal
        self.gaba_params = params
        self.gaba_idx = this_idx
        mean_params = stats.nanmean(params, 0)
        self.gaba_auc =  self._calc_auc(ut.gaussian, params, self.gaba_idx)


    def fit_glx(self, reject_outliers=3.0, fit_lb=3.5, fit_ub=4.5,
                 phase_correct=True):
        """
        Fit a Gaussian function to the Glu/Gln (GLX) peak at ~ 4 ppm).
        """
        choose_transients, model, signal, params,this_idx=self._gaussian_helper(
            reject_outliers, fit_lb, fit_ub, phase_correct)

        self._glx_transients = choose_transients
        self.glx_model = model
        self.glx_signal = signal
        self.glx_params = params
        self.glx_idx = this_idx
        mean_params = stats.nanmean(params, 0)
        self.glx_auc =  self._calc_auc(ut.gaussian, params, self.glx_idx)

    def fit_naa(self, reject_outliers=3.0, fit_lb=1.8, fit_ub=2.4,
                 phase_correct=True):
        """
        Fit a Gaussian function to the NAA peak at ~ 2 ppm.
        Example of fitting inverted peak:
        Foerster et al. 2013, An imbalance between excitatory and
        inhibitory neurotransmitters in amyothrophic lateral sclerosis
        revealed by use of 3T proton MRS
        """

        model, signal, params = ana.fit_lorentzian(self.diff_spectra,
                                                   self.f_ppm,
                                                   lb=fit_lb,
                                                   ub=fit_ub)

        # Store the params:
        self.naa_model = model
        self.naa_signal = signal
        self.naa_params = params
        self.naa_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        mean_params = stats.nanmean(params, 0)
        self.naa_auc = self._calc_auc(ut.lorentzian, params, self.naa_idx)


    def fit_glx2(self, reject_outliers=3.0, fit_lb=3.6, fit_ub=3.9, phase_correct=True, scalefit=False):
        """
        Fit a model to the portion of the diff spectra containing the
        glx signal. This treats the Glx signal as two gaussian peaks.
        Glx signal at 3.75ppm, +/- 0.15ppm (Hurd et al. 2004)

        Parameters
        ----------
        reject_outliers : float or bool
           If set to a float, this is the z score threshold for rejection (on
           any of the parameters). If set to False, no outlier rejection

        fit_lb, fit_ub : float
           What part of the spectrum (in ppm) contains the creatine peak.
           Default (3.5, 4.2)

        scalefit : boolean
           If this is set to true, attempt is made to prevent over or under-fitting
           with a second round of fitting where the fitted curve is fit with
           a scale factor. (default false)

        References
        ----------
        Hurd et al. 2004, Measurement of brain glutamate using TE-averaged PRESS at 3T
        """
        if not hasattr(self, 'creatine_params'):
            self.fit_creatine()

#        fit_spectra = np.ones(self.diff_spectra.shape) * np.nan
#
#        # Silence warnings:
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore")
#            fit_spectra[self._cr_transients] =\
#                self.diff_spectra[self._cr_transients].copy()
#
#        if phase_correct:
#            for ii, this_spec in enumerate(fit_spectra):
#                # Silence warnings:
#                with warnings.catch_warnings():
#                    warnings.simplefilter("ignore")
#                    fit_spectra[ii] = ut.phase_correct_zero(this_spec,
#                                        self.creatine_params[ii, 3])

        fit_spectra = self.diff_spectra

        # We fit a two-gaussian function to this entire chunk of the spectrum,
        # to catch both glx peaks
        model, signal, params = ana.fit_two_gaussian(fit_spectra, self.f_ppm,lb=fit_lb, ub=fit_ub)

        # Use an array of ones to index everything but the outliers and nans:
        ii = np.ones(signal.shape[0], dtype=bool)
        # Reject outliers:
        if reject_outliers:
            model, signal, params, ii = self._outlier_rejection(params,
                                                                model,
                                                                signal,
                                                                ii)

        # We'll keep around a private attribute to tell us which transients
        # were good:
        self._glx2_transients = np.where(ii)

        # Now we separate params of the two glx peaks from each other
        # (remember that they both share offset and drift!):
        self.glxp1_params = params[:, (0,2,4,6,7)]
        self.glxp2_params = params[:, (1,3,5,6,7)]

        self.glx2_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        # We'll need to generate the model predictions from these parameters,
        # because what we're holding in 'model' is for both together:
        self.glxp1_model = np.zeros((self.glxp1_params.shape[0],
                                    np.abs(self.glx2_idx.stop-self.glx2_idx.start)))

        self.glxp2_model = np.zeros((self.glxp2_params.shape[0],
                                    np.abs(self.glx2_idx.stop-self.glx2_idx.start)))

        for idx in range(self.glxp2_params.shape[0]):
            self.glxp2_model[idx] = ut.gaussian(self.f_ppm[self.glx2_idx],*self.glxp2_params[idx])
            self.glxp1_model[idx] = ut.gaussian(self.f_ppm[self.glx2_idx],
                                                    *self.glxp1_params[idx])

        if scalefit:
            combinedmodel = self.glxp2_model + self.glxp1_model
            scalefac, scalemodel = ana._do_scale_fit(self.f_ppm[self.glx2_idx], signal,combinedmodel)
            # Reject outliers:
            scalemodel, signal, params, ii = self._rm_outlier_by_amp(params,
                                                                scalemodel,
                                                                signal,
                                                                ii)
            self.glx2_model = scalemodel
        else:
            self.glx2_model = self.glxp1_model + self.glxp2_model


        self.glx2_signal = signal
        self.glx2_auc = (self._calc_auc(ut.gaussian, self.glxp2_params, self.glx2_idx) +
                        self._calc_auc(ut.gaussian, self.glxp1_params, self.glx2_idx))

    def _rm_outlier_by_amp(self, params, model, signal, ii):
        """
        Helper function to reject outliers based on mean amplitude
        """
#        # mean amplitudes per transient
#        meanamps = np.mean(model,1)
        # max amplitudes
        maxamps = np.nanmax(np.abs(model),0)
        # zscore
#        z_score = (meanamps - np.nanmean(meanamps,0))/np.nanstd(meanamps,0)
        z_score = (maxamps - np.nanmean(maxamps,0))/np.nanstd(maxamps,0)
        print z_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outlier_idx = np.where(np.abs(z_score)>2.0)[0]
            nan_idx = np.where(np.isnan(params))[0]
            outlier_idx = np.unique(np.hstack([nan_idx, outlier_idx]))
            ii[outlier_idx] = 0
            print sum(ii)
            model[outlier_idx] = np.nan
            signal[outlier_idx] = np.nan
            params[outlier_idx] = np.nan

        return model, signal, params, ii


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
        self.f_ppm = f_ppm
        self.spectra = spectra[:,self.idx]
