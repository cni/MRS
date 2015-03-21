"""

Visualization functions

"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def stacked_spectra(f, spectra, fig=None):

    """
    Plot stacked spectra

    Parameters
    ----------

    """
    if fig is None:
        fig, ax = plt.subplots(1)
    else:
        fig = fig
        ax = fig.get_axes()[0]

    offset = 0
    for s in spectra:
        ax.plot(f, s.squeeze()+offset, color='k')
        offset += np.std(s)/2.0
    ax.invert_xaxis()
    return fig

def plot_spectra(f, spectra, n_boots=1000, fig=None, pct=68):
    """
    Plot average spectra with boot-strapped confidence intervals

    Parameters
    ----------
    pct : how many percentiles to include in the CI (default to 95%)

    """
    if fig is None:
        fig, ax = plt.subplots(1)
    else:
        fig = fig
        ax = fig.get_axes()[0]

    boots = np.zeros((n_boots, spectra.shape[-1]))
    for b in xrange(n_boots):
        boots[b] = np.random.random_integers(0,
                                             spectra.shape[-1]-1,
                                             spectra.shape[-1])

    alpha = (100 - pct)/2.
    CI = np.zeros(spectra.shape[-1])
    for x in xrange(CI.shape[0]):
        CI[x] = (stats.scoreatpercentile(boots[:, x], 100 - alpha) -
                 stats.scoreatpercentile(boots[:, x], alpha))

    y = np.mean(spectra, 0)
    ax.plot(f, y, color='k')
    ax.fill_between(f, y-CI, y+CI, alpha=0.2, color='b')
    ax.invert_xaxis()
    return fig


def plot_gaba(G):
    """
    This is the basic three-panel plot: echo1/echo2/diff with model fits

    """
    G.fit_gaba()
    fig, ax = plt.subplots(3)
    ax[0].plot(G.f_ppm, np.mean(G.echo1,0))
    ax[0].plot(G.f_ppm[G.cr_idx], stats.nanmean(G.creatine_model, 0), 'r')
    Cr_text = 'Cr params \n freq0: %1.2f\n area: %1.4f \n hwhm: %1.2f \n phase: %1.2f \n offset: %1.2f \n drift %1.2f'%(tuple([w for w in np.mean(G.creatine_params, 0)]))
    ax[0].text(0.9, 0.6, Cr_text, horizontalalignment='center', verticalalignment='center', transform = ax[0].transAxes)
    ax[1].plot(G.f_ppm, np.mean(G.echo2,0))
    ax[2].plot(G.f_ppm, np.mean(G.diff_spectra, 0))
    ax[2].plot(G.f_ppm[G.gaba_idx], stats.nanmean(G.gaba_model, 0), 'r')
    gaba_text = 'GABA params \n freq0: %1.2f\n sigma: %1.2f \n amp: %1.4f \n offset: %1.2f \n drift %1.2f'%(tuple([w for w in np.mean(G.gaba_params, 0)]))
    ax[2].text(0.9, 0.5, gaba_text, horizontalalignment='center', verticalalignment='center', transform = ax[2].transAxes)
    ax[2].set_xlabel('ppm')
    fig.set_size_inches([10,8])
    for a in ax:
        a.invert_xaxis()

    return fig
