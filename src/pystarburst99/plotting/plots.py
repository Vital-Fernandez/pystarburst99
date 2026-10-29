import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc_context
from matplotlib.figure import Figure
from pystarburst99.plotting.ptools import PlotAccessor

_logger = logging.getLogger('pySB99')

# Sentinel for "in_fig not provided" (different from in_fig=None)
_NO_FIG = object()

# Figure style
COLORS = {'fg': 'black', 'stellar': 'red', 'total': 'black', 'hires': 'tab:red'}

_FIG_TYPES = {'default': {'axes.titlesize': 12, 'axes.labelsize': 12},
              'isochrone': {'axes.titlesize': 15, 'axes.labelsize': 15, 'xtick.labelsize': 15, 'ytick.labelsize': 15,
                            'legend.fontsize': 12},
              'sed': {'figure.figsize': (8, 6), 'figure.dpi': 300, 'axes.labelsize': 20, 'xtick.labelsize': 20,
                      'ytick.labelsize': 20, 'legend.fontsize': 18}}

_AX_CFG = {'isochrone': {'xlabel': r'$T_\mathrm{eff}$ [kK]', 'ylabel': r'log($L)\, [L_{\odot}$]', 'xlim': (80, 0),
                         'ylim': (2.5, 6.0)},
           'sed': {'xlabel': r'Wavelength [$\AA$]', 'ylabel': r'log (Luminosity [$erg s^{-1} \AA^{-1}$])',
                   'xlim': (200, 3000)},
           'hires': {'title': 'Hires ifa spectra', 'xlabel': 'Wave', 'ylabel': 'Flux', 'xlim': (900, 2000)},
           'hires_norm': {'title': 'Hires ifa spectra', 'xlabel': 'Wave', 'ylabel': 'Normalised Flux',
                          'xlim': (900, 2000)},
           'HI': {'title': 'Ionising Flux HI over Time', 'xlabel': 'Time', 'ylabel': 'Ionising Flux (Lbol)',
                  'xlim': (6., 8.), 'ylim': (44., 54.)},
           'HeI': {'title': 'Ionising Flux HeI over Time', 'xlabel': 'Time', 'ylabel': 'Ionising Flux (Lbol)',
                   'xlim': (6., 8.), 'ylim': (44., 54.)},
           'HeII': {'title': 'Ionising Flux HeII over Time', 'xlabel': 'Time', 'ylabel': 'Ionising Flux (Lbol)',
                    'xlim': (6., 8.), 'ylim': (38., 54.)},
           'Lbol': {'title': 'Luminosity over Time', 'xlabel': 'Time', 'ylabel': 'Luminosity log(L_bol) [erg/s] ',
                    'xlim': (6., 7.3)},
           'wind_power': {'title': 'Wind Power over Time', 'xlabel': 'Time', 'ylabel': 'Wind Power [erg/sec]',
                          'xlim': (6., 7.25), 'ylim': (34., 42.)},
           'wind_momentum': {'title': 'Wind Momentum over Time', 'xlabel': 'Time', 'ylabel': 'Wind Mom [erg/sec]',
                             'xlim': (6., 7.25), 'ylim': (27.5, 33.)},
           'uv_slope': {'title': 'Beta slope over Time', 'xlabel': 'Time', 'ylabel': 'Beta UV slope', 'xlim': (6., 9.)},
           'ew': {'title': 'Equivalent widths over Time', 'xlabel': 'Time', 'ylabel': 'Equivalent Width - EW',
                  'xlim': (6., 8.)},
           'MV': {'title': 'Evolution of $M_{V}$', 'xlabel': 'Time', 'ylabel': '$M_{V}$', 'xlim': (6., 8.)},
           'SN_rate': {'title': 'SN rate over Time', 'xlabel': 'Time', 'ylabel': 'SN rate ', 'xlim': (6., 8.),
                       'ylim': (-4., -1.8)},
           'SN_rate_scatter': {'title': 'SN rate over Time', 'xlabel': 'Time', 'ylabel': 'SN rate ', 'xlim': (6., 8.),
                               'ylim': (-5., -1.8)}}


def fig_defaults(fig_cfg=None, fig_type='default'):

    """Matplotlib RcParams for a figure type, updated with the user configuration."""

    return {**_FIG_TYPES[fig_type], **(fig_cfg or {})}


def label_maker(plot_type, title=None):

    """Axes configuration (labels, title and limits) for a plot type. A user title takes precedence."""

    ax_cfg = dict(_AX_CFG[plot_type])
    if title is not None:
        ax_cfg['title'] = title

    return ax_cfg


def _fig_ax(in_fig=_NO_FIG, in_ax=None):

    """Infer the figure structure: use the input axes, or add an axes to the input (or a new) figure."""

    if in_ax is not None:
        return in_ax.figure, in_ax

    fig = plt.figure() if (in_fig is None) or (in_fig is _NO_FIG) else in_fig

    return fig, fig.add_subplot()


def _set_ax_cfg(ax, ax_cfg=None, in_ax=None):

    """Apply the axes configuration without overwriting the title or labels of user axes."""

    if ax_cfg is not None:
        if in_ax is not None:
            current = {'title': ax.get_title(), 'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel()}
            ax_cfg = {key: value for key, value in ax_cfg.items() if not current.get(key)}
        ax.set(**ax_cfg)

    return


def matplotlib_isochrone_display(teff, log_lum, H_abundance, in_fig=_NO_FIG, in_ax=None, ax_cfg=None):

    # Infer figure structure
    fig, ax = _fig_ax(in_fig, in_ax)

    # Plot the isochrone coloured by the surface hydrogen abundance
    sc = ax.scatter(np.asarray(teff)/1000, log_lum, s=1, c=H_abundance, cmap='plasma_r', vmin=0.0, vmax=1.0)
    fig.colorbar(sc, ax=ax, label='H abundance')

    # Figure wording (Not overwritting if provided)
    _set_ax_cfg(ax, ax_cfg, in_ax)

    return fig, ax


def matplotlib_sed_display(wave, log_flux_stellar, log_flux_total, age, in_fig=_NO_FIG, in_ax=None, ax_cfg=None):

    # Infer figure structure
    fig, ax = _fig_ax(in_fig, in_ax)

    # Plot the stellar and stellar + nebular SEDs
    ax.plot(wave, 10**log_flux_stellar, label=f'{age}Myr - stellar', color=COLORS['stellar'])
    ax.plot(wave, 10**log_flux_total, label=f'{age}Myr - stellar+neb', color=COLORS['total'])

    # Legend
    ax.legend(loc='best')

    # Figure wording (Not overwritting if provided)
    _set_ax_cfg(ax, ax_cfg, in_ax)

    return fig, ax


def matplotlib_hires_display(wave, flux, label=None, in_fig=_NO_FIG, in_ax=None, ax_cfg=None):

    # Infer figure structure
    fig, ax = _fig_ax(in_fig, in_ax)

    # Plot the spectrum
    ax.plot(wave, flux, label=label, color=COLORS['hires'])

    # Legend
    if label is not None:
        ax.legend()

    # Figure wording (Not overwritting if provided)
    _set_ax_cfg(ax, ax_cfg, in_ax)

    return fig, ax


def matplotlib_evolution_display(log_time, y, label=None, scatter=False, invert_y=False, in_fig=_NO_FIG, in_ax=None,
                                 ax_cfg=None):

    # Infer figure structure
    fig, ax = _fig_ax(in_fig, in_ax)

    # Plot the time evolution
    if scatter:
        ax.scatter(log_time, y, label=label)
    else:
        ax.plot(log_time, y, label=label)

    # Legend
    if label is not None:
        ax.legend()

    # Figure wording (Not overwritting if provided)
    _set_ax_cfg(ax, ax_cfg, in_ax)

    # Magnitudes and beta slopes increase downwards
    if invert_y:
        ax.invert_yaxis()

    return fig, ax




class PySB_Plotter(PlotAccessor):


    def _time_index(self, age):
        return int(np.argmin(np.abs(np.asarray(self._obj.times) - age * 1e6)))

    def _log_time(self):
        with np.errstate(divide='ignore'):
            return np.log10(self._obj.times)

    def isochrone(self, age=4.0, fname=None, default_name='plot_isochrones', title=None, in_fig=_NO_FIG, in_ax=None,
                  fig_cfg=None):

        """
        Plot the population Hertzsprung-Russell diagram at a given age.

        The stars are coloured by their surface hydrogen abundance. The effective temperatures are not corrected
        for the Wolf-Rayet winds.

        Parameters
        ----------
        age : float, optional
            Population age in Myr. The closest time step is plotted. Default is ``4.0``.
        fname : str or pathlib.Path, optional
            Output file path. If it is a folder, the figure is saved there as ``<default_name><fname_suffix>.png``.
            If not provided, the plot is displayed in a new window.
        default_name : str, optional
            File name (without extension) used when ``fname`` is a folder. Default is ``"plot_isochrones"``.
        title : str, optional
            Figure title. Default describes the metallicity, rotation and age.
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into. If not provided, a new figure is created and shown (or saved and
            closed if ``fname`` is given). If ``None``, a new figure is created but not shown, so you can add your own
            artists and call ``plot.show()``. With an existing or ``None`` figure, ``fname`` saves it without
            closing it.
        in_ax : matplotlib.axes.Axes, optional
            Existing Matplotlib axes to plot into. It takes precedence over ``in_fig``, and the figure is
            neither shown nor saved.
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary (e.g., size, DPI).
            See `matplotlib.RcParams
            <https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.RcParams>`_.

        Examples
        --------
        Plot the isochrone at 4 Myr:

        >>> results.plot.isochrone()

        Save the isochrone at 10 Myr in the model folder:

        >>> results.plot.isochrone(age=10, fname='pysb_testmod')
        """

        # Clear previous figure
        self.reset_figure()

        # Unpack the data
        idx = self._time_index(age)
        rot_label = 'with' if self._obj.rot else 'without'
        title = title or f'Isochrone at {self._obj.times[idx]/1e6:.1f} Myr for Z={self._obj.Z} {rot_label} rotation'

        # Get the figure style
        plt_cfg = fig_defaults(fig_cfg, fig_type='isochrone')
        ax_cfg = label_maker('isochrone', title=title)

        # Generate figure
        with rc_context(plt_cfg):
            self.fig, self.ax = matplotlib_isochrone_display(self._obj.iso_teff[idx], self._obj.iso_lum[idx],
                                                             self._obj.iso_H[idx], in_fig=in_fig, in_ax=in_ax,
                                                             ax_cfg=ax_cfg)

            # Display | save | pass figure
            self.save_close_fig_swicth(self._resolve_fname(fname, default_name), 'tight', self.fig, in_fig=in_fig,
                                       in_ax=in_ax)

        return

    def sed(self, age=None, fname=None, default_name='SED_Time_Best', title=None, in_fig=_NO_FIG, in_ax=None,
            fig_cfg=None):

        """
        Plot the stellar and stellar + nebular spectral energy distribution at a given age.

        Parameters
        ----------
        age : float, optional
            Population age in Myr. The closest time step is plotted. Default is the model ``output_age``.
        fname : str or pathlib.Path, optional
            Output file path. If it is a folder, the figure is saved there as ``<default_name><fname_suffix>.png``.
            If not provided, the plot is displayed in a new window.
        default_name : str, optional
            File name (without extension) used when ``fname`` is a folder. Default is ``"SED_Time_Best"``.
        title : str, optional
            Figure title. Default is none.
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into (see :meth:`isochrone`).
        in_ax : matplotlib.axes.Axes, optional
            Existing Matplotlib axes to plot into (see :meth:`isochrone`).
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary (e.g., size, DPI).

        Examples
        --------
        >>> results.plot.sed(age=3.0)
        """

        # Clear previous figure
        self.reset_figure()

        # Unpack the data
        age = self._obj.output_age if age is None else age
        idx = self._time_index(age)

        # Get the figure style
        plt_cfg = fig_defaults(fig_cfg, fig_type='sed')
        ax_cfg = label_maker('sed', title=title)

        # Generate figure
        with rc_context(plt_cfg):
            self.fig, self.ax = matplotlib_sed_display(self._obj.sed_wave, self._obj.SED_stellar[idx],
                                                       self._obj.SED_total[idx], age, in_fig=in_fig, in_ax=in_ax,
                                                       ax_cfg=ax_cfg)

            # Display | save | pass figure
            self.save_close_fig_swicth(self._resolve_fname(fname, default_name), 'tight', self.fig, in_fig=in_fig,
                                       in_ax=in_ax)

        return

    def hires_spectrum(self, age=None, norm=False, fname=None, default_name=None, title=None, in_fig=_NO_FIG,
                       in_ax=None, fig_cfg=None):

        """
        Plot the high-resolution UV spectrum (stellar + nebular continuum) at a given age.

        Parameters
        ----------
        age : float, optional
            Population age in Myr. The closest time step is plotted. Default is the model ``output_age``.
        norm : bool, optional
            If ``True``, plot the continuum-normalised spectrum. Default is ``False``.
        fname : str or pathlib.Path, optional
            Output file path. If it is a folder, the figure is saved there as ``<default_name><fname_suffix>.png``.
            If not provided, the plot is displayed in a new window.
        default_name : str, optional
            File name (without extension) used when ``fname`` is a folder. Default is ``"Hires_ifa"``, or
            ``"normHires_ifa"`` if ``norm=True``.
        title : str, optional
            Figure title. Default is ``"Hires ifa spectra"``.
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into (see :meth:`isochrone`).
        in_ax : matplotlib.axes.Axes, optional
            Existing Matplotlib axes to plot into (see :meth:`isochrone`).
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary (e.g., size, DPI).

        Examples
        --------
        >>> results.plot.hires_spectrum(norm=True)
        """

        # Clear previous figure
        self.reset_figure()

        # Unpack the data
        age = self._obj.output_age if age is None else age
        idx = self._time_index(age)
        flux = self._obj.hires_flux_norm[idx] if norm else 10**self._obj.hires_flux_wneb[idx]
        default_name = default_name or ('normHires_ifa' if norm else 'Hires_ifa')

        # Get the figure style
        plt_cfg = fig_defaults(fig_cfg)
        ax_cfg = label_maker('hires_norm' if norm else 'hires', title=title)

        # Generate figure
        with rc_context(plt_cfg):
            self.fig, self.ax = matplotlib_hires_display(self._obj.hires_wave, flux, label=f'pySB t={age}Myr',
                                                         in_fig=in_fig, in_ax=in_ax, ax_cfg=ax_cfg)

            # Display | save | pass figure
            self.save_close_fig_swicth(self._resolve_fname(fname, default_name), 'tight', self.fig, in_fig=in_fig,
                                       in_ax=in_ax)

        return

    def evolution(self, quantity, fname=None, default_name=None, label=None, title=None, in_fig=_NO_FIG, in_ax=None,
                  fig_cfg=None):

        """
        Plot the time evolution of a population quantity.

        Parameters
        ----------
        quantity : {"HI", "HeI", "HeII", "Lbol", "wind_power", "wind_momentum", "uv_slope", "ew", "MV", "SN_rate", "SN_rate_scatter"}
            Quantity to plot: the ionising photon rates (``HI``, ``HeI``, ``HeII``), the bolometric luminosity,
            the wind power and momentum, the UV β slope, the Hα equivalent width, the absolute V magnitude or the
            supernova rate (as a line or as a scatter plot).
        fname : str or pathlib.Path, optional
            Output file path. If it is a folder, the figure is saved there as ``<default_name><fname_suffix>.png``.
            If not provided, the plot is displayed in a new window.
        default_name : str, optional
            File name (without extension) used when ``fname`` is a folder. Default depends on the quantity
            (e.g. ``"LbolHI"`` or ``"WindPower"``).
        label : str, optional
            Label for the plot legend. Default is none.
        title : str, optional
            Figure title. Default describes the quantity.
        in_fig : matplotlib.figure.Figure, optional
            Existing Matplotlib figure to plot into (see :meth:`isochrone`).
        in_ax : matplotlib.axes.Axes, optional
            Existing Matplotlib axes to plot into (see :meth:`isochrone`). For example, to compare two models:
            ``model_b.plot.evolution('HI', in_ax=model_a.plot.ax)``.
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary (e.g., size, DPI).

        Notes
        -----
        - The time axis is ``log10(t/yr)``.
        - For continuous star formation models the quantities are the cumulative values.

        Examples
        --------
        Plot the HI ionising photon rate:

        >>> results.plot.evolution('HI')

        Save the wind power in the model folder:

        >>> results.plot.evolution('wind_power', fname='pysb_testmod')
        """

        # Quantity: (results attribute, default file name, invert y-axis, scatter plot)
        quantities = {'HI': ('HI_flux', 'LbolHI', False, False),
                      'HeI': ('HeI_flux', 'LbolHeI', False, False),
                      'HeII': ('HeII_flux', 'LbolHeII', False, False),
                      'Lbol': ('Lbol', 'LbolTime', False, False),
                      'wind_power': ('wind_power', 'WindPower', False, False),
                      'wind_momentum': ('wind_momentum', 'WindMomentum', False, False),
                      'uv_slope': ('uv_slope', 'BetaSlope', True, False),
                      'ew': ('Ha_ew', 'EqWTime', False, False),
                      'MV': ('MV', 'EvolutoinMV', True, False),
                      'SN_rate': ('SN_rate', 'SNrateOverTime', False, False),
                      'SN_rate_scatter': ('SN_rate', 'SNrateOverTime2nd', False, True)}

        if quantity not in quantities:
            _logger.info(f'Quantity "{quantity}" not recognised. Options are: {list(quantities)}')
            return

        # Clear previous figure
        self.reset_figure()

        # Unpack the data
        attr, name, invert_y, scatter = quantities[quantity]
        default_name = default_name or name

        # Get the figure style
        plt_cfg = fig_defaults(fig_cfg)
        ax_cfg = label_maker(quantity, title=title)

        # Generate figure
        with rc_context(plt_cfg):
            self.fig, self.ax = matplotlib_evolution_display(self._log_time(), getattr(self._obj, attr), label=label,
                                                             scatter=scatter, invert_y=invert_y, in_fig=in_fig,
                                                             in_ax=in_ax, ax_cfg=ax_cfg)

            # Display | save | pass figure
            self.save_close_fig_swicth(self._resolve_fname(fname, default_name), 'tight', self.fig, in_fig=in_fig,
                                       in_ax=in_ax)

        return

    def all_results(self, folder, fig_cfg=None, evolution_plot_list=['HI', 'HeI', 'HeII', 'Lbol', 'wind_power',
                                                                     'wind_momentum', 'uv_slope', 'ew', 'MV', 'SN_rate',
                                                                     'SN_rate_scatter']):

        """
        Save all the model plots in a folder with their default names.

        Parameters
        ----------
        folder : str or pathlib.Path
            Output folder. It is created if it does not exist.
        fig_cfg : dict, optional
            Matplotlib figure configuration dictionary applied to all the plots.

        Examples
        --------
        >>> results.plot.all_results('pysb_testmod')
        """

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        self.isochrone(fname=folder, fig_cfg=fig_cfg)

        self.sed(fname=folder, fig_cfg=fig_cfg)

        self.hires_spectrum(fname=folder, fig_cfg=fig_cfg)

        self.hires_spectrum(norm=True, fname=folder, fig_cfg=fig_cfg)


        for plot in evolution_plot_list:
            self.evolution(plot, fname=folder, fig_cfg=fig_cfg)

        return