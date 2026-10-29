from pathlib import Path
from types import SimpleNamespace
import time
from functools import partial

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np


from pystarburst99 import tools, tools_experimental


from pystarburst99.ins_outs import (ProgressTracker, load_cfg, pySB99_cfg, MODEL_FOLDER, check_options, check_output_age, TOML_PATH,
                                    check_output_folder, check_time_settings, warn_message, merge_settings, prepare_output_folder)


from pystarburst99.tools import (read_spectra_grid, reform_spec_grid, reform_spec_grid_WR, reform_spec_grid_powr,
                                 integrate_spec_grid, calc_Nostars, get_specsyn_params, ionise, continuum,
                                 calc_wind, get_uv_slope, get_ew, colours, get_SN_rate, compute_radii, interpolate_tracks,
                                 interp_time, prepare_outputs)

from pystarburst99.tools_experimental import BETA_TOOLS, interpolate_tracks_exp, HiresLibrary




class Compiler(ProgressTracker):

    def __init__(self, Z, output_folder=None, track_choice=None, rotation=None, SED_library=None, spectra_library=None,
                 mass_loss_rate_option=None, star_formation_option=None, output_age=None, plot_results_check=None,
                 input_imf_option=None, stochastic_imf_file=None, M_total=None, SFR=None, IMF_exponents=None,
                 IMF_mass_limits=None, maximum_SN_mass=None, time_step_length=None, run_speed_mode=None,
                 log_intervals=None, log_age_start=None, log_age_end=None, log_age_step=None, n_spectra=None,
                 age_decimals=None, beta_tools=None, overwrite_results=None, toml_path=None, verbose=True,
                 progress_callback=None):

        """
        pyStarburst99 grid compiler.

        The compiler takes a unique metallicity ``Z`` which must always be provided. Any other input left as ``None`` takes
        its value from the ``[default_settings]`` section of the configuration file (``pystarburst99.toml``), whose keys
        must be inputs of this class. All the inputs are stored in the ``.inputs`` attribute.

        The initialization only reads and validates the inputs. The data loading and the computation are done by
        ``load_metallicity_data``, ``track_grids``, ``spectral_grids`` and ``population_synthesis``

        Parameters
        ----------
        Z : str
            Metallicity key of the model: MWC, MW, LMC, SMC, IZw18, XMP and Z0
        output_folder : str or pathlib.Path, optional
            Folder where the results are saved. Its parent directory must exist.
        track_choice : {"GENEC", "Pauli"}, optional
            Evolutionary tracks.
        rotation : bool, optional
            Use the rotating (``True``) or non-rotating (``False``) GENEC tracks. It has no effect on the ``"Pauli"``
            tracks.
        SED_library : str, optional
            Low-resolution OB spectra library (e.g. ``"FW"``). The options depend on the metallicity. With ``"FW"``
            and an upper IMF mass above 120 Msol, the FastWind very massive star grid is used, if available.
        spectra_library : {"WM", "PoWR"}, optional
            High-resolution OB spectra library.
        mass_loss_rate_option : str, optional
            Mass loss rate of the PoWR high-resolution spectra. The options depend on the metallicity. It is only
            used if ``spectra_library="PoWR"``.
        star_formation_option : {"ISF", "CSF"}, optional
            Instantaneous burst (``"ISF"``) or continuous star formation (``"CSF"``).
        output_age : float, optional
            Age in Myr of the example SED and high-resolution spectrum saved as text files and plotted. It must be
            within the output spectra ages (``times_spectra``).
        plot_results_check : bool, optional
            Generate the plots and save them in the output folder.
        input_imf_option : bool, optional
            Use a stochastic IMF sample (``stochastic_imf_file``) to set the number of stars per mass bin instead of the
            analytical IMF.
        stochastic_imf_file : str or pathlib.Path, optional
            Numpy file (``.npy``) with the initial stellar masses of the stochastic IMF sample, in Msol. It is only used
            if ``input_imf_option=True``, and it must exist.
        M_total : float, optional
            Total stellar mass formed in the burst, in Msol. For ``"CSF"`` it is overwritten with
            ``time_step_length * SFR`` (mass formed per time step).
        SFR : float, optional
            Star formation rate in Msol/yr. It is only used for ``"CSF"``.
        IMF_exponents : sequence of float, optional
            Power-law exponents of the IMF segments.
        IMF_mass_limits : sequence of float, optional
            Mass limits in Msol of the IMF segments (one more element than ``IMF_exponents``). The upper mass cannot
            exceed the maximum mass of the selected tracks.
        maximum_SN_mass : float, optional
            Maximum initial mass in Msol of the stars counted in the supernova rate.
        time_step_length : float, optional
            Length in yr of the internal time steps used to compute the population.
        run_speed_mode : {"FAST", "DEFAULT", "HIGH_RES"}, optional
            Mass resolution of the interpolated isochrone grids.
        log_intervals : bool, optional
            Logarithmic (``True``) or linear (``False``) spacing of the output spectra ages. Default is ``False``.
        log_age_start : float, optional
            First output age as log10(age/yr), for both spacings. For linear intervals, ``-np.inf`` (or ``-inf`` in the
            toml file) starts the ages at 0 yr.
        log_age_end : float, optional
            Last output age as log10(age/yr), for both spacings. It must be finite and larger than ``log_age_start``.
        log_age_step : float, optional
            Step in dex for logarithmic intervals, or log10(step/yr) for linear intervals (``5.0`` -> 0.1 Myr).
            As in ``np.arange``, the last age is the largest one not above ``log_age_end``. Alternative to
            ``n_spectra``.
        n_spectra : int, optional
            Number of output ages (>= 2), both limits included (as in ``np.linspace``). Alternative to
            ``log_age_step``.
        age_decimals : int, optional
            Rounding of the ages in the spacing scale (log10(age/yr) for logarithmic intervals, yr for linear
            intervals). If not provided, grids from ``log_age_step`` are rounded to the decimals of ``log_age_start``
            and ``log_age_step``, grids from ``n_spectra`` are not rounded, and linear grids are rounded to whole years.
        beta_tools : bool, optional
            Experimental version of tools using numpy functions based on numpy arrays.
        overwrite_results : bool, optional
            If ``False``, an error is raised when the output folder already contains files. If ``True``, the files
            are deleted when the results are saved, after a warning.
        toml_path : str or pathlib.Path, optional
            Configuration file with the ``[default_settings]`` section. Default is ``pystarburst99.toml`` in the
            package folder.
        verbose : bool, optional
            Print the messages and progress bars in the terminal. Default is ``True``.
        progress_callback : callable, optional
            Function called as ``callback(compiler)`` after each message and loop iteration.

        Attributes
        ----------
        inputs : types.SimpleNamespace
            All the inputs after filling in the ``[default_settings]`` values, plus the output time grids
            ``times_spectra`` (output ages, yr) and ``times_steps`` (internal time grid, yr).
        stage, counter, total, fraction, message
            Progress of the current loop (e.g. the time steps of ``population_synthesis``), inherited from
            ``ProgressTracker``. They can be read at any time.

        Raises
        ------
        KeyError
            If ``[default_settings]`` contains keys which are not inputs of the class.
        ValueError
            If an option is not available, the time settings are invalid or are set both in ``[default_settings]`` and as
            inputs, or ``output_age`` is outside the output spectra ages. All the invalid inputs are reported together.
        FileNotFoundError
            If the parent directory of ``output_folder`` does not exist.
        FileExistsError
            If ``output_folder`` is not empty and ``overwrite_results`` is ``False``.

        Notes
        -----
        The time settings (``log_intervals``, ``log_age_start``, ``log_age_end``, ``log_age_step``, ``n_spectra`` and
        ``age_decimals``) must be set either in ``[default_settings]`` or as inputs, not both.

        The population is computed on the internal time grid (``times_steps``), from 0 yr in ``time_step_length``
        steps up to the first step at or above the last output age. The outputs at the ``times_spectra`` ages are
        linearly interpolated from this grid.

        Examples
        --------
        Compile a grid with logarithmic output ages and save it, overwriting a previous run:

        >>> compiler = Compiler(Z='<Z_key>', output_folder='pysb_testmod', log_intervals=True, log_age_start=6.0,
        ...                     log_age_end=7.0, log_age_step=0.1, overwrite_results=True)
        >>> compiler.population_synthesis()
        >>> compiler.save_results()

        Create the compiler from the inputs of a toml file, overwriting one of them:

        >>> compiler = Compiler.from_file('my_inputs.toml', verbose=False)
        """

        # Inherit parent functions
        ProgressTracker.__init__(self, verbose=verbose, progress_callback=progress_callback)

        # Log displays the terminal messages unless verbose is False
        self.log(f'\nLaunching pyStarburst99 grid compiler:')
        self.log(f'- Reading input configuration')

        # Functionlity attributes
        self.toml_path = None
        self._tools = None

        # Data containers
        self.cfg = None
        self.inputs = None
        self.z_model = None
        self.grids = None
        self.spec_lib = None
        self.pop = None
        self.results = None

        # User settings overwrite default settings
        self.toml_path = Path(TOML_PATH) if toml_path is None else Path(toml_path)
        default_settings = load_cfg(self.toml_path)['default_settings']

        # User inputs (None if not provided)
        user_settings = {'Z': Z, 'output_folder': output_folder, 'track_choice': track_choice, 'rotation': rotation,
                         'SED_library': SED_library, 'spectra_library': spectra_library,
                         'mass_loss_rate_option': mass_loss_rate_option, 'star_formation_option': star_formation_option,
                         'output_age': output_age, 'plot_results_check': plot_results_check,
                         'input_imf_option': input_imf_option, 'stochastic_imf_file': stochastic_imf_file,
                         'M_total': M_total, 'SFR': SFR, 'IMF_exponents': IMF_exponents,
                         'IMF_mass_limits': IMF_mass_limits, 'maximum_SN_mass': maximum_SN_mass,
                         'time_step_length': time_step_length, 'run_speed_mode': run_speed_mode,
                         'log_intervals': log_intervals, 'log_age_start': log_age_start, 'log_age_end': log_age_end,
                         'log_age_step': log_age_step, 'n_spectra': n_spectra, 'age_decimals': age_decimals,
                         'beta_tools': beta_tools, 'overwrite_results': overwrite_results}

        # User settings overwrite default settings
        self.cfg = merge_settings(user_settings, default_settings, self.toml_path)

        # Container for the model inputs: user inputs overwrite the default values
        self.inputs = SimpleNamespace(**self.cfg)

        self.log(f'- Inputs review')
        check_options(self.inputs)
        check_time_settings(self.inputs, self.toml_path.name)
        check_output_age(self.inputs.output_age, self.inputs.times_spectra)
        check_output_folder(self.inputs.output_folder, self.inputs.overwrite_results)

        # Experimental functions for a faster treatment
        self._tools = SimpleNamespace(**{name: getattr(tools_experimental if self.inputs.beta_tools else tools, name) for name in BETA_TOOLS})

        return

    @classmethod
    def from_file(cls, cfg_file, section='inputs', **kwargs):

        """
        Create a pyStarburst99 grid compiler from the inputs in a toml file.

        The keys of the section have the same names as the inputs of the class. Any input which is not in the section
        takes its value from the ``[default_settings]`` section of the library configuration file (``pystarburst99.toml``), as
        in the class initialization.

        Parameters
        ----------
        cfg_file : str or pathlib.Path
            Toml file with the user inputs. The metallicity ``Z`` is required.
        section : str, optional
            Table of the toml file with the inputs. Default is ``"inputs"``.
        **kwargs
            Inputs which overwrite those in the file (e.g. ``verbose=False``).

        Returns
        -------
        Compiler
            Compiler created with the merged inputs.

        Raises
        ------
        KeyError
            If ``section`` is not in the toml file.
        ValueError
            If ``Z`` is not in the section or in ``kwargs``, or if the inputs are invalid (see ``Compiler``).

        Examples
        --------
        Create the compiler from the ``[inputs]`` table of a toml file:

        >>> compiler = Compiler.from_file('my_inputs.toml')

        Use another table and overwrite one of the inputs:

        >>> compiler = Compiler.from_file('my_inputs.toml', section='IZw18', output_folder='pysb_testmod')
        """

        with open(cfg_file, 'rb') as f:
            cfg = tomllib.load(f)

        if section not in cfg:
            tables = [k for k, v in cfg.items() if isinstance(v, dict)]
            raise KeyError(f'Section [{section}] not found in "{cfg_file}". Available sections: {tables}')

        inputs = {**cfg[section], **kwargs}
        if 'Z' not in inputs:
            raise ValueError(f'The metallicity "Z" must be provided in the section [{section}] of "{cfg_file}"')

        return cls(**inputs)

    def __repr__(self):

        inp = self.inputs

        # Model configuration
        track_str = inp.track_choice if inp.track_choice == 'Pauli' else f'{inp.track_choice} ({"rotating" if inp.rotation else "non-rotating"})'
        mass_loss_str = f' (Mdot {inp.mass_loss_rate_option})' if inp.spectra_library == 'PoWR' else ''
        imf_str = f'{inp.stochastic_imf_file} (stochastic)' if inp.input_imf_option else \
                  f'exponents {inp.IMF_exponents}, limits {list(inp.IMF_mass_limits)} Msol'
        if inp.star_formation_option == 'ISF':
            sf_str = f'ISF, M_total = {inp.M_total:.2e} Msol'
        else:
            sf_str = f'CSF, SFR = {inp.SFR} Msol/yr ({inp.M_total:.2e} Msol per step)'

        # Output spectra ages
        t_spec = inp.times_spectra
        step_str = f'{np.diff(np.log10(t_spec)).mean():.3g} dex' if inp.log_intervals else \
                   f'{np.diff(t_spec).mean() / 1e6:.3g} Myr'
        spacing_str = 'log' if inp.log_intervals else 'linear'

        return (f'\n{"=" * 60}\n'
                f'  pyStarburst99Compiler\n'
                f'{"=" * 60}\n'
                f'  Star formation   : {sf_str}\n'
                f'  Metallicity      : {inp.Z} (Z = {pySB99_cfg[inp.Z]["Z_value"]})\n'
                f'  Tracks           : {track_str}\n'
                f'  SED library      : {inp.SED_library}\n'
                f'  Spectra library  : {inp.spectra_library}{mass_loss_str}\n'
                f'  IMF              : {imf_str}\n'
                f'  Max SN mass      : {inp.maximum_SN_mass} Msol\n'
                f'  M_total          : {inp.M_total:.2e} Msol\n'
                f'  Rotation         : {inp.rotation}\n'
                f'  Input IMF file   : {inp.input_imf_option}\n'
                f'{"-" * 60}\n'
                f'  Time grid\n'
                f'    time steps     : {len(inp.times_steps)} x {inp.time_step_length / 1e6:.2f} Myr '
                f'(0 - {inp.times_steps[-1] / 1e6:.1f} Myr)\n'
                f'    spectra        : {len(t_spec)} x {step_str} ({spacing_str}, '
                f'{t_spec[0] / 1e6:.4g} - {t_spec[-1] / 1e6:.4g} Myr)\n'
                f'    run speed mode : {inp.run_speed_mode}\n'
                f'    beta tools     : {inp.beta_tools}\n'
                f'{"-" * 60}\n'
                f'  Outputs\n'
                f'    folder         : {Path(inp.output_folder).resolve()}\n'
                f'    output age     : {inp.output_age} Myr\n'
                f'    plots          : {inp.plot_results_check}\n'
                f'{"=" * 60}\n')

    def load_metallicity_data(self):

        """
        Load the evolutionary tracks and spectral libraries for the input metallicity.

        The file names are read from the metallicity section (``Z``) of the configuration file
        (``pystarburst99.toml``). The data folder is ``MODEL_FOLDER``, absolute or relative to the folder of the
        configuration file. The model is also stored in the ``.z_model`` attribute.

        Returns
        -------
        types.SimpleNamespace
            Model data with the same variable names as in the original pySB99.py:

            - Evolutionary tracks: ``evo_tracks``, ``mass_grid`` and ``minimum_wr_mass``, for the tracks selected by
              ``track_choice`` and ``rotation``.
            - Low-resolution OB spectra: ``spectra_grid_file``, the path of the ``SED_library`` grid. With ``"FW"`` and
              an upper IMF mass above 120 Msol, the FastWind very massive star grid (``FW_VMS``) is used, if available.
            - High-resolution OB spectra: ``hires_wave_grid``, ``hires_teffs`` and ``hires_loggs``, available for both
              spectra libraries, and ``empty_hires_flux`` (null flux on the ``hires_wave_grid``).
            - Low-mass stars: ``lowmass_params`` and ``lowmass_flux``, plus the high-resolution ``lowmass_hires_wave_grid``,
              ``lowmass_hires_flux``, ``lowmass_hires_cont_flux`` and ``lowmass_hires_params``. These always come from
              the WMbasic (IFA) files, whatever the ``spectra_library``.
            - Wolf-Rayet stars: ``WN_spec_params``, ``WN_spectra``, ``WC_spec_params`` and ``WC_spectra`` (CMFGEN), and
              the same with the ``_powr`` suffix (PoWR).
            - ``file_path``: the data folder of the metallicity.

            The entries which do not apply to the chosen ``spectra_library`` are ``None``: ``hires_flux``,
            ``hires_cont_flux`` and ``hires_params`` for ``"PoWR"``, and ``hires_logls``, ``hires_spec_reformed``,
            ``hires_radii`` and ``hires_spec_norm`` for ``"WM"``.

        Raises
        ------
        FileNotFoundError
            If the data folder of the metallicity does not exist.

        Notes
        -----
        The track, SED library and IMF options were validated at the compiler initialization (``check_options``), so
        they are not checked again here.

        For ``"PoWR"``, ``hires_wave_grid`` is the common 920-3100 Angstrom grid (10000 points) to which all the spectra
        were resampled, and ``hires_radii`` (in solar radii) is calculated from ``hires_logls`` and ``hires_teffs``.

        Examples
        --------
        Load the data of the compiler metallicity:

        >>> z_model = compiler.load_metallicity_data()
        """

        inp = self.inputs
        z_cfg = pySB99_cfg[inp.Z]
        self.log(f'- Reading model data for metallicity Z = {inp.Z}.')

        # Data folder (absolute, or relative to the configuration file)
        data_folder = MODEL_FOLDER
        if not data_folder.is_absolute():
            data_folder = self.toml_path.parent / data_folder
        file_path = data_folder / z_cfg['folder']
        if not file_path.is_dir():
            raise FileNotFoundError(f'Model data folder for Z = {inp.Z} not found: {file_path}')

        # Evolutionary tracks (the track and IMF options were checked at initialization)
        track_key = 'Pauli' if inp.track_choice == 'Pauli' else ('GENEC_v40' if inp.rotation else 'GENEC_v00')
        tracks = z_cfg['tracks'][track_key]

        # Low-resolution OB spectra (FastWind very massive star grid if the IMF goes above 120 Msol)
        SED = z_cfg['SED']
        SED_key = 'FW_VMS' if (inp.SED_library == 'FW' and inp.IMF_mass_limits[-1] > 120.0 and 'FW_VMS' in SED) \
                  else inp.SED_library

        z_model = SimpleNamespace(file_path=file_path,
                                  mass_grid=tracks['mass_grid'],
                                  evo_tracks=np.load(file_path / tracks['file']),
                                  minimum_wr_mass=tracks['minimum_wr_mass'],
                                  spectra_grid_file=str(file_path / SED[SED_key]),
                                  hires_flux=None, hires_cont_flux=None, hires_params=None,
                                  hires_logls=None, hires_spec_reformed=None, hires_radii=None, hires_spec_norm=None)

        # High-resolution OB spectra
        hires_WM = z_cfg['hires_WM']
        if inp.spectra_library == 'WM':
            z_model.hires_wave_grid = np.load(file_path / hires_WM['wave_grid'])
            z_model.hires_flux = np.load(file_path / hires_WM['line_flux'])
            z_model.hires_cont_flux = np.load(file_path / hires_WM['cont_flux'])
            z_model.hires_params = np.load(file_path / hires_WM['params'])
            z_model.hires_teffs = z_model.hires_params[:, 0]
            z_model.hires_loggs = z_model.hires_params[:, 1]

        else:  # PoWR
            hires_PoWR = z_cfg['hires_PoWR'][inp.mass_loss_rate_option]
            z_model.hires_wave_grid = np.linspace(920, 3100, 10000)
            z_model.hires_teffs = np.genfromtxt(file_path / hires_PoWR['teffs'])
            z_model.hires_loggs = np.genfromtxt(file_path / hires_PoWR['loggs'])
            z_model.hires_logls = np.genfromtxt(file_path / hires_PoWR['logls'])
            z_model.hires_spec_reformed = np.genfromtxt(file_path / hires_PoWR['calib'])  # each spectrum is a row
            z_model.hires_radii = (((3.828*10**(26)*10**z_model.hires_logls)/(4*3.14*5.67*10**-8*z_model.hires_teffs**4))**0.5)/(7*10**8)
            z_model.hires_spec_norm = np.genfromtxt(file_path / hires_PoWR['norm'])  # each spectrum is a row

        z_model.empty_hires_flux = np.full_like(z_model.hires_wave_grid, 0.0)

        # The low-mass high-resolution spectra always come from the WMbasic (IFA) files
        z_model.lowmass_hires_wave_grid = np.load(file_path / hires_WM['wave_grid'])
        z_model.lowmass_hires_flux = np.load(file_path / hires_WM['line_flux'])
        z_model.lowmass_hires_cont_flux = np.load(file_path / hires_WM['cont_flux'])
        z_model.lowmass_hires_params = np.load(file_path / hires_WM['params'])

        # Low-mass stars
        lowmass = z_cfg['lowmass']
        z_model.lowmass_params = np.load(file_path / lowmass['params'])
        z_model.lowmass_flux = np.load(file_path / lowmass['flux'])

        # Wolf-Rayet stars
        WR = z_cfg['WR']
        z_model.WN_spec_params = np.load(file_path / WR['WN_params_cmfgen'])
        z_model.WN_spectra = np.load(file_path / WR['WN_spectra_cmfgen'], allow_pickle=True)
        z_model.WC_spec_params = np.load(file_path / WR['WC_params_cmfgen'])
        z_model.WC_spectra = np.load(file_path / WR['WC_spectra_cmfgen'], allow_pickle=True)
        z_model.WN_spec_params_powr = np.load(file_path / WR['WN_params_powr'])
        z_model.WN_spectra_powr = np.load(file_path / WR['WN_spectra_powr'], allow_pickle=True)
        z_model.WC_spec_params_powr = np.load(file_path / WR['WC_params_powr'])
        z_model.WC_spectra_powr = np.load(file_path / WR['WC_spectra_powr'], allow_pickle=True)

        self.z_model = z_model

        return z_model

    def track_grids(self):
        """
        Interpolate the evolutionary tracks into the isochrone grids.

        The tracks are split per initial mass, extended with two points after the end of the evolution (luminosity
        -20, which flags the supernova) and interpolated in mass with the ``run_speed_mode`` resolution. The result
        is also stored in the ``.grids`` attribute.

        The metallicity data is loaded with ``load_metallicity_data`` if it was not before.

        Returns
        -------
        types.SimpleNamespace
            Isochrone grids, with one array per interpolated track:

            - ``grid_masses`` and ``grid_ages``: initial masses and ages of the tracks.
            - ``grid_lums`` and ``grid_temps``: luminosity and effective temperature (the end of the evolution is
              flagged with a luminosity of -20).
            - ``grid_H_abundances``, ``grid_He_abundances``, ``grid_12C_abundances``, ``grid_14N_abundances`` and
              ``grid_16O_abundances``: surface abundances.
            - ``grid_core_temps`` and ``grid_mass_loss_rates``: core temperature and mass loss rate.

            It also includes ``timestep_mass_ind``, the index of the unique initial masses below the upper limit of
            ``IMF_mass_limits``, which is used to read the stellar parameters at each time step.


        Examples
        --------
        Compute the isochrone grids of the compiler:

        >>> grids = compiler.track_grids()
        """

        z_model = self.z_model if self.z_model is not None else self.load_metallicity_data()
        inp = self.inputs
        self.log(f'- Interpolating the evolutionary tracks ({inp.run_speed_mode} mode)')

        if inp.beta_tools:
            grids = SimpleNamespace(**interpolate_tracks_exp(z_model.evo_tracks, len(z_model.mass_grid), inp.run_speed_mode))
        else:
            grids = interpolate_tracks(z_model, inp)

        grids.timestep_mass_ind = self._tools.get_timestep_0_ind(inp.times_steps[0], grids.grid_ages, grids.grid_masses,
                                                                 inp.times_steps, inp.IMF_mass_limits)

        self.grids = grids

        return grids

    def spectral_grids(self):

        """

        Prepare the low-resolution, Wolf-Rayet, high-resolution and low-mass spectral libraries.

        The spectra are reformatted and integrated so that they can be assigned to the stars at each time step of
        ``population_synthesis``. The result is also stored in the ``.spec_lib`` attribute.

        The metallicity data is loaded with ``load_metallicity_data`` if it was not before.

        Returns
        -------
        types.SimpleNamespace
            Spectral libraries with the same names as in pySB99.py:

            - Low-resolution OB spectra (``SED_library``): ``wave_grid``, ``empty_flux`` (null flux on the
              ``wave_grid``), ``reformed_spec_grid``, ``integrated_spectra`` and the grid parameters
              ``spec_params_reform``, ``spec_params_teff``, ``spec_params_logl`` and ``spec_params_logg``.
            - ``spectrum_wave``: wavelength of the output SEDs, taken from the fifth spectrum of the reformed grid
              (``ind_spectrum = 4`` in pySB99.py).
            - Wolf-Rayet CMFGEN spectra: ``WN_`` and ``WC_`` versions of ``spec_params_reform``, ``spec_params_teff``,
              ``spec_params_logl``, ``reformed_spec_grid`` and ``integrated_spectra``.
            - Wolf-Rayet PoWR spectra: ``WN_`` and ``WC_`` versions of ``spec_params_reform_powr`` and
              ``reformed_spec_grid_powr``.
            - High-resolution OB spectra (``spectra_library``): ``hires_wave_grid``, ``hires_spec``, ``hires_cont`` and
              ``hires_int_spec`` (integrated spectra).
            - Low-mass stars: ``lowmass_params``, ``lowmass_flux``, ``lowmass_wave_grid``, ``lowmass_teffs``,
              ``lowmass_loggs``, ``lm_spec_params_logl``, ``lowmass_spec`` and ``lowmass_int_spec``.
            - ``spectra_library`` and ``minimum_wr_mass``, used by the spectra assignment.

            If ``beta_tools=True`` it also includes ``hires_library``, a ``HiresLibrary`` with the quantities of the
            high-resolution assignment which do not depend on the time step.

        Notes
        -----
        The ``spec_params_logl`` of the Wolf-Rayet and low-mass spectra are placeholders (1.0), as in pySB99.py. The
        Wolf-Rayet PoWR spectra are only reformatted here: their effective temperature, radius and integrated spectra
        are computed in ``assign_spectra_to_grid_hires``.

        For ``"PoWR"``, the first 300 spectra are scaled with their radius (``hires_radii``) to the flux at 10 pc
        (1e20 factor), as in pySB99.py, and ``hires_cont`` holds the normalized spectra (``hires_spec_norm``) instead of
        the continuum.

        If ``beta_tools=True``, the quantities of the high-resolution spectra assignment which do not depend on the
        time step are computed once before the time loop (``HiresLibrary``) and stored in ``spec_lib.hires_library``.

        The result can be passed to the spectra assignment functions as ``vars(spec_lib)``.

        Examples
        --------
        Prepare the spectral libraries of the compiler:

        >>> spec_lib = compiler.spectral_grids()

        """

        z_model = self.z_model if self.z_model is not None else self.load_metallicity_data()
        inp = self.inputs
        self.log(f'- Preparing the spectral libraries ({inp.SED_library} SED, {inp.spectra_library} high resolution)')

        # Low-resolution OB spectra
        spec_params, spectra = read_spectra_grid(z_model.spectra_grid_file)
        reformed_spec_grid, spec_params_reform, spec_params_teff, spec_params_logl, spec_params_logg = reform_spec_grid(spectra, spec_params)
        wave_grid = reformed_spec_grid[0][:,0]
        empty_flux = np.full_like(wave_grid, 0.0)
        integrated_spectra = integrate_spec_grid(reformed_spec_grid)

        # Output SED wavelength (ind_spectrum = 4 in pySB99.py)
        spectrum_wave = reformed_spec_grid[4][:,0]

        # Wolf-Rayet CMFGEN spectra
        WN_reformed_spec_grid, WN_spec_params_reform, WN_spec_params_teff = reform_spec_grid_WR(z_model.WN_spectra, z_model.WN_spec_params)
        WC_reformed_spec_grid, WC_spec_params_reform, WC_spec_params_teff = reform_spec_grid_WR(z_model.WC_spectra, z_model.WC_spec_params)
        WN_integrated_spectra = integrate_spec_grid(WN_reformed_spec_grid)
        WC_integrated_spectra = integrate_spec_grid(WC_reformed_spec_grid)
        WN_spec_params_logl = np.full_like(WN_spec_params_teff, 1.)
        WC_spec_params_logl = np.full_like(WC_spec_params_teff, 1.)

        # Wolf-Rayet PoWR spectra (assign_spectra_to_grid_hires computes their teff, radius and integrated spectra)
        WN_reformed_spec_grid_powr, WN_spec_params_reform_powr = reform_spec_grid_powr(z_model.WN_spectra_powr, z_model.WN_spec_params_powr)[:2]
        WC_reformed_spec_grid_powr, WC_spec_params_reform_powr = reform_spec_grid_powr(z_model.WC_spectra_powr, z_model.WC_spec_params_powr)[:2]

        # High-resolution OB spectra
        if inp.spectra_library == 'PoWR':
            #using script read_powr_grid.py, all spectra have been resampled to a common wavelength grid defined below
            hires_spec1 = []
            hires_cont = [] #hires_spec_norm this is really instead of cont
            for i in range(len(z_model.hires_spec_reformed)):
                hires_spec1.append(np.column_stack((z_model.hires_wave_grid, z_model.hires_spec_reformed[i])))
                hires_cont.append(np.column_stack((z_model.hires_wave_grid, z_model.hires_spec_norm[i])))

            hires_spec = []
            for i in range(len(hires_spec1)):
                if i < 300.0:
                    hires_flux = hires_spec1[i][:,1]*(((3.0856e9)**2.0)/((z_model.hires_radii[i]*6.96e10)**2.0))*1e20
                else:
                    hires_flux = hires_spec1[i][:,1]
                hires_spec.append(np.column_stack((z_model.hires_wave_grid, hires_flux)))

        else:  # WM
            hires_spec = []
            hires_cont = []
            for i in range(len(z_model.hires_flux)):
                hires_spec.append(np.column_stack((z_model.hires_wave_grid, z_model.hires_flux[i])))
                hires_cont.append(np.column_stack((z_model.hires_wave_grid, z_model.hires_cont_flux[i])))

        hires_int_spec = integrate_spec_grid(hires_spec)

        # Low-mass stars spectra
        lowmass_teffs = z_model.lowmass_params[:,0]
        lowmass_loggs = z_model.lowmass_params[:,1]
        lowmass_wave_grid = wave_grid
        lowmass_spec = []

        for i in range(len(z_model.lowmass_flux)):
            lowmass_spec.append(np.column_stack((lowmass_wave_grid, z_model.lowmass_flux[i])))
        lowmass_int_spec = integrate_spec_grid(lowmass_spec)
        lm_spec_params_logl = np.full_like(lowmass_teffs, 1.)

        spec_lib = SimpleNamespace(wave_grid=wave_grid, empty_flux=empty_flux, minimum_wr_mass=z_model.minimum_wr_mass,
                                   spectra_library=inp.spectra_library, hires_wave_grid=z_model.hires_wave_grid,
                                   reformed_spec_grid=reformed_spec_grid, integrated_spectra=integrated_spectra,
                                   spec_params_reform=spec_params_reform, spec_params_teff=spec_params_teff,
                                   spec_params_logl=spec_params_logl, spec_params_logg=spec_params_logg,
                                   spectrum_wave=spectrum_wave,
                                   WN_spec_params_reform=WN_spec_params_reform, WN_spec_params_teff=WN_spec_params_teff,
                                   WN_spec_params_logl=WN_spec_params_logl, WN_reformed_spec_grid=WN_reformed_spec_grid,
                                   WN_integrated_spectra=WN_integrated_spectra,
                                   WC_spec_params_reform=WC_spec_params_reform, WC_spec_params_teff=WC_spec_params_teff,
                                   WC_spec_params_logl=WC_spec_params_logl, WC_reformed_spec_grid=WC_reformed_spec_grid,
                                   WC_integrated_spectra=WC_integrated_spectra,
                                   WN_spec_params_reform_powr=WN_spec_params_reform_powr,
                                   WN_reformed_spec_grid_powr=WN_reformed_spec_grid_powr,
                                   WC_spec_params_reform_powr=WC_spec_params_reform_powr,
                                   WC_reformed_spec_grid_powr=WC_reformed_spec_grid_powr,
                                   lowmass_params=z_model.lowmass_params, lowmass_flux=z_model.lowmass_flux,
                                   lowmass_wave_grid=lowmass_wave_grid, lowmass_teffs=lowmass_teffs,
                                   lowmass_loggs=lowmass_loggs, lm_spec_params_logl=lm_spec_params_logl,
                                   lowmass_spec=lowmass_spec, lowmass_int_spec=lowmass_int_spec,
                                   hires_spec=hires_spec, hires_cont=hires_cont, hires_int_spec=hires_int_spec,
                                   hires_library=None)

        # # Faster high-resolution assignment: quantities which do not depend on the time step are computed once
        # if inp.beta_tools:
        #     spec_lib.hires_library = HiresLibrary(vars(spec_lib))

        self.spec_lib = spec_lib

        return spec_lib

    def population_synthesis(self):

        """
        Compute the stellar population properties at each time step.

        For each internal time step (``times_steps``), the stellar parameters are read from the isochrone grids, the
        spectra are assigned to each star, and the population SED, high-resolution spectrum, ionising fluxes, winds,
        UV slope, equivalent widths, colours and supernovae are computed. The result is also stored in the ``.pop``
        attribute.

        The metallicity data, isochrone grids and spectral libraries are computed with ``load_metallicity_data``,
        ``track_grids`` and ``spectral_grids`` if they were not before.

        Returns
        -------
        types.SimpleNamespace
            The population properties. Each ``population_*`` attribute is a list with one entry per time step:

            - Stellar parameters: ``population_ages``, ``population_masses``, ``population_temps``,
              ``population_teffs``, ``population_uncorr_teffs``, ``population_radii``, ``population_lums``,
              ``population_loggs`` and the surface abundances ``population_H_abundances``,
              ``population_12C_abundances``, ``population_14N_abundances`` and ``population_16O_abundances``.
            - Assigned spectra: ``population_assigned_spec_teffs``, ``population_assigned_spec_logls``,
              ``population_assigned_spec_loggs``, ``population_assigned_spectra`` and ``population_choice_iterations``.
            - Low-resolution SED: ``population_flux_iterations``, ``population_flux_total_iterations`` (with the
              nebular continuum) and ``population_continuum_iterations``.
            - High-resolution spectrum: ``population_hires_flux_iterations``, ``population_hires_flux_wneb_iterations``,
              ``population_hires_flux_norm_iterations`` and ``population_hires_nebular_iterations``.
            - Ionising fluxes: ``population_ion_L_flux_iterations``, ``population_ion_HI_flux_iterations``,
              ``population_ion_HEI_flux_iterations`` and ``population_ion_HEII_flux_iterations``.
            - Winds: ``population_mass_loss_rates``, ``population_vinfs``, ``population_vescs``,
              ``population_windmoms`` and ``population_windpowers`` (and their ``_calc``, ``_vink``, ``_leuven`` and
              ``_xshootu`` variants).
            - Spectral indicators: ``population_uv_slopes_beta``, ``population_Ha_ew``, ``population_Hb_ew``,
              ``population_Pb_ew``, ``population_Bg_ew`` and the magnitudes ``population_Vmag``, ``population_Umag``,
              ``population_Imag``, ``population_Bmag`` and ``population_absVmag``.
            - Supernovae: ``population_SN_ind`` and ``population_SN_masses`` per time step, plus ``SN_rates`` and
              ``SN_rates_csf_calc``.

            It also includes ``initial_masses`` and ``No_stars``, the initial masses of the interpolated tracks and
            the number of stars per mass bin, both from the IMF at the first time step.

        Notes
        -----
        The number of stars per mass bin is calculated once, at the first time step, from ``IMF_exponents``,
        ``IMF_mass_limits`` and ``M_total``. If ``input_imf_option=True``, it is counted from the masses in
        ``stochastic_imf_file`` instead.

        The functions of ``tools_experimental`` are used for the track selection, the spectra assignment and the
        spectral synthesis if ``beta_tools=True``, otherwise those of ``tools`` are used. The results are identical.

        Stars with a luminosity of -20 (end of the evolution) and an initial mass not above ``maximum_SN_mass`` are
        counted as supernovae.

        Examples
        --------
        Run the compilation.

        >>> compiler = Compiler(Z='<Z_key>', output_folder='pysb_testmod', log_age_start=-np.inf, log_age_end=7.0,
        ...                     log_age_step=5.0)
        >>> pop = compiler.population_synthesis()
        """

        # Track the compilation speed
        start_time = time.time()

        # User inputs and model functions (tools.py, or tools_fast.py if beta_tools)
        inp = self.inputs
        T = self._tools

        # Run the track and library interpolation
        z_model = self.z_model if self.z_model is not None else self.load_metallicity_data()
        grids = self.grids if self.grids is not None else self.track_grids()

        # Compute the population properties
        self.log(f'- Computing the population at {len(inp.times_steps)} time steps '
              f'({inp.times_steps[0] / 1e6:g} - {inp.times_steps[-1] / 1e6:g} Myr, step = {inp.time_step_length / 1e6:g} Myr)')
        self.log(f'-- Output spectra: {len(inp.times_spectra)} ages, {inp.times_spectra[0] / 1e6:.4g} - {inp.times_spectra[-1] / 1e6:.4g} Myr '
              f'({"log" if inp.log_intervals else "linear"} spacing)')
        spec_lib = self.spec_lib if self.spec_lib is not None else self.spectral_grids()

        # Experimental high-resolution calculation
        spec_lib.hires_library = HiresLibrary(vars(spec_lib)) if inp.beta_tools else None

        # Time evolution of the population properties (one entry per time step)
        pop = SimpleNamespace(**{name: [] for name in (
            'population_flux_iterations', 'population_flux_iterations_send', 'population_ion_flux_iterations',
            'population_ion_L_flux_iterations', 'population_ion_HI_flux_iterations',
            'population_ion_HEI_flux_iterations', 'population_ion_HEII_flux_iterations',
            'population_ages', 'population_masses', 'population_masses_test', 'population_temps', 'population_teffs',
            'population_uncorr_teffs', 'population_radii', 'population_lums', 'population_H_abundances',
            'population_12C_abundances', 'population_14N_abundances', 'population_16O_abundances',
            'population_assigned_spec_teffs', 'population_assigned_spec_logls', 'population_assigned_spec_loggs',
            'population_assigned_spectra', 'population_loggs', 'population_mass_loss_rates',
            'population_vinfs', 'population_vescs', 'population_windmoms', 'population_windmoms_leuven',
            'population_windmoms_xshootu', 'population_windmoms_vink', 'population_vinfs_vink', 'population_mdots_vink',
            'population_windmoms_calc', 'population_windpowers', 'population_windpowers_calc',
            'population_windpowers_xshootu', 'population_co_tests', 'population_specsyn_loggs',
            'population_specsyn_teffs', 'pop_age_test', 'pop_mass_test', 'population_SN_ind', 'population_SN_masses',
            'population_continuum_iterations',
            'population_hires_flux_iterations', 'population_hires_flux_iterations_send',
            'population_hires_flux_wneb_iterations', 'population_hires_flux_wneb_iterations_send',
            'population_hires_flux_norm_iterations', 'population_hires_nebular_iterations',
            'population_hires_assigned_spec_teff', 'population_hires_assigned_spec_logg', 'population_hires_choice',
            'population_choice_iterations', 'population_uv_slopes_x', 'population_uv_slopes_y',
            'population_uv_slopes_beta', 'population_Ha_ew', 'population_Hb_ew', 'population_Pb_ew', 'population_Bg_ew',
            'population_flux_total_iterations', 'population_flux_total_iterations_send', 'population_Vmag',
            'population_Umag', 'population_Imag', 'population_Bmag', 'population_absVmag', 'population_Ha_luminosity',
            'population_Ha_continuum_flux')})

        with np.errstate(divide='ignore'):
            for i in self.track(range(len(inp.times_steps)), desc='-- Time steps'):
                timestep_ex = inp.times_steps[i]
                timestep_ages_final,timestep_temps_final,timestep_lums_final,timestep_masses_final,timestep_H_abundances_final, timestep_loggs_final, timestep_mass_loss_rates_final, timestep_12C_abundances_final, timestep_14N_abundances_final, timestep_16O_abundances_final, timestep_mass_test, timestep_cnr, timestep_coher, timestep_uncorr_teffs = T.get_timestep_params(timestep_ex, grids.timestep_mass_ind, inp.track_choice, vars(grids))

                # Number of stars per mass bin from the IMF at the first time step
                if i == 0:
                    initial_masses = timestep_masses_final
                    No_stars, c_masses, dens, xmhigh, xmlow = calc_Nostars(timestep_masses_final, inp.IMF_exponents, inp.IMF_mass_limits, inp.M_total) #second print

                    if inp.input_imf_option == True:
                        stochastic_imf = np.load(inp.stochastic_imf_file)

                        # Array to store counts
                        No_stars_stochastic = np.zeros(len(xmlow), dtype=int)

                        # Count values in each bin
                        for j in range(len(xmlow)):
                            No_stars_stochastic[j] = np.sum((stochastic_imf >= xmlow[j]) & (stochastic_imf < xmhigh[j]))

                        No_stars = No_stars_stochastic

                timestep_teffs_final = 10**timestep_temps_final
                timestep_radii_final = compute_radii(timestep_temps_final,timestep_lums_final)
                specsyn_teffs, specsyn_loggs, specsyn_radii, specsyn_cotests, specsyn_bbfluxes = get_specsyn_params(timestep_temps_final, timestep_masses_final, timestep_lums_final)
                assigned_integrated_spectra,assigned_spec_teff,assigned_spectra, population_choice, assigned_spec_logl, assigned_spec_logg = T.assign_spectra_to_grid_WR(timestep_temps_final,spec_lib.spec_params_reform, spec_lib.spec_params_teff, spec_lib.spec_params_logl, timestep_lums_final, timestep_H_abundances_final, spec_lib.spec_params_logg, timestep_masses_final, initial_masses, specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher, No_stars, timestep_loggs_final, vars(spec_lib))
                population_flux, assigned_flux_scaled = T.specsyn(assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, specsyn_radii, No_stars)
                timestep_vinfs, timestep_windpowers_calc, timestep_windpowers, timestep_windmoms_calc, timestep_windmoms, timestep_windmoms_vink, timestep_vinfs_vink, timestep_mdots_vink, timestep_vescs, timestep_windmoms_leuven, timestep_windmoms_xshootu, timestep_windpowers_xshootu = calc_wind(timestep_temps_final, timestep_lums_final, timestep_masses_final, timestep_mass_loss_rates_final, timestep_H_abundances_final, timestep_12C_abundances_final, timestep_14N_abundances_final, timestep_16O_abundances_final, initial_masses, No_stars, timestep_radii_final, timestep_cnr, z_model.minimum_wr_mass, inp.Z)

                # Low-resolution SED, ionising fluxes and nebular continuum
                pop.population_flux_iterations.append(population_flux)
                pop.population_flux_iterations_send.append(np.log10(population_flux)+20.)
                population_ion_flux = ionise(spec_lib.spectrum_wave, population_flux, 2)
                pop.population_ion_flux_iterations.append(population_ion_flux)
                pop.population_ion_L_flux_iterations.append(population_ion_flux[0])
                pop.population_ion_HI_flux_iterations.append(population_ion_flux[1][0])
                pop.population_ion_HEI_flux_iterations.append(population_ion_flux[2][0])
                pop.population_ion_HEII_flux_iterations.append(population_ion_flux[3][0])
                population_continuum = continuum(population_ion_flux[1][0])
                continuum_resampled = np.interp(spec_lib.reformed_spec_grid[0][:,0], population_continuum[:,0], population_continuum[:,1])
                population_flux_total = population_flux + continuum_resampled

                # Equivalent widths, UV slope and colours
                timestep_Ha_ew, timestep_Hb_ew, timestep_Pb_ew, timestep_Bg_ew, timestep_Ha_luminosity, timestep_Ha_continuum_flux = get_ew(spec_lib.reformed_spec_grid[0][:,0], population_flux_total, population_ion_flux[1][0])
                pop.population_choice_iterations.append(population_choice)
                timestep_uv_slope_x, timestep_uv_slope_y, timestep_uv_slope_beta = get_uv_slope(np.log10(spec_lib.reformed_spec_grid[0][:,0]), np.log10(population_flux_total)+20.)
                pop.population_uv_slopes_x.append(timestep_uv_slope_x)
                pop.population_uv_slopes_y.append(timestep_uv_slope_y)
                pop.population_uv_slopes_beta.append(timestep_uv_slope_beta)
                pop.population_Ha_ew.append(timestep_Ha_ew)
                pop.population_Hb_ew.append(timestep_Hb_ew)
                pop.population_Pb_ew.append(timestep_Pb_ew)
                pop.population_Bg_ew.append(timestep_Bg_ew)
                pop.population_flux_total_iterations.append(population_flux_total)
                timestep_Vmag, timestep_Umag, timestep_Imag, timestep_Bmag, timestep_absVmag = colours(population_flux_total, spec_lib.wave_grid)
                pop.population_flux_total_iterations_send.append(np.log10(population_flux_total)+20.)
                pop.population_Vmag.append(timestep_Vmag)
                pop.population_Umag.append(timestep_Umag)
                pop.population_Imag.append(timestep_Imag)
                pop.population_Bmag.append(timestep_Bmag)
                pop.population_absVmag.append(timestep_absVmag)
                pop.population_Ha_luminosity.append(timestep_Ha_luminosity)
                pop.population_Ha_continuum_flux.append(timestep_Ha_continuum_flux)

                # High-resolution spectrum (WM or PoWR)
                hires_assigned_integrated_spectra,hires_assigned_spec_teff,hires_assigned_spec_logg,hires_assigned_spectra,hires_assigned_cont, assigned_hires_choice = T.assign_spectra_to_grid_hires(timestep_temps_final, z_model.hires_teffs, z_model.hires_loggs, timestep_lums_final, timestep_H_abundances_final, timestep_masses_final, initial_masses, specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher, timestep_radii_final, vars(spec_lib))
                hires_population_flux, hires_population_flux_wneb, hires_population_flux_norm, hires_population_nebular = T.specsyn_hires(assigned_integrated_spectra, specsyn_bbfluxes, hires_assigned_spectra, hires_assigned_cont, specsyn_radii, No_stars, population_continuum, assigned_hires_choice, z_model.hires_wave_grid, inp.spectra_library)
                pop.population_hires_flux_iterations.append(hires_population_flux)
                pop.population_hires_flux_iterations_send.append(np.log10(hires_population_flux+1.0e-35)+20.)
                pop.population_hires_flux_wneb_iterations.append(hires_population_flux_wneb)
                pop.population_hires_flux_wneb_iterations_send.append(np.log10(hires_population_flux_wneb+1.0e-35)+20.)
                pop.population_hires_flux_norm_iterations.append(hires_population_flux_norm)
                pop.population_hires_assigned_spec_teff.append(hires_assigned_spec_teff)
                if inp.spectra_library == 'PoWR':
                    pop.population_hires_assigned_spec_logg.append(hires_assigned_spec_logg)
                pop.population_hires_choice.append(assigned_hires_choice)
                pop.population_hires_nebular_iterations.append(hires_population_nebular)

                # Stars which have gone supernova
                ind_SN_masses = [i for i in range(len(timestep_lums_final)) if timestep_lums_final[i] == -20.0 and initial_masses[i] <= inp.maximum_SN_mass]

                # Stellar parameters and winds
                pop.population_ages.append(timestep_ages_final)
                pop.population_masses.append(timestep_masses_final)
                pop.population_temps.append(timestep_temps_final)
                pop.population_teffs.append(timestep_teffs_final)
                pop.population_lums.append(timestep_lums_final)
                pop.population_radii.append(timestep_radii_final)
                pop.population_H_abundances.append(timestep_H_abundances_final)
                pop.pop_age_test.append(timestep_ages_final[-1])
                pop.pop_mass_test.append(timestep_masses_final[-1])
                pop.population_assigned_spectra.append(assigned_spectra)
                pop.population_assigned_spec_teffs.append(assigned_spec_teff)
                pop.population_assigned_spec_logls.append(assigned_spec_logl)
                pop.population_assigned_spec_loggs.append(assigned_spec_logg)
                pop.population_loggs.append(timestep_loggs_final)
                pop.population_mass_loss_rates.append(timestep_mass_loss_rates_final)
                pop.population_12C_abundances.append(timestep_12C_abundances_final)
                pop.population_14N_abundances.append(timestep_14N_abundances_final)
                pop.population_16O_abundances.append(timestep_16O_abundances_final)
                pop.population_uncorr_teffs.append(timestep_uncorr_teffs)
                pop.population_vinfs.append(timestep_vinfs)
                pop.population_vescs.append(timestep_vescs)
                pop.population_windmoms.append(timestep_windmoms)
                pop.population_windmoms_vink.append(timestep_windmoms_vink)
                pop.population_windmoms_leuven.append(timestep_windmoms_leuven)
                pop.population_windmoms_xshootu.append(timestep_windmoms_xshootu)
                pop.population_windpowers_xshootu.append(timestep_windpowers_xshootu)
                pop.population_vinfs_vink.append(timestep_vinfs_vink)
                pop.population_mdots_vink.append(timestep_mdots_vink)
                pop.population_windmoms_calc.append(timestep_windmoms_calc)
                pop.population_windpowers.append(timestep_windpowers)
                pop.population_windpowers_calc.append(timestep_windpowers_calc)
                pop.population_co_tests.append(specsyn_cotests)
                pop.population_specsyn_loggs.append(specsyn_loggs)
                pop.population_specsyn_teffs.append(specsyn_teffs)
                pop.population_masses_test.append(timestep_mass_test)
                pop.population_continuum_iterations.append(population_continuum)

                SN_masses = np.array(initial_masses)[ind_SN_masses]
                pop.population_SN_masses.append(SN_masses)
                pop.population_SN_ind.append(ind_SN_masses)

        # IMF and supernova rates
        pop.initial_masses, pop.No_stars = initial_masses, No_stars
        pop.SN_rates, pop.SN_rates_csf_calc = get_SN_rate(pop.population_SN_ind, No_stars)
        self.log(f'-- Execution time: {time.time() - start_time:.2f} seconds')

        self.pop = pop

        return pop

    def save_results(self, output_age=None, include_plots=None):

        """
        Save the compiled grids, and optionally the plots, in the output_folder output folder.

        This function does not run the compilation: the metallicity data, isochrone grids, spectral libraries and
        population must have been computed before with ``load_metallicity_data``, ``track_grids``, ``spectral_grids``
        and ``population_synthesis``. The previous content of the folder is deleted. The SEDs and high-resolution
        spectra are saved at the times_spectra ages, interpolated from the internal time grid. The other outputs are
        saved at the internal time steps (timesteps.txt). The output arrays (internal time grid) are also stored in
        the .results attribute.

        Parameters
        ----------
        output_age : float, optional
            Age (Myr) of the example SED and high-resolution spectrum saved as text files and plotted. It must be within
            the output spectra ages. The plots are only generated if it is exactly one of the output spectra ages and
            internal time steps, otherwise a warning is raised. Default is the output_age input.
        include_plots : bool, optional
            Plot the results and save the figures in the output folder. Default is the plot_results_check input.

        Returns
        -------
        SimpleNamespace
            Output arrays: SED_stellar, SED_total, hires_flux, hires_flux_wneb, hires_flux_norm, Lbol, HI_flux, HeI_flux,
            HeII_flux, wind_power, wind_momentum, uv_slope, Ha_ew, Hb_ew, Pb_ew, Bg_ew, colours and SN_rate.

        Raises
        ------
        RuntimeError
            If any of the .inputs, .z_model, .grids, .spec_lib or .pop attributes is None. All the missing ones are
            reported together.
        """

        # Check that the compiler has succesfullied compiled the grids
        required = {'inputs': 'the Compiler initialization', 'z_model': 'load_metallicity_data()',
                    'grids': 'track_grids()', 'spec_lib': 'spectral_grids()', 'pop': 'population_synthesis()'}
        missing = [f'- .{name} is None. Run {step} first' for name, step in required.items()
                   if getattr(self, name) is None]

        if missing:
            raise RuntimeError('Cannot save the results before the compilation:\n' + '\n'.join(missing))

        inp = self.inputs
        output_age = inp.output_age if output_age is None else output_age
        include_plots = inp.plot_results_check if include_plots is None else include_plots
        age = check_output_age(output_age, inp.times_spectra)

        # The plots are only generated if the output age is exactly in the output time grids (checked before the run)
        ratio = np.round(inp.times_spectra / inp.time_step_length)
        plot_ages = inp.times_spectra[np.isclose(ratio * inp.time_step_length, inp.times_spectra, rtol=1e-9, atol=0)]

        if include_plots and not np.isclose(plot_ages, age, rtol=1e-9, atol=0).any():
            closest = (f'closest valid age: {plot_ages[np.argmin(np.abs(plot_ages - age))] / 1e6:.6g} Myr'
                       if plot_ages.size else 'none of the output spectra ages is an internal time step')
            warn_message(f'output_age = {output_age} Myr is not in the output time grids (times_spectra and '
                         f'times_steps): the plots are not generated ({closest})')
            include_plots = False

        # Prepare data for output files
        self.log(f'- Preparing outputs for {inp.star_formation_option} model')
        pop, z_model, spec_lib = self.pop, self.z_model, self.spec_lib
        res = prepare_outputs(pop, spec_lib, inp.star_formation_option)

        # Interpolator for the user time steps
        interp = partial(interp_time, times_steps=inp.times_steps)

        # Clearing previous results
        self.log(f'- Saving compiled grids to path "{inp.output_folder}"')
        out_folder = prepare_output_folder(inp.output_folder, inp.overwrite_results)

        # Model summary
        summary_name = f'pystarburst99v{pySB99_cfg["metadata"]["version"]}_{inp.Z}.txt'
        with open(out_folder / summary_name, 'w') as summary_file:
            summary_file.write(repr(self).strip('\n') + '\n')

        # Save the compiled grid
        out_folder = Path(inp.output_folder)
        age_label = str(int(output_age))

        with open(out_folder / 'input.txt', 'w') as inputs_file:
            inputs_file.write('M_total = ' + str(inp.M_total) + 'Msol' + '\n')
            inputs_file.write('IMF_exponents = ' + str(inp.IMF_exponents) + '\n')
            inputs_file.write('IMF_mass_limits = ' + str(tuple(inp.IMF_mass_limits)) + '\n')
            inputs_file.write('run_speed_mode = ' + inp.run_speed_mode + '\n')
            inputs_file.write('Metallicity input = ' + inp.Z + '\n')
            inputs_file.write('SED library input = ' + inp.SED_library + '\n')
            inputs_file.write('spectra library input = ' + inp.spectra_library + '\n')

            if inp.star_formation_option == 'ISF':
                inputs_file.write('Instant burst of star formation' + '\n')
            if inp.star_formation_option == 'CSF':
                inputs_file.write('Continuous star formation' + '\n')
            inputs_file.write('Rotation input = ' + str(inp.rotation == True) + '\n')

            # All the outputs are always saved (lines kept for compatibility with previous input.txt files)
            for output_type in ['Ionising flux', 'Wind', 'UV slope', 'Equivalent width', 'Hires spectra']:
                inputs_file.write(output_type + ' output choice = True' + '\n')

        np.savetxt(out_folder / 'timesteps.txt', inp.times_steps)

        # SEDs
        np.save(out_folder / 'pySB_SED_stellar.npy', interp(res.SED_stellar, inp.times_spectra))
        np.save(out_folder / 'pySB_SED_stellar_and_nebular.npy', interp(res.SED_total, inp.times_spectra))
        np.savetxt(out_folder / f'pySB_SED_stellar{age_label}_Myr.txt', np.column_stack((spec_lib.spectrum_wave, interp(res.SED_stellar, age)[0])))
        np.savetxt(out_folder / f'pySB_SED_stellar_and_nebular{age_label}_Myr.txt', np.column_stack((spec_lib.spectrum_wave, interp(res.SED_total, age)[0])))
        np.savetxt(out_folder / 'SED_wavelength.txt', spec_lib.spectrum_wave)
        np.savetxt(out_folder / 'SED_times.txt', inp.times_spectra)

        # High resolution spectra
        np.save(out_folder / 'pySB_hires_spectrum.npy', interp(res.hires_flux_wneb, inp.times_spectra))
        np.save(out_folder / 'pySB_hires_norm_spectrum.npy', interp(res.hires_flux_norm, inp.times_spectra))

        np.savetxt(out_folder / f'pySB_spectrum_{age_label}_Myr.txt', np.column_stack((z_model.hires_wave_grid, interp(res.hires_flux_wneb, age)[0])))
        np.savetxt(out_folder / f'pySB_without_neb_spectrum_{age_label}_Myr.txt', np.column_stack((z_model.hires_wave_grid, interp(res.hires_flux, age)[0])))
        np.savetxt(out_folder / f'pySB_norm_spectrum_{age_label}_Myr.txt', np.column_stack((z_model.hires_wave_grid, interp(res.hires_flux_norm, age)[0])))
        np.savetxt(out_folder / 'spectrum_wavelength.txt', z_model.hires_wave_grid)
        np.savetxt(out_folder / 'times_spectra.txt', inp.times_spectra)

        # As in pySB99.py, this file has the spectra per time step (not cumulative) and it is only saved for CSF
        if inp.star_formation_option == 'CSF':
            np.save(out_folder / 'pySB_hires_spectrum_without_neb.npy', interp(pop.population_hires_flux_iterations_send, inp.times_spectra))

        # Bolometric luminosity and ionising fluxes
        np.savetxt(out_folder / 'bololum.txt', res.Lbol)
        np.savetxt(out_folder / 'ion_flux_HI.txt', res.HI_flux)
        np.savetxt(out_folder / 'ion_flux_HeI.txt', res.HeI_flux)
        np.savetxt(out_folder / 'ion_flux_HeII.txt', res.HeII_flux)

        # Winds
        np.savetxt(out_folder / 'windpower.txt', res.wind_power)
        np.savetxt(out_folder / 'windmom.txt', res.wind_momentum)

        # UV slope
        np.savetxt(out_folder / 'uv_slope.txt', res.uv_slope)

        # Equivalent widths
        np.savetxt(out_folder / 'Haew.txt', res.Ha_ew)
        np.savetxt(out_folder / 'Hbew.txt', res.Hb_ew)
        np.savetxt(out_folder / 'Pbew.txt', res.Pb_ew)
        np.savetxt(out_folder / 'Bgew.txt', res.Bg_ew)

        # Colours
        np.savetxt(out_folder / 'colours.txt', res.colours, header='V, U, I, B, M_V')

        # Supernova rate
        np.savetxt(out_folder / 'SNrate.txt', np.column_stack((inp.times_steps, res.SN_rate)))
        self.log(f'-- All files were succesfully saved')

        # -------- Plotting the grid results
        if include_plots:
            from .plotting.plots import PySB_Plotter

            self.log(f'- Generating plots (output age is {output_age})')

            # Pack the plotting data
            results = SimpleNamespace(times=inp.times_steps, Z=inp.Z, rot=inp.rotation, output_age=output_age,
                                      sed_wave=spec_lib.wave_grid, SED_stellar=res.SED_stellar, SED_total=res.SED_total,
                                      hires_wave=z_model.hires_wave_grid, hires_flux_wneb=res.hires_flux_wneb,
                                      hires_flux_norm=res.hires_flux_norm,
                                      Lbol=res.Lbol, HI_flux=res.HI_flux, HeI_flux=res.HeI_flux, HeII_flux=res.HeII_flux,
                                      wind_power=res.wind_power, wind_momentum=res.wind_momentum, uv_slope=res.uv_slope,
                                      Ha_ew=res.Ha_ew, MV=res.colours[:, 4], SN_rate=res.SN_rate,
                                      iso_teff=pop.population_uncorr_teffs, iso_lum=pop.population_lums,
                                      iso_H=pop.population_H_abundances)

            # Plotting class
            run_label = f'_{inp.run_speed_mode}_{inp.spectra_library}-{inp.Z}_{inp.star_formation_option}_{inp.SED_library}'
            results.plot = PySB_Plotter(results, fname_suffix=run_label)

            # Save the results
            results.plot.all_results(inp.output_folder)

        self.results = res

        return res