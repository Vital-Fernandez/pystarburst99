import sys
import tomllib
import warnings
from pathlib import Path
from decimal import Decimal
from types import SimpleNamespace
import shutil

import numpy as np


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class pySB99_Error(Exception):
    """pyStarburst99 exception function"""


# Reading file with the format and export status for the measurements
_pySB99_FOLDER = Path(__file__).parent


# Read lime configuration .toml
_conf_path = _pySB99_FOLDER/'pystarburst99.toml'
with open(_conf_path, mode="rb") as fp:
    pySB99_cfg = tomllib.load(fp)


# Default location for the model data
TOML_PATH = Path(__file__).with_name('pystarburst99.toml')
MODEL_FOLDER = _pySB99_FOLDER / pySB99_cfg['metadata']['model_folder']

# Options which do not depend on the data files
TRACK_CHOICES = ('GENEC', 'Pauli')
SPECTRA_LIBRARIES = ('WM', 'PoWR')
STAR_FORMATION_OPTIONS = ('ISF', 'CSF')
RUN_SPEED_MODES = ('FAST', 'DEFAULT', 'HIGH_RES')


def _decimals(x):
    """Number of decimals of a number as Python writes it: 0.1 -> 1, 0.25 -> 2, 6.0 -> 0."""
    return max(0, -Decimal(repr(float(x))).normalize().as_tuple().exponent)


def warn_message(message, category=UserWarning):
    """Issue a warning which is displayed as 'Category: message', without the file and line of the code."""
    standard_format = warnings.formatwarning
    warnings.formatwarning = lambda msg, cat, *args, **kwargs: f'{cat.__name__}: {msg}\n'
    try:
        warnings.warn(message, category, stacklevel=2)
    finally:
        warnings.formatwarning = standard_format  # The format changes only for this warning
    return


def store_err_inputs(name, options, inp, errors):
    if getattr(inp, name) not in options:
        errors.append(f'- {name} = {getattr(inp, name)!r}. Available options: {list(options)}')
    return


def check_options(inp):
    """
    Check the model options against the available ones.

    The categorical options are compared with the values accepted by the code (``TRACK_CHOICES``,
    ``SPECTRA_LIBRARIES``, ``STAR_FORMATION_OPTIONS`` and ``RUN_SPEED_MODES``), and the options which depend on the
    metallicity (``SED_library``, ``mass_loss_rate_option``, the tracks and the upper IMF mass) with the metallicity
    section of the configuration. All the invalid inputs are reported together.

    Parameters
    ----------
    inp : types.SimpleNamespace
        Compiler inputs, after filling in the ``[default_settings]`` values.

    Raises
    ------
    ValueError
        If any of the options is not available.
    """

    errors = []

    metallicities = [k for k, v in pySB99_cfg.items() if isinstance(v, dict) and 'folder' in v]
    store_err_inputs('Z', metallicities, inp, errors)
    store_err_inputs('track_choice', TRACK_CHOICES, inp, errors)
    store_err_inputs('spectra_library', SPECTRA_LIBRARIES, inp, errors)
    store_err_inputs('star_formation_option', STAR_FORMATION_OPTIONS, inp, errors)
    store_err_inputs('run_speed_mode', RUN_SPEED_MODES, inp, errors)

    for name in ('rotation', 'beta_tools', 'log_intervals'):
        if not isinstance(getattr(inp, name), bool):
            errors.append(f'- {name} = {getattr(inp, name)!r}. Available options: [True, False]')

    if inp.input_imf_option and not Path(inp.stochastic_imf_file).is_file():
        errors.append(f'- stochastic_imf_file = {inp.stochastic_imf_file!r} not found')

    # Options which depend on the metallicity data
    if inp.Z in metallicities:
        z_cfg = pySB99_cfg[inp.Z]
        store_err_inputs('SED_library', [k for k in z_cfg['SED'] if k != 'FW_VMS'], inp, errors)
        store_err_inputs('mass_loss_rate_option', list(z_cfg['hires_PoWR']), inp, errors)

        if inp.track_choice in TRACK_CHOICES and isinstance(inp.rotation, bool):
            track_key = 'Pauli' if inp.track_choice == 'Pauli' else ('GENEC_v40' if inp.rotation else 'GENEC_v00')
            if track_key not in z_cfg['tracks']:
                errors.append(f'- There are no {track_key} tracks for Z = {inp.Z}. '
                              f'Available tracks: {list(z_cfg["tracks"])}')
            elif inp.IMF_mass_limits[-1] > z_cfg['tracks'][track_key]['max_mass']:
                errors.append(f'- IMF_mass_limits upper mass = {inp.IMF_mass_limits[-1]} Msol is above the '
                              f'{track_key} tracks limit for Z = {inp.Z} ({z_cfg["tracks"][track_key]["max_mass"]} Msol)')

    if errors:
        raise ValueError('Invalid pyStarburst99 inputs:\n' + '\n'.join(errors))

    return


def check_time_settings(inp, toml_name='pystarburst99.toml'):
    """
    Check the time settings and compute the output and internal time grids.

    The output ages (``times_spectra``) are defined by ``log_age_start``, ``log_age_end``, ``log_intervals`` and either
    ``log_age_step`` or ``n_spectra`` (see ``Compiler``). The internal grid (``times_steps``) starts at 0 yr with
    ``time_step_length`` steps and reaches the first step at or above the last output age, so the outputs can be
    interpolated at all the ``times_spectra`` ages. All the invalid inputs are reported together.

    Parameters
    ----------
    inp : types.SimpleNamespace
        Compiler inputs, after filling in the ``[default_settings]`` values. It is modified in place (see Notes).
    toml_name : str, optional
        Name of the configuration file, for the error messages. Default is ``"pystarburst99.toml"``.

    Raises
    ------
    ValueError
        If the time settings are missing, inconsistent or invalid, if ``log_age_step`` is larger than the output age
        interval, or if the rounding to ``age_decimals`` merges some of the output ages.

    Warns
    -----
    UserWarning
        If some output ages are closer than ``time_step_length``.

    Notes
    -----
    The following attributes are added to ``inp``:

    - ``times_spectra``: output spectra ages, in yr.
    - ``times_steps``: internal time grid of the population synthesis, in yr.

    For ``"CSF"``, ``inp.M_total`` is overwritten with ``time_step_length * SFR`` (mass formed per time step).
    """

    start, end, step, n, dt = inp.log_age_start, inp.log_age_end, inp.log_age_step, inp.n_spectra, inp.time_step_length
    errors = []

    for name in ('log_age_start', 'log_age_end'):
        if getattr(inp, name) is None:
            errors.append(f'- {name} must be provided (input or [default_settings] of "{toml_name}")')
    if start is not None and end is not None and end <= start:
        errors.append(f'- log_age_end = {end} must be larger than log_age_start = {start}')
    if end is not None and not np.isfinite(end):
        errors.append(f'- log_age_end = {end} must be finite')
    if start is not None and not np.isfinite(start) and (inp.log_intervals or start > 0):
        errors.append(f'- log_age_start = {start}: only -inf (age 0 yr) with linear intervals is accepted')

    if step is None and n is None:
        errors.append(f'- Provide log_age_step or n_spectra for log_intervals = {inp.log_intervals} (the '
                      f'[default_settings] value only applies to the default spacing)')
    elif step is not None and n is not None:
        errors.append(f'- Provide either log_age_step = {step} or n_spectra = {n}, not both')
    elif n is not None and (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2):
        errors.append(f'- n_spectra = {n!r} must be an integer >= 2')
    elif step is not None and inp.log_intervals and step <= 0:
        errors.append(f'- log_age_step = {step} dex must be positive')

    if inp.age_decimals is not None and (isinstance(inp.age_decimals, bool) or
                                         not isinstance(inp.age_decimals, (int, np.integer))):
        errors.append(f'- age_decimals = {inp.age_decimals!r} must be an integer')
    if not dt > 0:
        errors.append(f'- time_step_length = {dt} yr must be positive')

    if errors:
        raise ValueError('Invalid pyStarburst99 time grid:\n' + '\n'.join(errors))

    # Output ages in the spacing scale: log10(age/yr) for log intervals, yr for linear intervals
    lo, hi = (start, end) if inp.log_intervals else (10.0**start, 10.0**end)
    if n is not None:
        ages = np.linspace(lo, hi, n)
        decimals = None if inp.log_intervals else 0
    else:
        delta = step if inp.log_intervals else 10.0**step
        n_ages = int(np.floor(np.round((hi - lo) / delta, 9))) + 1
        if n_ages < 2:
            raise ValueError(f'log_age_step = {step} is larger than the output age interval '
                             f'(log_age_start = {start}, log_age_end = {end})')
        ages = lo + delta * np.arange(n_ages)
        decimals = max(_decimals(start), _decimals(step)) if inp.log_intervals else 0

    # Remove the trailing decimals of the spacing functions
    decimals = decimals if inp.age_decimals is None else inp.age_decimals
    if decimals is not None:
        ages = np.round(ages, decimals)
        if np.any(np.diff(ages) <= 0):
            raise ValueError(f'Rounding the output ages to {decimals} decimals merges some of them. Increase '
                             f'age_decimals')

    inp.times_spectra = 10**ages if inp.log_intervals else ages

    # Internal time grid, from 0 yr to the first time step at or above the last output age
    n_steps = int(np.ceil(np.round(inp.times_spectra[-1] / dt, 9))) + 1
    inp.times_steps = np.arange(n_steps) * dt

    # Output ages closer than the internal time step do not add information
    min_spacing = np.diff(inp.times_spectra).min()
    if min_spacing < dt:
        warn_message(f'Some output ages are closer ({min_spacing:.3g} yr) than the internal time step '
                     f'(time_step_length = {dt:.3g} yr): their spectra are interpolated between the same time steps')

    if inp.star_formation_option == 'CSF':
        inp.M_total = inp.time_step_length * inp.SFR

    return


def check_output_age(output_age, times_spectra):
    """
    Check that the output age is within the output spectra ages.

    Parameters
    ----------
    output_age : float
        Age in Myr of the example SED and high-resolution spectrum.
    times_spectra : numpy.ndarray
        Output spectra ages, in yr.

    Returns
    -------
    float
        Output age in yr, clipped to the output spectra ages to remove the floating point errors at the limits.

    Raises
    ------
    ValueError
        If ``output_age`` is outside the output spectra ages.
    """

    age = output_age * 1e6
    if not (times_spectra[0] * (1 - 1e-9) <= age <= times_spectra[-1] * (1 + 1e-9)):
        raise ValueError(f'output_age = {output_age} Myr is outside the output spectra ages '
                         f'[{times_spectra[0] / 1e6:.6g}, {times_spectra[-1] / 1e6:.6g}] Myr')

    return float(np.clip(age, times_spectra[0], times_spectra[-1]))


def check_output_folder(output_folder, overwrite_results=False):
    """
    Check that the output folder can be used to save the results.

    The parent directory of the folder must exist. The folder itself does not need to exist, but if it contains
    files, they are only accepted if ``overwrite_results`` is ``True``. In that case, a warning is raised to
    inform that the files will be deleted when the results are saved (see ``Compiler.prepare_output_folder``).

    Parameters
    ----------
    output_folder : str or pathlib.Path
        Folder where the results will be saved.
    overwrite_results : bool, optional
        If ``False``, an error is raised when the folder already contains files. If ``True``, a warning is raised
        instead. Default is ``False``.

    Raises
    ------
    FileNotFoundError
        If the parent directory of ``output_folder`` does not exist.
    FileExistsError
        If ``output_folder`` is not empty and ``overwrite_results`` is ``False``.

    Warns
    -----
    UserWarning
        If ``output_folder`` is not empty and ``overwrite_results`` is ``True``.
    """

    folder = Path(output_folder)
    if not folder.resolve().parent.is_dir():
        raise FileNotFoundError(f'The parent directory of the output folder "{folder}" does not exist')

    if folder.is_dir() and any(folder.iterdir()):
        if not overwrite_results:
            raise FileExistsError(f'The output folder "{folder}" is not empty. Use another output_folder or '
                                  f'overwrite_results=True to delete its files')
        warn_message(f'The output folder "{folder}" contains files which will be deleted')

    return


def merge_settings(user_settings, default_settings, toml_path, TIME_SETTINGS=None):

    """
    Model inputs from the user inputs and the [default_settings] of the configuration file.

    The user inputs which are not None overwrite the default values. The [default_settings] keys must be inputs of
    the Compiler. The time settings (TIME_SETTINGS) must be set either in the configuration file or as inputs, not
    both. If log_intervals is not provided anywhere it is False (linear spacing).

    Parameters
    ----------
    user_settings : dict
        All the Compiler inputs, None for those not provided by the user.
    default_settings : dict
        The [default_settings] section of the configuration file.
    toml_path : Path
        Configuration file, for the error messages.

    Returns
    -------
    dict
        One entry per Compiler input.
    """

    TIME_SETTINGS = TIME_SETTINGS if TIME_SETTINGS is not None else ('log_intervals', 'log_age_start', 'log_age_end',
                                                                     'log_age_step', 'n_spectra', 'age_decimals')

    unknown = sorted(set(default_settings) - set(user_settings))
    if unknown:
        raise KeyError(f'Unknown [default_settings] entries in "{toml_path.name}": {unknown}. '
                       f'Available inputs: {list(user_settings)}')

    time_cfg = [k for k in TIME_SETTINGS if default_settings.get(k) is not None]
    time_user = [k for k in TIME_SETTINGS if user_settings[k] is not None]
    if time_cfg and time_user:
        raise ValueError(f'Time settings found in both the [default_settings] of "{toml_path.name}" {time_cfg} and the '
                         f'inputs {time_user}. Set the time settings only in the configuration file or only as inputs')

    settings = {k: v if v is not None else default_settings.get(k) for k, v in user_settings.items()}
    if settings['log_intervals'] is None:
        settings['log_intervals'] = False

    return settings



# Progress bar and messaging tool
class ProgressTracker:

    def __init__(self, verbose=True, progress_callback=None):

        """
        Parameters
        ----------
        verbose : bool, optional
            Print the messages and progress bars in the terminal. Default is True.
        progress_callback : callable, optional
            Function called with the tracker (callback(tracker)) after each message and loop iteration.
        """

        self.verbose = verbose
        self.progress_callback = progress_callback

        self.stage = ''      # Description of the current (or last) loop
        self.counter = 0     # Iterations completed in the current loop
        self.total = 0       # Number of iterations of the current loop
        self.message = ''    # Last message
        self._bar = None     # tqdm bar of the active loop
        self._text_bar = False  # Plain text bar of the active loop (no tqdm)

    @property
    def fraction(self):
        """Completed fraction of the current loop (0 to 1)."""
        return min(self.counter / self.total, 1.0) if self.total else 0.0

    def log(self, message):
        """Terminal message, printed above an active progress bar."""
        self.message = message
        if self.verbose:
            if self._bar is not None:
                tqdm.write(message)
            elif self._text_bar:
                sys.stdout.write(f'\r\033[K{message}\n')
                self._draw_text_bar()
            else:
                print(message)
        self._notify()

    def track(self, iterable, desc, total=None):
        """
        Iterate over iterable updating the counter, the terminal progress bar and the callback after each iteration.
        total is required for iterables without len (e.g. generators).
        """
        self.stage, self.counter = desc, 0
        self.total = len(iterable) if total is None else total
        self._notify()

        if self.verbose:
            if tqdm is not None:
                self._bar = tqdm(total=self.total, desc=desc, unit='step')
            else:
                self._text_bar = True
                self._draw_text_bar()

        try:
            for item in iterable:
                yield item
                self.counter += 1
                if self._bar is not None:
                    self._bar.update(1)
                elif self._text_bar:
                    self._draw_text_bar()
                self._notify()
        finally:
            if self._bar is not None:
                self._bar.close()
                self._bar = None
            if self._text_bar:
                sys.stdout.write('\n')
                sys.stdout.flush()
                self._text_bar = False

    def _draw_text_bar(self, width=30):
        filled = int(width * self.fraction)
        sys.stdout.write(f'\r{self.stage}: |{"#" * filled}{" " * (width - filled)}| {self.counter}/{self.total}')
        sys.stdout.flush()

    def _notify(self):
        if self.progress_callback is not None:
            self.progress_callback(self)


# Read configuration file
DEFAULT_TOML = Path(__file__).with_name('pystarburst99.toml')


def load_cfg(file_address):

    """

    This function reads a configuration file with the `toml format <https://toml.io/en/>`_.

    :param file_address: Input configuration file address.
    :type file_address: str, pathlib.Path

    :return: Parsed configuration data
    :type: dict

    """

    file_path = Path(file_address)

    # Open the file
    if file_path.is_file():

        # Toml file
        with open(file_path, mode="rb") as fp:
            cfg = tomllib.load(fp)

    else:
        raise pySB99_Error(f'The configuration file was not found at: {file_address}')

    return cfg


def load_metallicity_data(Z, track_choice, rot, SED_library, spectra_library, IMF_mass_limits,
                          mass_loss_rate_option='DEFAULT', toml_path=None):

    """
    Load the evolutionary tracks and spectral libraries for a metallicity.

    Parameters
    ----------
    Z : str
        Metallicity: MWC, MW, LMC, SMC, IZw18, XMP or Z0.
    track_choice : str
        GENEC or Pauli.
    rot : bool
        Rotating (True) or non-rotating (False) GENEC tracks. Ignored for Pauli tracks.
    SED_library : str
        Low-resolution OB library: FW, WM or PoWR.
    spectra_library : str
        High-resolution OB library: WM or PoWR.
    IMF_mass_limits : sequence
        IMF mass boundaries. The last one is checked against the track upper mass.
    mass_loss_rate_option : str
        PoWR high-resolution grid mass-loss option (only DEFAULT available).
    toml_path : str or Path
        Configuration file with the file names.

    Returns
    -------
    dict
        Loaded variables, with the same names as in the original pySB99.py. Entries which do not apply to the
        chosen spectra_library are None (hires_flux, hires_cont_flux, hires_params for PoWR and hires_teffs,
        hires_loggs, hires_logls, hires_spec_reformed, hires_radii, hires_spec_norm for WM).
    """

    # Use the default configuration unless the user provide one
    cfg = load_cfg(Path(DEFAULT_TOML)) if toml_path is None else load_cfg(Path(toml_path))

    metallicities = [key for key in cfg if key != 'paths']
    if Z not in metallicities:
        raise ValueError(f'Metallicity Z={Z!r} not recognised. Options are: {metallicities}')
    z_cfg = cfg[Z]

    data_folder = Path(cfg['paths']['data_folder'])
    if not data_folder.is_absolute():
        data_folder = toml_path.parent / data_folder
    file_path = data_folder / z_cfg['folder']

    # Evolutionary tracks
    track_key = 'Pauli' if track_choice == 'Pauli' else ('GENEC_v40' if rot else 'GENEC_v00')
    if track_key not in z_cfg['tracks']:
        raise ValueError(f'There are no {track_key} tracks at Z={z_cfg["Z_value"]}')
    tracks = z_cfg['tracks'][track_key]
    if IMF_mass_limits[-1] > tracks['max_mass']:
        raise ValueError(f'{track_key} tracks do not exist at Z={z_cfg["Z_value"]} above {tracks["max_mass"]} Msol')

    evo_tracks = np.load(file_path / tracks['file'])

    # Low-resolution OB spectra (FastWind very massive star grid if the IMF goes above 120 Msol)
    SED = z_cfg['SED']
    if SED_library not in SED:
        raise ValueError(f'SED_library={SED_library!r} not recognised. Options are: {[k for k in SED if k != "FW_VMS"]}')
    SED_key = 'FW_VMS' if (SED_library == 'FW' and IMF_mass_limits[-1] > 120.0 and 'FW_VMS' in SED) else SED_library
    spectra_grid_file = str(file_path / SED[SED_key])

    # High-resolution spectra
    hires_WM = z_cfg['hires_WM']
    hires_flux = hires_cont_flux = hires_params = None
    hires_teffs = hires_loggs = hires_logls = hires_spec_reformed = hires_radii = hires_spec_norm = None

    if spectra_library == 'WM':
        hires_wave_grid = np.load(file_path / hires_WM['wave_grid'])
        hires_flux = np.load(file_path / hires_WM['line_flux'])
        hires_cont_flux = np.load(file_path / hires_WM['cont_flux'])
        hires_params = np.load(file_path / hires_WM['params'])

    elif spectra_library == 'PoWR':
        hires_wave_grid = np.linspace(920, 3100, 10000)

        if mass_loss_rate_option not in z_cfg['hires_PoWR']:
            raise ValueError(f'mass_loss_rate_option={mass_loss_rate_option!r} not available. '
                             f'Options are: {list(z_cfg["hires_PoWR"])}')
        hires_PoWR = z_cfg['hires_PoWR'][mass_loss_rate_option]
        hires_teffs = np.genfromtxt(file_path / hires_PoWR['teffs'])
        hires_loggs = np.genfromtxt(file_path / hires_PoWR['loggs'])
        hires_logls = np.genfromtxt(file_path / hires_PoWR['logls'])
        hires_spec_reformed = np.genfromtxt(file_path / hires_PoWR['calib'])  # each spectrum is a row
        hires_radii = (((3.828*10**(26)*10**hires_logls)/(4*3.14*5.67*10**-8*hires_teffs**4))**0.5)/(7*10**8)
        hires_spec_norm = np.genfromtxt(file_path / hires_PoWR['norm'])  # each spectrum is a row

    else:
        raise ValueError(f'spectra_library={spectra_library!r} not recognised. Options are: WM, PoWR')

    empty_hires_flux = np.full_like(hires_wave_grid, 0.0)

    # The low-mass high-resolution spectra always come from the WMbasic (IFA) files
    lowmass_hires_wave_grid = np.load(file_path / hires_WM['wave_grid'])
    lowmass_hires_flux = np.load(file_path / hires_WM['line_flux'])
    lowmass_hires_cont_flux = np.load(file_path / hires_WM['cont_flux'])
    lowmass_hires_params = np.load(file_path / hires_WM['params'])

    # Low-mass stars and Wolf-Rayet libraries
    lowmass, WR = z_cfg['lowmass'], z_cfg['WR']

    return dict(file_path=file_path,
                mass_grid=tracks['mass_grid'],
                evo_tracks=evo_tracks,
                minimum_wr_mass=tracks['minimum_wr_mass'],
                spectra_grid_file=spectra_grid_file,
                hires_wave_grid=hires_wave_grid,
                empty_hires_flux=empty_hires_flux,
                hires_flux=hires_flux,
                hires_cont_flux=hires_cont_flux,
                hires_params=hires_params,
                lowmass_hires_wave_grid=lowmass_hires_wave_grid,
                lowmass_hires_flux=lowmass_hires_flux,
                lowmass_hires_cont_flux=lowmass_hires_cont_flux,
                lowmass_hires_params=lowmass_hires_params,
                hires_teffs=hires_teffs,
                hires_loggs=hires_loggs,
                hires_logls=hires_logls,
                hires_spec_reformed=hires_spec_reformed,
                hires_radii=hires_radii,
                hires_spec_norm=hires_spec_norm,
                lowmass_params=np.load(file_path / lowmass['params']),
                lowmass_flux=np.load(file_path / lowmass['flux']),
                WN_spec_params=np.load(file_path / WR['WN_params_cmfgen']),
                WN_spectra=np.load(file_path / WR['WN_spectra_cmfgen'], allow_pickle=True),
                WC_spec_params=np.load(file_path / WR['WC_params_cmfgen']),
                WC_spectra=np.load(file_path / WR['WC_spectra_cmfgen'], allow_pickle=True),
                WN_spec_params_powr=np.load(file_path / WR['WN_params_powr']),
                WN_spectra_powr=np.load(file_path / WR['WN_spectra_powr'], allow_pickle=True),
                WC_spec_params_powr=np.load(file_path / WR['WC_params_powr']),
                WC_spectra_powr=np.load(file_path / WR['WC_spectra_powr'], allow_pickle=True))

def prepare_output_folder(output_folder, overwrite_results=False):
    """
    Delete the previous results and create an empty output folder.

    The check at the compiler initialization (``check_output_folder``) can be outdated, since the folder may have been
    filled after the Compiler was created, so the folder is checked again before deleting anything.

    Parameters
    ----------
    output_folder : str or pathlib.Path
        Folder where the results will be saved. Its parent directory must exist.
    overwrite_results : bool, optional
        If ``False``, an error is raised when the folder already contains files. If ``True``, the folder and its
        content are deleted. Default is ``False``.

    Returns
    -------
    pathlib.Path
        The new empty output folder.

    Raises
    ------
    FileExistsError
        If ``output_folder`` is not empty and ``overwrite_results`` is ``False``.
    """

    folder = Path(output_folder)
    if folder.is_dir():
        if not overwrite_results and any(folder.iterdir()):
            raise FileExistsError(f'The output folder "{folder}" is not empty. Use another output_folder or '
                                  f'overwrite_results=True to delete its files')
        shutil.rmtree(folder)
    folder.mkdir()

    return folder


