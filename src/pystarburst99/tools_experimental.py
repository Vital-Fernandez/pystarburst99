import numpy as np
from scipy.interpolate import interp1d
from functools import lru_cache

try:
    from pystarburst99.tools import planck, integrate_spec_grid
except ImportError:
    from pystarburst99.tools import planck, integrate_spec_grid

# Columns of the evolutionary track files
TRACK_COLUMNS = {'ages': 1, 'masses': 2, 'lums': 3, 'temps': 4, 'H_abundances': 5, 'He_abundances': 6,
                 '12C_abundances': 7, '14N_abundances': 8, '16O_abundances': 9, 'core_temps': 10, 'mass_loss_rates': 11}

# Wavelengths (Angstrom) for the Kurucz continuum in assign_spectra_to_grid_hires
CONT_WAVE_GRID = [899.9, 1085., 1150., 1280., 1310., 1360., 1430., 1480., 1510., 1580., 1630., 1680., 1740., 1785., 1820.,
                  1875., 2035., 2235., 2565., 2745., 2895., 2999.]

# Container for the experimental tools
BETA_TOOLS = ('get_timestep_0_ind', 'get_timestep_params', 'assign_spectra_to_grid_WR', 'assign_spectra_to_grid_hires',
              'specsyn', 'specsyn_hires')


def prepare_tracks(evo_tracks, n_tracks):
    """
    Evolutionary tracks as a (n_tracks, n_points + 2, n_columns) array.

    As in track_grids: the first age is set to 1e-3 yr, two points are added after the end of the evolution
    (luminosity -20 flags the supernova) and the mass-loss rates above -1 are set to -1000.
    """
    evo_tracks = np.asarray(evo_tracks, dtype=float)
    if evo_tracks.shape[0] % n_tracks:
        raise ValueError(f'The {evo_tracks.shape[0]} track points cannot be split in {n_tracks} equal tracks')

    tracks = evo_tracks.reshape(n_tracks, -1, evo_tracks.shape[1]).copy()
    tracks[:, 0, 1] = 1.0e-3

    end = np.repeat(tracks[:, -1:, :], 2, axis=1)
    end_age = tracks[:, -1, 1] + 100.
    end[:, 0, 1], end[:, 1, 1] = end_age, end_age * 10**6
    end[:, :, 3] = -20.
    tracks = np.concatenate((tracks, end), axis=1)

    mdot = tracks[:, :, 11]
    mdot[mdot > -1] = -1000

    return tracks


def _inter_track_sampling(initial_mass_upper, initial_mass_lower, run_speed_mode, previous):
    """Number of interpolated tracks between two tracks, with the rules of interpolate_param."""
    sampling = previous
    dm = initial_mass_upper - initial_mass_lower + 1
    m = round(initial_mass_upper)

    if run_speed_mode == 'DEFAULT':
        sampling = dm * 15 if 7 < m < 35 else dm
    elif run_speed_mode == 'FAST':
        sampling = dm * 1
    elif run_speed_mode == 'HIGH_RES':
        # NOTE: there is no rule for round(M_upper) = 9, 15, 35 or 60 Msol, so interpolate_param reuses the value of the
        # previous pair of tracks
        if 9 < m < 15:
            sampling = dm * 15
        elif 15 < m < 35:
            sampling = dm * 5
        elif m < 9:
            sampling = dm * 10
        elif 35 < m < 60:
            sampling = dm * 5
        elif m > 60:
            sampling = dm

    if sampling is None:
        raise ValueError(f'No {run_speed_mode} sampling rule for the tracks with initial mass {initial_mass_upper} Msol')

    return sampling


def _linspace_rows(start, stop, num):
    """np.linspace(start[j], stop[j], num) for every j (one row each), with the arithmetic of the scalar call."""
    k = np.arange(num, dtype=float)
    delta = (stop - start)[:, None]
    if num > 1:
        step = delta / (num - 1)
        y = np.where(step == 0, k / (num - 1) * delta, k * step)
    else:
        y = k * delta
    y = y + start[:, None]
    if num > 1:
        y[:, -1] = stop

    return y


def _lerp(x0, x1, y0, y1, x_new, form):
    """
    interp1d([x0, x1], [y0, y1], kind='linear', fill_value='extrapolate')(x_new) for every row: x0, x1, y0, y1 have
    shape (n,) and x_new (n, m). The two points are sorted as in interp1d.
    """
    swap = x0 > x1
    x_lo, x_hi = np.where(swap, x1, x0)[:, None], np.where(swap, x0, x1)[:, None]
    y_lo, y_hi = np.where(swap, y1, y0)[:, None], np.where(swap, y0, y1)[:, None]

    with np.errstate(divide='ignore', invalid='ignore'):
        if form == 'weights':
            return ((x_new - x_lo) / (x_hi - x_lo)) * y_hi + ((x_hi - x_new) / (x_hi - x_lo)) * y_lo
        return (y_hi - y_lo) / (x_hi - x_lo) * (x_new - x_lo) + y_lo


@lru_cache(maxsize=None)
def _interp1d_form():
    """
    interp1d(kind='linear') has used two different formulas across SciPy releases. Return the one of the installed
    version ('weights' or 'slope') so that the vectorised interpolation gives the same bits.
    """
    rng = np.random.default_rng(0)
    x, y, x_new = rng.uniform(0.5, 120., (200, 2)), rng.uniform(-20., 6., (200, 2)), rng.uniform(0.5, 120., (200, 1))
    ref = np.array([interp1d(x[i], y[i], kind='linear', fill_value='extrapolate')(x_new[i]) for i in range(len(x))])
    for form in ('weights', 'slope'):
        if np.array_equal(ref, _lerp(x[:, 0], x[:, 1], y[:, 0], y[:, 1], x_new, form)):
            return form

    raise RuntimeError('The interp1d linear formula of this SciPy version is not recognised')


def interpolate_tracks_exp(evo_tracks, n_tracks, run_speed_mode):
    """
    Isochrone grids: the evolutionary tracks interpolated in mass between adjacent initial masses.

    Same values as interpolate_param + rearrange_grid_array for each parameter, but each grid is a
    (n_grid_tracks, n_points) array (one row per interpolated track, same row order) and the mass grid is computed once
    for all the parameters.

    Returns
    -------
    dict
        grid_ages, grid_masses, grid_lums, grid_temps, grid_H_abundances, grid_He_abundances, grid_12C_abundances,
        grid_14N_abundances, grid_16O_abundances, grid_core_temps and grid_mass_loss_rates.
    """
    tracks = prepare_tracks(evo_tracks, n_tracks)[:, ::-1, :]  # np.flip of each track, as in interpolate_param
    form = _interp1d_form()
    rows = {name: [] for name in TRACK_COLUMNS}

    sampling = None
    for i in range(n_tracks - 1):
        upper, lower = tracks[i], tracks[i + 1]
        sampling = _inter_track_sampling(upper[-1, 2], lower[-1, 2], run_speed_mode, sampling)
        grid_mass = _linspace_rows(lower[:, 2], upper[:, 2], round(sampling))  # (n_points, n_samples)

        # Order of rearrange_grid_array: one row per interpolated track, from the upper to the lower mass
        for name, col in TRACK_COLUMNS.items():
            values = grid_mass if name == 'masses' else _lerp(lower[:, 2], upper[:, 2], lower[:, col], upper[:, col],
                                                               grid_mass, form)
            rows[name].append(values[:, ::-1].T)

    return {f'grid_{name}': np.concatenate(r) for name, r in rows.items()}


def get_timestep_0_ind(timestep, grid_ages, grid_masses, times_steps, IMF_mass_limits):
    """Same as tools.get_timestep_0_ind for (n_tracks, n_points) grids."""
    ind_nearest_age = np.argmin((grid_ages - times_steps[0])**2, axis=1)
    masses = grid_masses[np.arange(len(grid_masses)), ind_nearest_age]
    unique_masses, first_index = np.unique(masses, return_index=True)

    return first_index[unique_masses < IMF_mass_limits[-1]]


def get_timestep_params(timestep, timestep_mass_ind, track_choice, grids):
    """
    Same as tools.get_timestep_params for (n_tracks, n_points) grids (from interpolate_tracks). All the outputs are
    numpy arrays.
    """
    grid_ages = grids['grid_ages']
    ind_age = np.argmin(abs(grid_ages - timestep), axis=1)  # nearest point of every interpolated track
    rows = np.asarray(timestep_mass_ind)
    cols = ind_age[rows]

    def pick(name):
        return grids[name][rows, cols]

    ages, masses, temps, lums = pick('grid_ages'), pick('grid_masses'), pick('grid_temps'), pick('grid_lums')
    H, He, C12, N14, O16 = (pick(f'grid_{n}_abundances') for n in ('H', 'He', '12C', '14N', '16O'))
    core_temps, mass_loss_rates = pick('grid_core_temps'), pick('grid_mass_loss_rates')
    timestep_masses = grids['grid_masses'][np.arange(len(grid_ages)), ind_age]  # all the interpolated tracks

    teffs_calc = 10**temps
    lum_calc = (10**lums * 3.839e33)
    radii_denom = (12.566 * 5.670e-5 * (teffs_calc**4))
    radii_calc = ((lum_calc / radii_denom)**0.5) * 100
    logg_num = (6.67e-12) * masses * (1.9891e33) * 1e8
    loggs = np.log10((logg_num) / (radii_calc**2))

    cnr = C12 / N14
    coher = ((C12/12.)+(O16/16.))/(He/4.)

    # Wolf-Rayet temperature correction (a short loop over the hydrogen-poor stars keeps the scalar powers of tools.py)
    if track_choice == 'GENEC':
        WR_correction_factor = 0.6
        for i in np.flatnonzero(H < 0.1):
            core_teff = 10**core_temps[i]
            temps[i] = np.log10(core_teff + (WR_correction_factor-1.0)*(core_teff - teffs_calc[i]))

    return (ages, temps, lums, masses, H, loggs, mass_loss_rates, C12, N14, O16, timestep_masses, cnr, coher,
            teffs_calc)


def assign_spectra_to_grid_WR(timestep_temps_final, spec_params_reform, spec_params_teff, spec_params_logl,
                              timestep_lums_final, timestep_H_abundances_final, spec_params_logg, timestep_masses_final,
                              initial_masses, specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher, No_stars,
                              timestep_loggs_final, spec_lib):
    """Same as tools.assign_spectra_to_grid_WR, with the nearest spectrum search done by numpy."""
    minimum_wr_mass = spec_lib['minimum_wr_mass']
    WN = [spec_lib[f'WN_{k}'] for k in ('spec_params_teff', 'spec_params_logl', 'reformed_spec_grid', 'integrated_spectra')]
    WC = [spec_lib[f'WC_{k}'] for k in ('spec_params_teff', 'spec_params_logl', 'reformed_spec_grid', 'integrated_spectra')]
    reformed_spec_grid, integrated_spectra = spec_lib['reformed_spec_grid'], spec_lib['integrated_spectra']
    lowmass_teffs, lowmass_loggs = spec_lib['lowmass_teffs'], spec_lib['lowmass_loggs']
    lowmass_spec, lowmass_int_spec = spec_lib['lowmass_spec'], spec_lib['lowmass_int_spec']
    empty_spectrum = np.column_stack((spec_lib['wave_grid'], spec_lib['empty_flux']))  # shared by the 'nope' stars

    out = dict(integrated=[], teff=[], spectra=[], choice=[], logl=[], logg=[])

    def add(choice, teff, spectrum, integrated, logl=None, logg=None):
        out['choice'].append(choice), out['teff'].append(teff), out['spectra'].append(spectrum)
        out['integrated'].append(integrated)
        if logl is not None:
            out['logl'].append(logl), out['logg'].append(logg)

    timestep_teffs_final = 10**timestep_temps_final
    for j in range(len(timestep_temps_final)):
        T = timestep_temps_final[j]
        teff = 10**T  # scalar power, as in tools.py

        if timestep_lums_final[j] == -20.0:
            add('nope', 0.0, empty_spectrum, 1e-30)

        elif T > 4.4 and timestep_H_abundances_final[j] < 0.4 and initial_masses[j] > minimum_wr_mass:
            if timestep_H_abundances_final[j] > 0.1 or timestep_cnr[j] < 10.:
                lib_teff, lib_logl, lib_spectra, lib_int = WN
            elif timestep_coher[j] < 1. or timestep_coher[j] >= 1.:
                lib_teff, lib_logl, lib_spectra, lib_int = WC
            else:  # NaN C/O/He ratio: tools.py appends nothing but the choice
                out['choice'].append('WR')
                continue
            k = np.argmin((lib_teff - teff)**2)
            add('WR', lib_teff[k], lib_spectra[k], lib_int[k], lib_logl[k], -1.)

        elif T < 3.65 and initial_masses[j] > minimum_wr_mass:
            # NOTE: tools.py appends only to the choice list, so these stars get no spectrum and the following stars are
            # shifted with respect to the radii and No_stars in specsyn
            out['choice'].append(No_stars[j])

        elif timestep_teffs_final[j] > 25000.:
            k = np.argmin(abs(spec_params_teff - teff) + abs(spec_params_logg - timestep_loggs_final[j]))
            add('OB', spec_params_teff[k], reformed_spec_grid[k], integrated_spectra[k], spec_params_logl[k],
                spec_params_logg[k])

        elif timestep_teffs_final[j] < 2000.:
            add('nope', 0.0, empty_spectrum, 1e-30)

        else:
            k = np.argmin((lowmass_teffs - teff)**2 + (lowmass_loggs - timestep_loggs_final[j])**2)
            add('lowmass', lowmass_teffs[k], lowmass_spec[k], lowmass_int_spec[k], timestep_lums_final[j],
                lowmass_loggs[k])

    return out['integrated'], out['teff'], out['spectra'], out['choice'], out['logl'], out['logg']


class HiresLibrary:

    def __init__(self, spec_lib):
        self.wave = spec_lib['hires_wave_grid']
        self.empty = np.column_stack((self.wave, np.full_like(self.wave, 0.0)))

        WN, WC = spec_lib['WN_spec_params_reform_powr'], spec_lib['WC_spec_params_reform_powr']
        self.WN_teff, self.WN_radius = WN[:, 1], WN[:, 2]
        self.WC_teff, self.WC_radius = WC[:, 1], WC[:, 2]
        self.WN_spectra, self.WC_spectra = spec_lib['WN_reformed_spec_grid_powr'], spec_lib['WC_reformed_spec_grid_powr']
        self.WN_int = integrate_spec_grid(self.WN_spectra)

        self.lowmass_teffs = spec_lib['lowmass_params'][:, 0]
        self.lowmass_spec = [np.column_stack((spec_lib['lowmass_wave_grid'], f)) for f in spec_lib['lowmass_flux']]
        self.lowmass_int_spec = integrate_spec_grid(self.lowmass_spec)

        self._flux, self._cont = {}, {}

    def lowmass_flux(self, k):
        """Low-mass library spectrum k on the high-resolution grid."""
        if k not in self._flux:
            spectrum = self.lowmass_spec[k]
            self._flux[k] = np.column_stack((self.wave, np.interp(self.wave, spectrum[:, 0], spectrum[:, 1])))
        return self._flux[k]

    def kurucz_cont(self, k):
        """Continuum of the low-mass library spectrum k on the high-resolution grid."""
        if k not in self._cont:
            spectrum = self.lowmass_spec[k]
            cont_flux_lowres_sample = np.interp(CONT_WAVE_GRID, spectrum[:, 0], spectrum[:, 1])
            cont_flux_average = []
            for i in range(len(cont_flux_lowres_sample)):
                ind_flux_norm = np.abs(cont_flux_lowres_sample[i] - spectrum[:, 1]).argmin()
                flux_norm_lower = spectrum[:, 1][ind_flux_norm - 1]
                flux_norm_higher = spectrum[:, 1][ind_flux_norm + 1]
                cont_flux_average.append(np.log10((1.0e-30 + cont_flux_lowres_sample[i] + flux_norm_lower +
                                                   flux_norm_higher) / 3))
            cont_flux_resampled = np.interp(self.wave, CONT_WAVE_GRID, cont_flux_average)
            self._cont[k] = np.column_stack((self.wave, 10**cont_flux_resampled))
        return self._cont[k]


def assign_spectra_to_grid_hires(timestep_temps_final, hires_teffs, hires_loggs, timestep_lums_final,
                                 timestep_H_abundances_final, timestep_masses_final, initial_masses, specsyn_cotests,
                                 specsyn_loggs, timestep_cnr, timestep_coher, timestep_radii_final, spec_lib):

    lib = spec_lib.get('hires_library') or HiresLibrary(spec_lib)
    minimum_wr_mass, spectra_library = spec_lib['minimum_wr_mass'], spec_lib['spectra_library']
    hires_spec, hires_cont, hires_int_spec = spec_lib['hires_spec'], spec_lib['hires_cont'], spec_lib['hires_int_spec']
    hires_wave_grid = lib.wave
    hires_temps = np.log10(hires_teffs)

    out = dict(integrated=[], teff=[], logg=[], spectra=[], cont=[], choice=[])

    def add(choice, teff, spectrum, cont, integrated, logg=None):
        out['choice'].append(choice), out['teff'].append(teff), out['spectra'].append(spectrum)
        out['cont'].append(cont), out['integrated'].append(integrated)
        if logg is not None:
            out['logg'].append(logg)

    timestep_teffs_final = 10**timestep_temps_final
    for j in range(len(timestep_temps_final)):
        T, teff, M_ini = timestep_temps_final[j], timestep_teffs_final[j], initial_masses[j]

        if timestep_lums_final[j] == -20.0:
            add('nope', 0.0, lib.empty, lib.empty, 1e-30)

        elif T > 4.4 and timestep_H_abundances_final[j] < 0.40 and M_ini >= minimum_wr_mass:
            if timestep_H_abundances_final[j] > 0.098:
                choice, WN = 'WNL', True
            elif timestep_cnr[j] < 10.:
                choice, WN = 'WNE', True
            elif timestep_coher[j] < 0.5:
                choice, WN = 'WCL', False
            elif timestep_coher[j] < 1.:
                choice, WN = 'WCE', False
            elif timestep_coher[j] >= 1.:
                choice, WN = 'WO', False
            else:  # NaN C/O/He ratio: tools.py appends nothing
                continue

            lib_teff, lib_radius, lib_spectra = ((lib.WN_teff, lib.WN_radius, lib.WN_spectra) if WN else
                                                 (lib.WC_teff, lib.WC_radius, lib.WC_spectra))
            k = np.argmin(abs(teff - lib_teff))
            near_spec_temp, assigned_spectrum = lib_teff[k], lib_spectra[k]
            flux = assigned_spectrum[:, 1] * (((3.0856e9)**2.0)/((lib_radius[k]*6.96e10)**2.0)) * 1.0e20
            flux = flux * teff / near_spec_temp
            flux = np.interp(hires_wave_grid, assigned_spectrum[:, 0], flux)

            cont = lib.empty
            if choice == 'WNL':
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont = np.column_stack((hires_wave_grid, spec_cont[:, 1] * flux[4050] / spec_cont[:, 1][4050]))

            # NOTE: tools.py uses the WN integrals also for the WC stars (the hires integrals are not used downstream)
            add(choice, near_spec_temp, np.column_stack((hires_wave_grid, flux)), cont, lib.WN_int[k], 1.0)

        elif teff < 2000:
            k = np.argmin((lib.lowmass_teffs - 10**T)**2)
            spectrum = lib.lowmass_flux(k)
            add('lowmass', lib.lowmass_teffs[k], spectrum, spectrum, lib.lowmass_int_spec[k], 1.0)

        elif M_ini > 10. and teff < 22000:
            k = np.argmin((lib.lowmass_teffs - 10**T)**2)
            add('kurucz', lib.lowmass_teffs[k], lib.lowmass_flux(k), lib.kurucz_cont(k), lib.lowmass_int_spec[k], 1.0)

        elif M_ini < 5. or M_ini > 5. and teff < 17000:  # NOTE: "or" before "and" excludes M_ini = 5 exactly
            k = np.argmin((lib.lowmass_teffs - 10**T)**2)
            add('kurucz2', lib.lowmass_teffs[k], lib.lowmass_flux(k), lib.kurucz_cont(k), lib.lowmass_int_spec[k], 1.0)

        else:
            k = np.argmin((T - hires_temps)**2 + 5*(hires_loggs - specsyn_loggs[j])**2)
            if spectra_library == 'WM':
                add('OB - WM', hires_teffs[k], hires_spec[k], hires_cont[k], hires_int_spec[k], hires_loggs[k])
            elif spectra_library == 'PoWR':
                flux = hires_spec[k][:, 1] * teff / hires_teffs[k]
                add('OB - PoWR', hires_teffs[k], np.column_stack((hires_wave_grid, flux)), hires_cont[k],
                    hires_int_spec[k], hires_loggs[k])
            elif spectra_library == 'FW':
                flux = np.column_stack((hires_wave_grid, np.interp(hires_wave_grid, hires_spec[k][:, 0], hires_spec[k][:, 1])))
                add('OB - FW', hires_teffs[k], flux, flux, hires_int_spec[k], hires_loggs[k])

    return out['integrated'], out['teff'], out['logg'], out['spectra'], out['cont'], out['choice']


def _running_sum(arrays):
    """Sum of a sequence of arrays added in order: the same result as np.array(arrays).sum(axis=0), without the stack."""
    arrays = iter(arrays)
    total = np.array(next(arrays), dtype=float)
    for a in arrays:
        total += a

    return total


def specsyn(assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, radii, No_stars):
    """Same as tools.specsyn."""
    # NOTE: the loop runs over the assigned spectra, which can be fewer than the stars (see assign_spectra_to_grid_WR)
    terms = (12.566 * radii[i] * radii[i] / 1e20 * (assigned_spectra[i][:, 1] /
                                                    (assigned_integrated_spectra[i] / specsyn_bbfluxes[i])) * No_stars[i]
             for i in range(len(assigned_integrated_spectra)))

    return _running_sum(terms), []


def specsyn_hires(assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, assigned_cont, radii, No_stars,
                  population_nebular, assigned_hires_choice, hires_wave_grid, spectra_library):
    """
    Same as tools.specsyn_hires. The black-body renormalisation, the PoWR continuum and the other quantities which are
    computed but not returned by tools.py are skipped.
    """
    population_nebular_resampled = np.interp(hires_wave_grid, population_nebular[:, 0], population_nebular[:, 1])

    # NOTE: as in tools.py, the number of stars comes from the low-resolution assignment
    n_stars = range(len(assigned_integrated_spectra))
    coeffs = [12.566 * radii[i]**2 / 1e20 for i in n_stars]
    population_flux = _running_sum(coeffs[i] * assigned_spectra[i][:, 1] * No_stars[i] for i in n_stars)

    if spectra_library == 'PoWR':
        population_flux_norm = np.array([assigned_cont[i][:, 1] for i in n_stars
                                         if assigned_hires_choice[i] == 'OB - PoWR']).mean(axis=0)
    else:
        population_cont = _running_sum(coeffs[i] * assigned_cont[i][:, 1] * No_stars[i] for i in n_stars)
        population_flux_norm = population_flux / (population_cont+1.0e-35)

    population_flux_wneb = np.add(population_flux, population_nebular_resampled)

    return population_flux, population_flux_wneb, population_flux_norm, population_nebular_resampled