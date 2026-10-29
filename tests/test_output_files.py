import os
import re
from pathlib import Path

import difflib
import pytest
import numpy as np


def current_dir(metallicity):
    return CURRENT_ROOT / f'{metallicity}_grids'


def baseline_dir(metallicity):
    return BASELINE_ROOT / f'pysb_{metallicity}'


def file_checker(folder, *suffixes):
    """
    Sorted file names in a folder, optionally filtered by suffix.
    """
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and (not suffixes or p.suffix in suffixes))


def file_loader(path):
    """
    Load a pySB99 output array (.npy or np.savetxt .txt).
    """
    if path.suffix == '.npy':
        return np.load(path)
    return np.loadtxt(path, ndmin=1)


def comment_hdr(path):
    """
    Comment lines written by np.savetxt(header=...), e.g. in colours.txt
    """
    with open(path) as f:
        return [line.rstrip() for line in f if line.startswith('#')]


def comparison_indexing(name, current, baseline):
    """
    Row indices (current, baseline) of the ages in both runs for the outputs with one row per age, or None for the
    outputs without a time axis.
    """
    time_file = next((t for t, group in TIME_FILES.items() if name in group), None)
    if time_file is None:
        return None

    t_curr, t_base = file_loader(current / time_file), file_loader(baseline / time_file)
    match = np.isclose(t_curr[:, None], t_base[None, :], rtol=AGE_RTOL, atol=0)
    idx_curr, idx_base = np.nonzero(match)
    return idx_curr, idx_base


def file_counter(select):
    return [pytest.param(met, name, id=f'{met}-{name}') for met in METALLICITIES for name in select(file_checker(baseline_dir(met), '.txt', '.npy', '.png'))]


# Files location  # 210 passed 105 ignored
METALLICITIES = ['MWC', 'MW', 'LMC', 'SMC', 'IZw18', 'XMP', 'Z0']
CURRENT_ROOT = Path('/home/vital/Astrodata/SpecSy_tests/pyStarburst99_tests/version1')
BASELINE_ROOT = Path('/home/vital/PycharmProjects/pystarburst99/tests/baseline')

# Comparison parameters
RTOL = float(os.environ.get('PYSB_RTOL', 1e-10))
ATOL = float(os.environ.get('PYSB_ATOL', 0.0))
AGE_RTOL = 1e-9
CHECK_PNG = os.environ.get('PYSB_CHECK_PNG', '0') == '1'

# Comparison file lists
TEXT_FILES = {'input.txt'}  # compared line by line instead of numerically

STEP_FILES = {'timesteps.txt', 'bololum.txt', 'ion_flux_HI.txt', 'ion_flux_HeI.txt', 'ion_flux_HeII.txt',
              'windpower.txt', 'windmom.txt', 'uv_slope.txt', 'Haew.txt', 'Hbew.txt', 'Pbew.txt', 'Bgew.txt',
              'colours.txt', 'SNrate.txt'}

SPECTRA_FILES = {'times_spectra.txt', 'SED_times.txt', 'pySB_SED_stellar.npy', 'pySB_SED_stellar_and_nebular.npy',
                 'pySB_hires_spectrum.npy', 'pySB_hires_norm_spectrum.npy', 'pySB_hires_spectrum_without_neb.npy'}

TIME_FILES = {'timesteps.txt': STEP_FILES, 'times_spectra.txt': SPECTRA_FILES}


NUMERIC_CASES = file_counter(lambda files: [f for f in files if not f.endswith('.png') and f not in TEXT_FILES])
TEXT_CASES = file_counter(lambda files: [f for f in files if f in TEXT_FILES])
PLOT_CASES = file_counter(lambda files: [f for f in files if f.endswith('.png')])


@pytest.mark.parametrize('metallicity', METALLICITIES)
def test_folders_exist(metallicity):
    baseline, current = baseline_dir(metallicity), current_dir(metallicity)
    assert baseline.is_dir(), f'Baseline folder not found: {baseline.resolve()}'
    assert current.is_dir(), f'Current output folder not found: {current.resolve()}'
    assert any(f not in TEXT_FILES for f in file_checker(baseline, '.txt', '.npy')), f'No .txt/.npy outputs found in {baseline.resolve()}'



# Summary file added after the baseline was generated: pystarburst99v{version}_{Z}.txt
NEW_FILES = re.compile(r'^pystarburst99v.+_.+\.txt$')

@pytest.mark.parametrize('metallicity', METALLICITIES)
def test_same_file_list(metallicity):
    """Every baseline output is still produced, and no unexpected new ones appear."""
    baseline, current = baseline_dir(metallicity), current_dir(metallicity)
    suffixes = () if CHECK_PNG else ('.txt', '.npy')
    base, curr = set(file_checker(baseline, *suffixes)), set(file_checker(current, *suffixes))
    curr = {f for f in curr if not NEW_FILES.match(Path(f).name)}
    missing, extra = sorted(base - curr), sorted(curr - base)
    assert not missing, f'Outputs missing from {current}: {missing}'
    assert not extra, f'Outputs in {current} not present in the baseline: {extra}'


@pytest.mark.parametrize('metallicity, name', TEXT_CASES)
def test_input_parameters(metallicity, name):
    """The model was run with the same input parameters as the baseline."""
    base = (baseline_dir(metallicity) / name).read_text().splitlines()
    curr = (current_dir(metallicity) / name).read_text().splitlines()
    diff = '\n'.join(difflib.unified_diff(base, curr, 'baseline', 'current', lineterm=''))
    assert curr == base, f'{metallicity}/{name} differs:\n{diff}'


@pytest.mark.parametrize('metallicity, name', NUMERIC_CASES)
def test_numeric_output(metallicity, name):
    """
    Arrays match the baseline in shape and values within RTOL/ATOL. For the outputs with one row per age only the ages
    in both runs are compared (the runs can have different time grids).
    """
    baseline, current = baseline_dir(metallicity), current_dir(metallicity)
    curr_path = current / name
    assert curr_path.exists(), f'{name} not found in {current}'

    base, curr = file_loader(baseline / name), file_loader(curr_path)

    rows = comparison_indexing(name, current, baseline)
    if rows is not None:
        idx_curr, idx_base = rows
        if idx_curr.size == 0:
            pytest.skip(f'{metallicity}/{name}: no common ages between the current and baseline runs')
        assert curr.shape[1:] == base.shape[1:], \
            f'{metallicity}/{name}: shape {curr.shape} != baseline {base.shape} (per age)'
        curr, base = curr[idx_curr], base[idx_base]

    assert curr.shape == base.shape, f'{metallicity}/{name}: shape {curr.shape} != baseline {base.shape}'
    np.testing.assert_allclose(curr, base, rtol=RTOL, atol=ATOL, equal_nan=True,
                               err_msg=f'{metallicity}/{name} differs from baseline')

    if curr_path.suffix == '.txt':
        assert comment_hdr(curr_path) == comment_hdr(baseline / name), f'{metallicity}/{name}: header changed'


@pytest.mark.skipif(not CHECK_PNG, reason='set PYSB_CHECK_PNG=1 to compare plots')
@pytest.mark.parametrize('metallicity, name', PLOT_CASES)
def test_plot(metallicity, name):
    """Optional pixel comparison (sensitive to matplotlib version and fonts)."""
    import matplotlib.image as mpimg
    base, curr = mpimg.imread(baseline_dir(metallicity) / name), mpimg.imread(current_dir(metallicity) / name)
    assert curr.shape == base.shape, f'{metallicity}/{name}: image size {curr.shape} != {base.shape}'
    np.testing.assert_allclose(curr, base, atol=1e-3, err_msg=f'{metallicity}/{name} differs')