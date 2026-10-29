# pyStarburst99: stellar population synthesis in Python

pyStarburst99 is a Python translation and update of the Starburst99 Fortran code. It predicts the integrated properties 
of star-forming galaxies from evolutionary tracks and stellar spectral libraries.

## Key features

- Instantaneous burst (ISF) and continuous star formation (CSF) models.
- Seven metallicities, from the Milky Way centre (Z = 0.02) to Z = 0.
- GENEC (rotating and non-rotating) and MESA (Pauli+2026) evolutionary tracks, with stars up to 300–500 M$_\odot$.
- Fastwind, WMbasic and PoWR OB stellar libraries, plus Wolf-Rayet and low-mass star spectra.
- Outputs: SEDs, high-resolution UV spectra, ionising fluxes, bolometric luminosity, wind power and momentum, 
  hydrogen line equivalent widths, UV slopes, colours and supernova rates.

## Quick start

```python
import pystarburst99

# Select model and output folder
metallicity = 'IZw18'  #MWC, MW, LMC, SMC, IZw18, XMP and Z0
output_folder = '/home/vital/Astrodata/SpecSy_tests/pyStarburst99_tests/version1'

# Create model compiler
sb99 = pystarburst99.Compiler(output_folder=f'{output_folder}/{metallicity}_grids',
                              Z=metallicity, track_choice='GENEC', rotation=False,
                              SED_library='FW', spectra_library='WM', star_formation_option='ISF',
                              log_age_start=6.0, log_age_end=8.0, log_age_step=0.1, log_intervals=True,
                              beta_tools=True, overwrite_results=True)

# # Or using a text file
# cfg_file = './psb99_example_input_file.toml'
# sb99 = pystarburst99.Compiler.from_file(cfg_file)

# Run the track computations and grid interpolation
pop = sb99.population_synthesis()

# Save the results (previous folder content is deleted)
sb99.save_results()
```

The inputs not provided take their values from the `[default_settings]` section of `pystarburst99.toml`.

## Model options

| Input                                | Options                                 | Description                                                           |
|--------------------------------------|-----------------------------------------|-----------------------------------------------------------------------|
| `Z`                                  | MWC, MW, LMC, SMC, IZw18, XMP, Z0       | Metallicity: 0.02, 0.014, 0.006, 0.002, 0.0004, 1e-5 and 0            |
| `star_formation_option`              | ISF, CSF                                | Instantaneous burst (`M_total`) or continuous star formation (`SFR`)  |
| `track_choice`                       | GENEC, Pauli                            | GENEC or MESA (Pauli+2026) evolutionary tracks                        |
| `rotation`                           | True, False                             | GENEC tracks with 0.4 v<sub>crit</sub> rotation                       |
| `SED_library`                        | FW, WM, PoWR                            | Low-resolution OB library for the SEDs and ionising fluxes            |
| `spectra_library`                    | WM, PoWR                                | High-resolution OB library for the UV line profiles                   |
| `IMF_exponents`, `IMF_mass_limits`   | e.g. [1.3, 2.3], [0.1, 0.5, 120]        | Power-law IMF (Kroupa by default)                                     |
| `run_speed_mode`                     | FAST, DEFAULT, HIGH_RES                 | Resolution of the isochrone mass interpolation                        |

## Citation

If you found pyStarburst99 useful, please remember to cite 
[Hawcroft et al. (2025)](https://doi.org/10.3847/1538-4365/adddb6) 📝:

```bibtex
@ARTICLE{pySB99_paper,
       author = {{Hawcroft}, C. and {Leitherer}, C. and {Arangur{\'e}}, O. and {Chisholm}, J. and
                 {Ekstr{\"o}m}, S. and {Martinet}, S. and {Martins}, L.~P. and {Meynet}, G. and
                 {Morisset}, C. and {Sander}, A.~A.~C. and {Wofford}, A.},
        title = "{pySTARBURST99: The Next Generation of STARBURST99}",
      journal = {\apjs},
         year = 2025,
        month = sep,
       volume = {280},
       number = {1},
          eid = {5},
        pages = {5},
          doi = {10.3847/1538-4365/adddb6},
archivePrefix = {arXiv},
       eprint = {2505.24841},
}
```

## Contact

Please let us know your thoughts and questions at pystarburst99@gmail.com.
