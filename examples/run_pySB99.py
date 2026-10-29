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

# Model properties display
print(sb99)

# Run the track computations and grid interpolation
pop = sb99.population_synthesis()

# Save the results (previous folder content is deleted)
sb99.save_results()