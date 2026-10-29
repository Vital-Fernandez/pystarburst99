import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from types import SimpleNamespace


def calc_Nostars(IMF_masses, IMF_exponents, IMF_mass_limits, M_total):
    '''
    Compute normalisation constants for IMF for a 1-power or 2-power law IMF. Then computes number of stars per mass bin.
    A is the normalisation constants for each part of the IMF.
    Only supports 1 or 2 IMF intervals for now, has to be updated to have more than 2
    '''
    #Determine number of IMF intervals based on input IMF intervals
    if len(IMF_exponents) == 1:
        N_IMF_intervals = 1
    elif len(IMF_exponents) > 1:
        N_IMF_intervals = 2

    #compute variables for IMF normalisation constants
    if N_IMF_intervals > 1:
        N_IMF_intervals = len(IMF_exponents)
        A_ic = np.zeros_like(IMF_exponents)
        A_ic[0] = 1.

        for exponent_index in range(len(IMF_exponents)):
            exponent_index=exponent_index+1 #skip first IMF_exponent
            if exponent_index == len(IMF_exponents):
                break #this is just a quick stop when we reach the end of the IMF exponents (if i=2 when we have len=2 for 2 exponents then it stops instead of looking for ind=2 )
            A_i = A_ic[exponent_index-1] * (IMF_mass_limits[exponent_index]**(IMF_exponents[exponent_index] - IMF_exponents[exponent_index-1]))
            #print(Aici)
            A_ic[exponent_index] = A_i #set new value
        k_ic=[]
        for exponent_index in range(len(IMF_exponents)):
            if IMF_exponents[exponent_index] == 2.0:
                k_i = np.log(IMF_mass_limits[exponent_index+1])-np.log(IMF_mass_limits[exponent_index])
            else:
                k_i = ( IMF_mass_limits[exponent_index+1]**(2.0 - IMF_exponents[exponent_index]) - IMF_mass_limits[exponent_index]**(2.0 - IMF_exponents[exponent_index]) ) / (2 - IMF_exponents[exponent_index])
            k_ic.append(k_i)

        Ak_ic = A_ic * k_ic
        Ak = sum(Ak_ic)

    else: #If N_IMF_intervals = 1
        A_ic = 1
        k_ic = ( IMF_mass_limits[1]**(2.0 - IMF_exponents[0]) - IMF_mass_limits[0]**(2.0 - IMF_exponents[0]) ) / (2 - IMF_exponents[0])
        Ak = A_ic * k_ic

    S = M_total / Ak
    A = A_ic * S #Here are the normalisation constants

    xmhigh = np.full_like(IMF_masses, 0)
    xmlow = np.full_like(IMF_masses, 0)

    for mass_index in range(len(IMF_masses)): #set width of bin for each individual IMF mass through upper and lower limits
        if IMF_masses[mass_index] == min(IMF_masses):
            xmhigh[mass_index] = 0.5*(IMF_masses[mass_index]+IMF_masses[mass_index+1])
            xmlow[mass_index]  = IMF_masses[mass_index]
        elif IMF_masses[mass_index] == max(IMF_masses):
            xmhigh[mass_index] = IMF_masses[mass_index]
            xmlow[mass_index]  = 0.5*(IMF_masses[mass_index-1]+IMF_masses[mass_index])
        else:
            xmhigh[mass_index] = 0.5*(IMF_masses[mass_index]+IMF_masses[mass_index+1])
            xmlow[mass_index] = 0.5*(IMF_masses[mass_index]+IMF_masses[mass_index-1])

    dens = []
    #compute number of stars per mass bin
    if N_IMF_intervals == 2:
        for mass_index in range(len(IMF_masses)):
            dens_i = A[1] * ( xmhigh[mass_index]**(1-IMF_exponents[1]) - xmlow[mass_index]**(1-IMF_exponents[1]) ) / (1 - IMF_exponents[1])
            dens.append(dens_i)
    else:
        for mass_index in range(len(IMF_masses)):
            dens_i = A * ( xmhigh[mass_index]**(1-IMF_exponents[0]) - xmlow[mass_index]**(1-IMF_exponents[0]) )  / (1 - IMF_exponents[0])
            dens.append(dens_i)

    N_stars = np.array(dens)
    total_mass = N_stars * IMF_masses
    #print('Total No stars = ','{:0.4E}'.format(sum(N_stars).item()),'Total mass = ','{:0.4E}'.format(np.sum(total_mass)))
    return(N_stars, IMF_masses, dens, xmhigh, xmlow)

def read_spectra_grid(spec_file):
    df = pd.read_csv(spec_file)

    ind_spec = [i for i in range(len(df)) if df['start'][i] == ' CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC']

    spec_params = []
    spectra = []
    for i in range(len(ind_spec)-1):
        spec_param = df['start'][ind_spec[i]+1]
        spec_params.append(spec_param)

    for i in range(len(ind_spec)-1):
        spec_param = df['start'][ind_spec[i]+2 : ind_spec[i+1]]
        spectra.append(spec_param)
    return(spec_params, spectra)

def reformat_spec(spectrum):
    wave_reform = []
    flux_reform = []
    for i in spectrum:
        spec_wavep, spec_fluxp = i.split()
        wavep_app = float(spec_wavep)
        wave_reform.append(wavep_app)
        fluxp_app = float(spec_fluxp) * 4.0 * 3.14142
        flux_reform.append(fluxp_app)
    flux_reform = np.array(flux_reform)# + 12.566# * 4e10
    spec_reform = np.column_stack((wave_reform, flux_reform))
    return (spec_reform)

def reformat_spec_WR(spectrum):
    wave_reform = []
    flux_reform = []
    for i in spectrum:
        spec_wavep, spec_fluxp = i.split()
        wavep_app = float(spec_wavep)
        wave_reform.append(wavep_app)
        fluxp_app = float(spec_fluxp)# * 12.566
        flux_reform.append(fluxp_app)
    flux_reform = np.array(flux_reform)# + 12.566# * 4e10
    spec_reform = np.column_stack((wave_reform, flux_reform))
    return (spec_reform)

def reform_spec_grid(spectra, spec_params):
    reformed_spec_grid = []
    for i in range(len(spectra)):
        reformatted_spec = reformat_spec(spectra[i])
        reformed_spec_grid.append(reformatted_spec)

    spec_params_id = []
    spec_params_teff = []
    spec_params_logl = []
    spec_params_logg = []
    for i in spec_params:
        spec_id, spec_teff, spec_logl, spec_logg = i.split()
        spec_id = float(spec_id)
        spec_params_id.append(spec_id)
        spec_teff = float(spec_teff)
        spec_params_teff.append(spec_teff)
        spec_logl = float(spec_logl)
        spec_params_logl.append(spec_logl)
        spec_logg = float(spec_logg)
        spec_params_logg.append(spec_logg)
    spec_params_teff = 10**np.array(spec_params_teff)
    spec_params_reform = np.column_stack((spec_params_id, spec_params_teff, spec_params_logl, spec_params_logg))
    return(reformed_spec_grid, spec_params_reform, spec_params_teff, spec_params_logl, spec_params_logg)

def reform_spec_grid_WR(spectra, spec_params):
    reformed_spec_grid = []
    for i in range(len(spectra)):
        reformatted_spec = reformat_spec_WR(spectra[i])
        reformed_spec_grid.append(reformatted_spec)

    spec_params_id = []
    spec_params_teff = []
    for i in spec_params:
        spec_id, spec_teff = i.split()
        spec_id = float(spec_id)
        spec_params_id.append(spec_id)
        spec_teff = float(spec_teff)
        spec_params_teff.append(spec_teff)
    spec_params_teff = np.array(spec_params_teff)
    spec_params_reform = np.column_stack((spec_params_id, spec_params_teff))
    return(reformed_spec_grid, spec_params_reform, spec_params_teff)

def reform_spec_grid_powr(spectra, spec_params):
    reformed_spec_grid = []
    for i in range(len(spectra)):
        reformatted_spec = reformat_spec_WR(spectra[i])
        reformed_spec_grid.append(reformatted_spec)

    spec_params_id = []
    spec_params_teff = []
    spec_params_radius = []
    spec_params_length = []
    for i in spec_params:
        spec_id, spec_teff, spec_radius, spec_length = i.split()
        spec_id = float(spec_id)
        spec_params_id.append(spec_id)
        spec_teff = float(spec_teff)
        spec_params_teff.append(spec_teff)
        spec_radius = float(spec_radius)
        spec_params_radius.append(spec_radius)
        spec_length = float(spec_length)
        spec_params_length.append(spec_length)
    spec_params_teff = np.array(spec_params_teff)
    spec_params_reform = np.column_stack((spec_params_id, spec_params_teff, spec_params_radius, spec_params_length))
    return(reformed_spec_grid, spec_params_reform, spec_params_teff, spec_params_radius, spec_params_length)

def integrate_spec_grid(reformed_spec_grid):
    integrated_spectra = []
    for i in range(len(reformed_spec_grid)): # TODO np.trapz has been renamed
        integrated_spectrum = np.trapezoid(reformed_spec_grid[i][:,1], reformed_spec_grid[i][:,0])
        integrated_spectra.append(integrated_spectrum)
    return(integrated_spectra)

def integrate_spec_grid_powr(wave_grid, reformed_spec_grid):
    integrated_spectra = []
    for i in range(len(reformed_spec_grid)):
        integrated_spectrum = np.trapezoid(wave_grid, reformed_spec_grid[i,:])
        integrated_spectra.append(integrated_spectrum)
    return(integrated_spectra)

def interpolate_param(tracks_parameter, track_masses, run_speed_mode):
    '''
    Performs an interpolation between two adjacent tracks in evolutionary diagrams. This is done to create synthetic
    tracks for each mass in the grid to fill in the gaps offered by the tracks.
    Returns an array of 23 sub-arrays (the interpolation between each of the 24 tracks). These contain 400 arrays (for each point in the tracks)
    which have a length indicating how many synthetic tracks are produced between the two tracks, depends on mass range
    covered by the two tracks, e.g. 36 tracks are needed between 120 and 85 Msol but only one is needed between the lowest mass tracks.

    Should really rework the arrays so the output is just an array of 120 tracks of length 400.
    '''
    grid_masses_adjinterp_total = []
    grid_params_adjinterp_total = []
    for i in range(len(track_masses)-1):
        track_mass_upper = np.flip(track_masses[i])
        track_param_upper = np.flip(tracks_parameter[i])
        track_mass_lower = np.flip(track_masses[i+1])
        track_param_lower = np.flip(tracks_parameter[i+1])

        initial_mass_upper = track_mass_upper[-1]
        initial_mass_lower = track_mass_lower[-1]

        if run_speed_mode == 'DEFAULT':
            if round(initial_mass_upper) > 7 and round(initial_mass_upper) < 35:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 15
            else:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 )

        if run_speed_mode == 'FAST':
            inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 1

        if run_speed_mode == 'HIGH_RES':
            if round(initial_mass_upper) > 9 and round(initial_mass_upper) < 15:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 15
            elif round(initial_mass_upper) > 15 and round(initial_mass_upper) < 35:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 5
            elif round(initial_mass_upper) < 9:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 10
            elif round(initial_mass_upper) > 35 and round(initial_mass_upper) < 60:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 ) * 5
            elif round(initial_mass_upper) > 60:
                inter_track_sampling = ( initial_mass_upper - initial_mass_lower +1 )


        track_masses_adjinterp = np.column_stack((track_mass_lower, track_mass_upper))
        track_param_adjinterp = np.column_stack((track_param_lower, track_param_upper))

        grid_masses_adjinterp = []

        for j in range(len(track_mass_upper)):
            grid_mass_adjinterp = np.linspace(track_mass_lower[j], track_mass_upper[j], round(inter_track_sampling))
            grid_masses_adjinterp.append(grid_mass_adjinterp)

        grid_params_adjinterp = []
        for track_param, track_mass, grid_mass in zip(track_param_adjinterp,track_masses_adjinterp,grid_masses_adjinterp):
            #interpolate in log temp
            #    temps_interp_grid_fn = interp1d(track_mass, np.log10(track_temp), kind='linear')#, fill_value='extrapolate')
            #interpolate in temp (not log temp)

            #grid_params_adjinterp_fn = np.interp(grid_mass, track_mass, track_param)

            grid_params_adjinterp_fn = interp1d(track_mass, track_param, kind='linear', fill_value='extrapolate')
            grid_params_adjinterp_loop = grid_params_adjinterp_fn(grid_mass)
            grid_params_adjinterp.append(grid_params_adjinterp_loop)

        grid_masses_adjinterp_total.append(grid_masses_adjinterp)
        grid_params_adjinterp_total.append(grid_params_adjinterp)

    return grid_masses_adjinterp_total, grid_params_adjinterp_total

def rearrange_grid_array(grid_array):

    '''
    The interpolation returns a grid with essentially the parameters running along the x-axis of the array with the masses along the y.
    This rearranges the grid interpolation array to give one array per mass.
    '''

    lengths = []
    for i in range(len(grid_array)):
        grid_sizes = len(grid_array[i][0])
        lengths.append(grid_sizes)

    total_grid_size = sum(lengths)

    rearranged = []
    for k in range(len(grid_array)):
        for j in range(len(grid_array[k][0])-1, -1, -1):
            for i in range(len(grid_array[k])):
                rearranged.append(grid_array[k][i][j])
    rearranged = np.array_split(rearranged, total_grid_size)
    return(rearranged)

def get_timestep_0_ind(timestep, grid_ages, grid_masses, times_steps, IMF_mass_limits):
    '''
    As a result of the interpolation there will be a duplicate track for each original Geneva track.
    This function indexes the arrays which have unique initial masses and returns only the unique masses, excluding the duplicates.
    There is still a slight issue in that this returns tracks which have decimal masses. The initial grid set-up is 120 masses,
    so 1-120 in steps of 1. However the tracks have masses at e.g. 0.8, 2.5 etc. so these are returned as well.
    If the tracks where all of integer mass this wouldn't be an issue, for now the choice is between adding the additional tracks masses to the grid set-up,
    or we can remove the decimal masses from the interpolated grid. Currently I'm just deleting the lowest masses up until the length of the intial grid set-up is matched.
    '''
    timestep_masses = []
    for i in range(len(grid_ages)):
        ind_nearest_age = np.argmin((grid_ages[i] - times_steps[0])**2)
        nearest_mass = grid_masses[i][ind_nearest_age]

        timestep_masses.append(nearest_mass)

    timestep_masses_unique = np.unique(timestep_masses, return_index=True) #Gives unique initial masses to avoid duplication and adds index as second column

    timestep_masses_IMF_ind = [i for i in range(len(timestep_masses_unique[0])) if timestep_masses_unique[0][i] < IMF_mass_limits[-1]]
    #timestep_masses_red = timestep_masses_2[0][timestep_masses_100_ind]
    timestep_masses_red_ind = timestep_masses_unique[1][timestep_masses_IMF_ind]

    #return(timestep_masses_ind)
    return(timestep_masses_red_ind)

def get_timestep_params(timestep, timestep_mass_ind, track_choice, grids):
    '''
    Get stellar parameters for each timestep. A time is given as input, this is matched to the closest age in each
    grid_track and the corresponding parameters are pulled for that point. Then the index to remove duplicates, defined above,
    is applied here also. Finally, if a star has a mass of 0.0 as flagged for SN, the temperature is also set to 0.0 here.

    grids is a dictionary with the interpolated grids (keys grid_ages, grid_masses, grid_temps, grid_lums,
    grid_H_abundances, grid_He_abundances, grid_12C_abundances, grid_14N_abundances, grid_16O_abundances,
    grid_core_temps and grid_mass_loss_rates).
    '''
    grid_ages = grids['grid_ages']
    grid_masses = grids['grid_masses']
    grid_temps = grids['grid_temps']
    grid_lums = grids['grid_lums']
    grid_H_abundances = grids['grid_H_abundances']
    grid_He_abundances = grids['grid_He_abundances']
    grid_12C_abundances = grids['grid_12C_abundances']
    grid_14N_abundances = grids['grid_14N_abundances']
    grid_16O_abundances = grids['grid_16O_abundances']
    grid_core_temps = grids['grid_core_temps']
    grid_mass_loss_rates = grids['grid_mass_loss_rates']

    timestep_ages = []
    timestep_masses = []
    timestep_temps = []
    timestep_lums = []
    timestep_H_abundances = []
    timestep_He_abundances = []
    timestep_12C_abundances = []
    timestep_14N_abundances = []
    timestep_16O_abundances = []
    timestep_core_temps = []
    timestep_mass_loss_rates = []

    '''Timestep parameters are determined through loop of grid sampling therefore grid sets size/resolution of masses/parameters in the final populations'''
    for i in range(len(grid_ages)):
        #ind_nearest_age = np.argmin((grid_ages[i] - timestep)**2)
        nearest_agecalc = abs(grid_ages[i] - timestep)
        ind_nearest_age = np.argmin(nearest_agecalc)
        nearest_age = grid_ages[i][ind_nearest_age]

        nearest_mass = grid_masses[i][ind_nearest_age]
        nearest_temp = grid_temps[i][ind_nearest_age]
        nearest_lum = grid_lums[i][ind_nearest_age]
        nearest_H_abundance = grid_H_abundances[i][ind_nearest_age]
        nearest_He_abundance = grid_He_abundances[i][ind_nearest_age]
        nearest_12C_abundance = grid_12C_abundances[i][ind_nearest_age]
        nearest_14N_abundance = grid_14N_abundances[i][ind_nearest_age]
        nearest_16O_abundance = grid_16O_abundances[i][ind_nearest_age]
        nearest_core_temp = grid_core_temps[i][ind_nearest_age]
        nearest_mass_loss_rate = grid_mass_loss_rates[i][ind_nearest_age]

        timestep_ages.append(nearest_age)
        timestep_masses.append(nearest_mass)
        timestep_temps.append(nearest_temp)
        timestep_lums.append(nearest_lum)
        timestep_H_abundances.append(nearest_H_abundance)
        timestep_He_abundances.append(nearest_He_abundance)
        timestep_12C_abundances.append(nearest_12C_abundance)
        timestep_14N_abundances.append(nearest_14N_abundance)
        timestep_16O_abundances.append(nearest_16O_abundance)
        timestep_core_temps.append(nearest_core_temp)
        timestep_mass_loss_rates.append(nearest_mass_loss_rate)

    timestep_ages_final = []
    timestep_masses_final = []
    timestep_temps_final = []
    timestep_lums_final = []
    timestep_H_abundances_final = []
    timestep_He_abundances_final = []
    timestep_12C_abundances_final = []
    timestep_14N_abundances_final = []
    timestep_16O_abundances_final = []
    timestep_core_temps_final = []
    timestep_mass_loss_rates_final = []

    for i in timestep_mass_ind:
        timestep_age_temp = timestep_ages[i]
        timestep_temp_temp = timestep_temps[i]
        timestep_lum_temp = timestep_lums[i]
        timestep_mass_temp = timestep_masses[i]
        timestep_H_abundance_temp = timestep_H_abundances[i]
        timestep_He_abundance_temp = timestep_He_abundances[i]
        timestep_12C_abundance_temp = timestep_12C_abundances[i]
        timestep_14N_abundance_temp = timestep_14N_abundances[i]
        timestep_16O_abundance_temp = timestep_16O_abundances[i]
        timestep_core_temp_temp = timestep_core_temps[i]
        timestep_mass_loss_rates_temp = timestep_mass_loss_rates[i]
        timestep_ages_final.append(timestep_age_temp)
        timestep_temps_final.append(timestep_temp_temp)
        timestep_lums_final.append(timestep_lum_temp)
        timestep_masses_final.append(timestep_mass_temp)
        timestep_H_abundances_final.append(timestep_H_abundance_temp)
        timestep_He_abundances_final.append(timestep_He_abundance_temp)
        timestep_12C_abundances_final.append(timestep_12C_abundance_temp)
        timestep_14N_abundances_final.append(timestep_14N_abundance_temp)
        timestep_16O_abundances_final.append(timestep_16O_abundance_temp)
        timestep_core_temps_final.append(timestep_core_temp_temp)
        timestep_mass_loss_rates_final.append(timestep_mass_loss_rates_temp)

    timestep_teffs_calc = 10**np.array(timestep_temps_final)
    timestep_lum_calc = (10**np.array(timestep_lums_final) * 3.839e33)
    radii_denom = (12.566 * 5.670e-5 * (timestep_teffs_calc**4))
    timestep_radii_calc = ((timestep_lum_calc / radii_denom)**0.5) * 100

    logg_num = (6.67e-12) * np.array(timestep_masses_final) * (1.9891e33) *1e8
    timestep_gravities_final =  (logg_num) / (timestep_radii_calc**2)
    timestep_loggs_final = np.log10( timestep_gravities_final )

    timestep_H_abundances_final = np.array(timestep_H_abundances_final)
    timestep_He_abundances_final = np.array(timestep_He_abundances_final)
    timestep_12C_abundances_final = np.array(timestep_12C_abundances_final)
    timestep_14N_abundances_final = np.array(timestep_14N_abundances_final)
    timestep_16O_abundances_final = np.array(timestep_16O_abundances_final)

    timestep_cnr = timestep_12C_abundances_final / timestep_14N_abundances_final
    timestep_coher=((timestep_12C_abundances_final/12.)+(timestep_16O_abundances_final/16.))/(timestep_He_abundances_final/4.)

    timestep_uncorr_teff_final = timestep_teffs_calc

    #Turn off for plotting isochrones
    if track_choice == 'GENEC':
        WR_correction_factor = 0.6
        for i in range(len(timestep_masses_final)):
            if timestep_H_abundances_final[i] < 0.1:# and timestep_temps_final[i] > 4.4:
                timestep_core_teff_final = 10**timestep_core_temps_final[i]
                corrected_teff=timestep_core_teff_final+(WR_correction_factor-1.0)*(timestep_core_teff_final-timestep_teffs_calc[i])
                timestep_temps_final[i]=np.log10(corrected_teff)
                #print('Grid teff = ',grid_teff)
                #print('Grid core teff = ',grid_core_teff)
                #print('Grid teff corr= ',corrected_teff)

    #ind_SN_stars = [i for i in range(len(timestep_masses_final)) if timestep_masses_final[i] < 0.5]
    #ind_SN_stars = [i for i in range(len(timestep_lums_final)) if timestep_lums_final[i] < -19.]
    timestep_temps_final = np.array(timestep_temps_final)
    #timestep_temps_final[ind_SN_stars] = 0.0


    return(timestep_ages_final,timestep_temps_final,timestep_lums_final,timestep_masses_final,timestep_H_abundances_final, timestep_loggs_final, timestep_mass_loss_rates_final, timestep_12C_abundances_final, timestep_14N_abundances_final, timestep_16O_abundances_final, timestep_masses, timestep_cnr, timestep_coher, timestep_uncorr_teff_final)

def get_specsyn_params(timestep_temps_final, timestep_masses_final, timestep_lums_final):

    specsyn_teffs = []
    specsyn_loggs = []
    specsyn_radii = []
    specsyn_cotests = []
    specsyn_bbfluxes = []
    for i in range(len(timestep_temps_final)):
        teff_specsyn = 10**(timestep_temps_final[i])

        logg_specsyn = np.log10(timestep_masses_final[i]) + (4.* timestep_temps_final[i]) - timestep_lums_final[i] - 10.6
        radius_specsyn = 10.**(10.8426 + 0.5*timestep_lums_final[i] - (2.*timestep_temps_final[i]) + 7.52)
        cotest=5.71*np.log10(teff_specsyn) - 21.95
        if teff_specsyn == 0.0:
            specsyn_bbflux = 0.0
        else:
            specsyn_bbflux = 5.6696196e-05 * teff_specsyn**4.

        specsyn_teffs.append(teff_specsyn)
        specsyn_loggs.append(logg_specsyn)
        specsyn_radii.append(radius_specsyn)
        specsyn_cotests.append(cotest)
        specsyn_bbfluxes.append(specsyn_bbflux)

    ind_SN_stars = [i for i in range(len(specsyn_teffs)) if specsyn_teffs[i] == 1.0]
    specsyn_teffs = np.array(specsyn_teffs)
    specsyn_teffs[ind_SN_stars] = 0.0

    return(specsyn_teffs, specsyn_loggs, specsyn_radii, specsyn_cotests, specsyn_bbfluxes)

def assign_spectra_to_grid_WR(timestep_temps_final,spec_params_reform, spec_params_teff, spec_params_logl, timestep_lums_final, timestep_H_abundances_final, spec_params_logg, timestep_masses_final, initial_masses, specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher, No_stars, timestep_loggs_final, spec_lib):
    wave_grid = spec_lib['wave_grid']
    empty_flux = spec_lib['empty_flux']
    minimum_wr_mass = spec_lib['minimum_wr_mass']
    WN_spec_params_reform = spec_lib['WN_spec_params_reform']
    WN_spec_params_teff = spec_lib['WN_spec_params_teff']
    WN_spec_params_logl = spec_lib['WN_spec_params_logl']
    WN_reformed_spec_grid = spec_lib['WN_reformed_spec_grid']
    WN_integrated_spectra = spec_lib['WN_integrated_spectra']
    WC_spec_params_reform = spec_lib['WC_spec_params_reform']
    WC_spec_params_teff = spec_lib['WC_spec_params_teff']
    WC_spec_params_logl = spec_lib['WC_spec_params_logl']
    WC_reformed_spec_grid = spec_lib['WC_reformed_spec_grid']
    WC_integrated_spectra = spec_lib['WC_integrated_spectra']
    reformed_spec_grid = spec_lib['reformed_spec_grid']
    integrated_spectra = spec_lib['integrated_spectra']
    lowmass_params = spec_lib['lowmass_params']
    lowmass_teffs = spec_lib['lowmass_teffs']
    lowmass_loggs = spec_lib['lowmass_loggs']
    lm_spec_params_logl = spec_lib['lm_spec_params_logl']
    lowmass_spec = spec_lib['lowmass_spec']
    lowmass_int_spec = spec_lib['lowmass_int_spec']

    assigned_integrated_spectra=[]
    assigned_spec_teff=[]
    assigned_spec_logl=[]
    assigned_spec_logg=[]
    assigned_spectra=[]
    timestep_teffs_final = 10**timestep_temps_final
    population_choice = []

    for j in range(len(timestep_temps_final)):
        distance_to_spec = []

        if timestep_lums_final[j] == -20.0:
            population_choice.append('nope')
            assigned_spec_teff.append(0.0)
            assigned_spectra.append(np.column_stack((wave_grid, empty_flux)))
            assigned_integrated_spectra.append(1e-30)

        elif timestep_temps_final[j] > 4.4 and timestep_H_abundances_final[j] < 0.4 and initial_masses[j] > minimum_wr_mass:# e.g. 20 for solar but can change
        #elif timestep_temps_final[j] > 4.4 and timestep_H_abundances_final[j] > 0.9 and initial_masses[j] > minimum_wr_mass:# e.g. 20 for solar but can change
            population_choice.append('WR')
            if timestep_H_abundances_final[j] > 0.1:
                for i in range(len(WN_spec_params_reform)):
                    distance_to_spec.append((WN_spec_params_teff[i] - 10**timestep_temps_final[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WN_spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = WN_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                assigned_spec_logg.append(-1.)
                assigned_spectra.append(WN_reformed_spec_grid[nearest_spec_ind])
                assigned_integrated_spectra.append(WN_integrated_spectra[nearest_spec_ind])
            elif timestep_cnr[j] < 10.:
                for i in range(len(WN_spec_params_reform)):
                    distance_to_spec.append((WN_spec_params_teff[i] - 10**timestep_temps_final[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WN_spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = WN_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                assigned_spec_logg.append(-1.)
                assigned_spectra.append(WN_reformed_spec_grid[nearest_spec_ind])
                assigned_integrated_spectra.append(WN_integrated_spectra[nearest_spec_ind])
            elif timestep_coher[j] < 0.5:
                for i in range(len(WC_spec_params_reform)):
                    distance_to_spec.append((WC_spec_params_teff[i] - 10**timestep_temps_final[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                assigned_spec_logg.append(-1.)
                assigned_spectra.append(WC_reformed_spec_grid[nearest_spec_ind])
                assigned_integrated_spectra.append(WC_integrated_spectra[nearest_spec_ind])
#                vinfs.append(vinf)
            elif timestep_coher[j] < 1.:
                for i in range(len(WC_spec_params_reform)):
                    distance_to_spec.append((WC_spec_params_teff[i] - 10**timestep_temps_final[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                assigned_spec_logg.append(-1.)
                assigned_spectra.append(WC_reformed_spec_grid[nearest_spec_ind])
                assigned_integrated_spectra.append(WC_integrated_spectra[nearest_spec_ind])
            elif timestep_coher[j] >= 1.:
                for i in range(len(WC_spec_params_reform)):
                    distance_to_spec.append((WC_spec_params_teff[i] - 10**timestep_temps_final[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                near_spec_logl = WC_spec_params_logl[nearest_spec_ind]
                assigned_spec_logl.append(near_spec_logl)
                assigned_spec_logg.append(-1.)
                assigned_spectra.append(WC_reformed_spec_grid[nearest_spec_ind])
                assigned_integrated_spectra.append(WC_integrated_spectra[nearest_spec_ind])

        elif timestep_temps_final[j] < 3.65 and initial_masses[j] > minimum_wr_mass:
            population_choice.append(No_stars[j])

        elif timestep_teffs_final[j] > 25000.:# and specsyn_loggs[j] > 2.2 and specsyn_loggs[j] < specsyn_cotests[j]:

            population_choice.append('OB')
            for i in range(len(spec_params_reform)):
                #distance_to_spec.append((spec_params_teff[i] - 10**timestep_temps_final[j])**2 + 1e5*(spec_params_logl[i] - timestep_lums_final[j])**2)
                #distance_to_spec.append((spec_params_teff[i] - 10**timestep_temps_final[j])**2)# + 1e5*(spec_params_logl[i] - timestep_lums_final[j])**2)
                #distance_to_spec.append((spec_params_teff[i] - 10**timestep_temps_final[j])**2 + (spec_params_logl[i] - timestep_lums_final[j])**2)
                #distance_to_spec.append((10**timestep_temps_final[j] - spec_params_teff[i])**2 + 5*(timestep_loggs_final[j] - spec_params_logg[i])**2)
                #distance_to_spec.append((spec_params_teff[i] - 10**timestep_temps_final[j])**2 + (spec_params_logg[i] - timestep_loggs_final[j])**2)
                distance_to_spec.append(abs(spec_params_teff[i] - 10**timestep_temps_final[j]) + abs(spec_params_logg[i] - timestep_loggs_final[j]))
            nearest_spec_ind = np.argmin(distance_to_spec)
            near_spec_temp = spec_params_teff[nearest_spec_ind]
            assigned_spec_teff.append(near_spec_temp)
            near_spec_logl = spec_params_logl[nearest_spec_ind]
            assigned_spec_logl.append(near_spec_logl)
            near_spec_logg = spec_params_logg[nearest_spec_ind]
            assigned_spec_logg.append(near_spec_logg)
            assigned_spectra.append(reformed_spec_grid[nearest_spec_ind])
            assigned_integrated_spectra.append(integrated_spectra[nearest_spec_ind])

        elif timestep_teffs_final[j] < 2000.:
            population_choice.append('nope')
            assigned_spec_teff.append(0.0)
            assigned_spectra.append(np.column_stack((wave_grid, empty_flux)))
            assigned_integrated_spectra.append(1e-30)
        else:
            population_choice.append('lowmass')
            for i in range(len(lowmass_params)):
                distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2 + (lowmass_loggs[i] - timestep_loggs_final[j])**2)
            nearest_spec_ind = np.argmin(distance_to_spec)
            near_spec_temp = lowmass_teffs[nearest_spec_ind]
            assigned_spec_teff.append(near_spec_temp)
            near_spec_logl = lm_spec_params_logl[nearest_spec_ind]
            assigned_spec_logl.append(timestep_lums_final[j])
            near_spec_logg = lowmass_loggs[nearest_spec_ind]
            assigned_spec_logg.append(near_spec_logg)
            #assigned_spec_logl.append(near_spec_logl)
            assigned_spectra.append(lowmass_spec[nearest_spec_ind])
            assigned_integrated_spectra.append(lowmass_int_spec[nearest_spec_ind])

    return(assigned_integrated_spectra,assigned_spec_teff,assigned_spectra, population_choice, assigned_spec_logl, assigned_spec_logg)

def assign_spectra_to_grid_hires(timestep_temps_final, hires_teffs, hires_loggs, timestep_lums_final, timestep_H_abundances_final, timestep_masses_final, initial_masses, specsyn_cotests, specsyn_loggs, timestep_cnr, timestep_coher, timestep_radii_final, spec_lib):
    lowmass_params = spec_lib['lowmass_params']
    lowmass_flux = spec_lib['lowmass_flux']
    lowmass_wave_grid = spec_lib['lowmass_wave_grid']
    hires_wave_grid = spec_lib['hires_wave_grid']
    minimum_wr_mass = spec_lib['minimum_wr_mass']
    spectra_library = spec_lib['spectra_library']
    WN_spec_params_reform_powr = spec_lib['WN_spec_params_reform_powr']
    WC_spec_params_reform_powr = spec_lib['WC_spec_params_reform_powr']
    WN_reformed_spec_grid_powr = spec_lib['WN_reformed_spec_grid_powr']
    WC_reformed_spec_grid_powr = spec_lib['WC_reformed_spec_grid_powr']
    hires_spec = spec_lib['hires_spec']
    hires_cont = spec_lib['hires_cont']
    hires_int_spec = spec_lib['hires_int_spec']

    assigned_integrated_spectra=[]
    assigned_spec_teff=[]
    assigned_spec_logg=[]
    assigned_spectra=[]
    assigned_cont=[]
    timestep_teffs_final = 10**timestep_temps_final
    hires_temps = np.log10(hires_teffs)
    hires_choice = []

    lowmass_teffs = lowmass_params[:,0]
    lowmass_loggs = lowmass_params[:,1]
    lm_spec_params_logl = np.full_like(lowmass_teffs, 1.) #very wrong for now
    lowmass_spec = []
    for i in range(len(lowmass_flux)):
        lm_spectrum = np.column_stack((lowmass_wave_grid, lowmass_flux[i]))
        lowmass_spec.append(lm_spectrum)
    lowmass_int_spec = integrate_spec_grid(lowmass_spec)

    empty_hires_flux = np.full_like(hires_wave_grid, 0.0)
    WN_spec_params_teff_powr = WN_spec_params_reform_powr[:,1]
    WC_spec_params_teff_powr = WC_spec_params_reform_powr[:,1]
    WN_spec_params_radius_powr = WN_spec_params_reform_powr[:,2]
    WC_spec_params_radius_powr = WC_spec_params_reform_powr[:,2]
    WN_integrated_spectra_powr = integrate_spec_grid(WN_reformed_spec_grid_powr)
    WC_integrated_spectra_powr = integrate_spec_grid(WC_reformed_spec_grid_powr)

    cont_wave_grid_calc = [899.9,1085.,1150.,1280.,1310.,1360.,1430.,1480.,1510.,1580.,1630.,1680.,1740.,1785.,1820.,1875.,2035.,2235.,2565.,2745.,2895.,2999.]

    for j in range(len(timestep_temps_final)):
        distance_to_spec = []

        if timestep_lums_final[j] == -20.0:
            hires_choice.append('nope')
            assigned_spec_teff.append(0.0)
            assigned_spectra.append(np.column_stack((hires_wave_grid, empty_hires_flux)))
            assigned_integrated_spectra.append(1e-30)
            assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

        elif timestep_temps_final[j] > 4.4 and timestep_H_abundances_final[j] < 0.40 and initial_masses[j] >= minimum_wr_mass:# e.g. 20 for solar but can change
            if timestep_H_abundances_final[j] > 0.098:
                hires_choice.append('WNL')
                for i in range(len(WN_spec_params_reform_powr)):
                    distance_to_spec.append(abs(timestep_teffs_final[j] - WN_spec_params_teff_powr[i]))
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WN_spec_params_teff_powr[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(1.0)
                assigned_spectrum = WN_reformed_spec_grid_powr[nearest_spec_ind]
                assinged_radius = WN_spec_params_radius_powr[nearest_spec_ind]
                assigned_flux_scaled = assigned_spectrum[:,1] *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_flux_scaled)
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(WN_integrated_spectra_powr[nearest_spec_ind])
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont_flux_scaled = spec_cont[:,1] * assigned_flux_resampled[4050] / spec_cont[:,1][4050]
                assigned_cont.append(np.column_stack((hires_wave_grid, cont_flux_scaled)))
            elif timestep_cnr[j] < 10.:
                hires_choice.append('WNE')
                for i in range(len(WN_spec_params_reform_powr)):
                    distance_to_spec.append(abs(timestep_teffs_final[j] - WN_spec_params_teff_powr[i]))
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WN_spec_params_teff_powr[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(1.0)
                assigned_spectrum = WN_reformed_spec_grid_powr[nearest_spec_ind]
                assinged_radius = WN_spec_params_radius_powr[nearest_spec_ind]
                assigned_flux_scaled = assigned_spectrum[:,1] *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_flux_scaled)
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(WN_integrated_spectra_powr[nearest_spec_ind])
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont_flux_scaled = spec_cont[:,1] * assigned_flux_resampled[4050] / spec_cont[:,1][4050]
                #assigned_cont.append(np.column_stack((hires_wave_grid, cont_flux_scaled)))
                assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

            elif timestep_coher[j] < 0.5:
                hires_choice.append('WCL')
                for i in range(len(WC_spec_params_reform_powr)):
                    distance_to_spec.append(abs(timestep_teffs_final[j] - WC_spec_params_teff_powr[i]))
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff_powr[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(1.0)
                assigned_spectrum = WC_reformed_spec_grid_powr[nearest_spec_ind]
                assinged_radius = WC_spec_params_radius_powr[nearest_spec_ind]
                assigned_flux_scaled = assigned_spectrum[:,1] *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_flux_scaled)
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(WN_integrated_spectra_powr[nearest_spec_ind])
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont_flux_scaled = spec_cont[:,1] * assigned_flux_resampled[4050] / spec_cont[:,1][4050]
                #assigned_cont.append(np.column_stack((hires_wave_grid, cont_flux_scaled)))
                assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

            elif timestep_coher[j] < 1.:
                hires_choice.append('WCE')
                for i in range(len(WC_spec_params_reform_powr)):
                    distance_to_spec.append(abs(timestep_teffs_final[j] - WC_spec_params_teff_powr[i]))
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff_powr[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(1.0)
                assigned_spectrum = WC_reformed_spec_grid_powr[nearest_spec_ind]
                assinged_radius = WC_spec_params_radius_powr[nearest_spec_ind]
                assigned_flux_scaled = assigned_spectrum[:,1] *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_flux_scaled)
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(WN_integrated_spectra_powr[nearest_spec_ind])
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont_flux_scaled = spec_cont[:,1] * assigned_flux_resampled[4050] / spec_cont[:,1][4050]
                #assigned_cont.append(np.column_stack((hires_wave_grid, cont_flux_scaled)))
                assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

            elif timestep_coher[j] >= 1.:
                hires_choice.append('WO')
                for i in range(len(WC_spec_params_reform_powr)):
                    distance_to_spec.append(abs(timestep_teffs_final[j] - WC_spec_params_teff_powr[i]))
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = WC_spec_params_teff_powr[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(1.0)
                assigned_spectrum = WC_reformed_spec_grid_powr[nearest_spec_ind]
                assinged_radius = WC_spec_params_radius_powr[nearest_spec_ind]
                assigned_flux_scaled = assigned_spectrum[:,1] *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_flux_scaled)
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(WN_integrated_spectra_powr[nearest_spec_ind])
                spec_cont = planck(hires_wave_grid, near_spec_temp)
                cont_flux_scaled = spec_cont[:,1] * assigned_flux_resampled[4050] / spec_cont[:,1][4050]
                #assigned_cont.append(np.column_stack((hires_wave_grid, cont_flux_scaled)))
                assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))


        #elif initial_masses[j] < 5. or timestep_teffs_final[j] < 2000:
        elif timestep_teffs_final[j] < 2000:
            hires_choice.append('lowmass')
            for i in range(len(lowmass_params)):
                #distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2 + (lowmass_loggs[i] - timestep_loggs_final[j])**2)
                distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2)
            nearest_spec_ind = np.argmin(distance_to_spec)
            near_spec_temp = lowmass_teffs[nearest_spec_ind]
            assigned_spec_teff.append(near_spec_temp)
            assigned_spec_logg.append(1.0)
            assigned_spectrum = lowmass_spec[nearest_spec_ind]
            assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_spectrum[:,1])
            assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
            assigned_integrated_spectra.append(lowmass_int_spec[nearest_spec_ind])
            assigned_cont.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))

            # hires_choice.append('nope')
            # assigned_spec_teff.append(0.0)
            # assigned_spectra.append(np.column_stack((hires_wave_grid, empty_hires_flux)))
            # assigned_integrated_spectra.append(1e-30)
            # assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

        elif initial_masses[j] > 10. and timestep_teffs_final[j] < 22000:
            hires_choice.append('kurucz')
            for i in range(len(lowmass_params)):
                #distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2 + (lowmass_loggs[i] - timestep_loggs_final[j])**2)
                distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2)
            nearest_spec_ind = np.argmin(distance_to_spec)
            near_spec_temp = lowmass_teffs[nearest_spec_ind]
            assigned_spec_logg.append(1.0)
            assigned_spec_teff.append(near_spec_temp)
            assigned_spectrum = lowmass_spec[nearest_spec_ind]
            assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_spectrum[:,1])
            #assigned_spectra.append(assigned_spectrum)
            #assigned_integrated_spectra.append(lowmass_hires_int_spec[nearest_spec_ind])
            assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
            assigned_integrated_spectra.append(lowmass_int_spec[nearest_spec_ind])

            cont_flux_lowres_sample = np.interp(cont_wave_grid_calc, assigned_spectrum[:,0], assigned_spectrum[:,1])

            cont_flux_average = []
            for i in range(len(cont_flux_lowres_sample)):
                ind_flux_norm = np.abs(cont_flux_lowres_sample[i] - assigned_spectrum[:,1]).argmin()
                flux_norm_lower = assigned_spectrum[:,1][ind_flux_norm - 1]
                flux_norm_higher = assigned_spectrum[:,1][ind_flux_norm + 1]
                cont_flux_average_i = np.log10( (1.0e-30 + cont_flux_lowres_sample[i] + flux_norm_lower + flux_norm_higher ) / 3 )
                cont_flux_average.append(cont_flux_average_i)

            cont_flux_resampled = np.interp(hires_wave_grid, cont_wave_grid_calc, cont_flux_average)
            cont_flux_resampled_scaled = 10**cont_flux_resampled

            assigned_cont.append(np.column_stack((hires_wave_grid,cont_flux_resampled_scaled)))
            #assigned_cont.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
            #assigned_cont.append(lowmass_hires_cont[nearest_spec_ind])

            #hires_choice.append('nope')
            # assigned_spec_teff.append(0.0)
            # assigned_spectra.append(np.column_stack((hires_wave_grid, empty_hires_flux)))
            # assigned_integrated_spectra.append(1e-30)
            # assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

        elif initial_masses[j] < 5. or initial_masses[j] > 5. and timestep_teffs_final[j] < 17000:
            hires_choice.append('kurucz2')
            for i in range(len(lowmass_params)):
                #distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2 + (lowmass_loggs[i] - timestep_loggs_final[j])**2)
                distance_to_spec.append((lowmass_teffs[i] - 10**timestep_temps_final[j])**2)
            nearest_spec_ind = np.argmin(distance_to_spec)
            near_spec_temp = lowmass_teffs[nearest_spec_ind]
            assigned_spec_logg.append(1.0)
            assigned_spec_teff.append(near_spec_temp)
            assigned_spectrum = lowmass_spec[nearest_spec_ind]
            assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_spectrum[:,1])
            #assigned_spectra.append(assigned_spectrum)
            assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
            assigned_integrated_spectra.append(lowmass_int_spec[nearest_spec_ind])
            #assigned_integrated_spectra.append(lowmass_hires_int_spec[nearest_spec_ind])

            cont_flux_lowres_sample = np.interp(cont_wave_grid_calc, assigned_spectrum[:,0], assigned_spectrum[:,1])

            cont_flux_average = []
            for i in range(len(cont_flux_lowres_sample)):
                ind_flux_norm = np.abs(cont_flux_lowres_sample[i] - assigned_spectrum[:,1]).argmin()
                flux_norm_lower = assigned_spectrum[:,1][ind_flux_norm - 1]
                flux_norm_higher = assigned_spectrum[:,1][ind_flux_norm + 1]
                cont_flux_average_i = np.log10( (1.0e-30 + cont_flux_lowres_sample[i] + flux_norm_lower + flux_norm_higher ) / 3 )
                cont_flux_average.append(cont_flux_average_i)

            cont_flux_resampled = np.interp(hires_wave_grid, cont_wave_grid_calc, cont_flux_average)
            cont_flux_resampled_scaled = 10**cont_flux_resampled

            assigned_cont.append(np.column_stack((hires_wave_grid,cont_flux_resampled_scaled)))
            #assigned_cont.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
            #assigned_cont.append(lowmass_hires_cont[nearest_spec_ind])

            # hires_choice.append('nope')
            # assigned_spec_teff.append(0.0)
            # assigned_spectra.append(np.column_stack((hires_wave_grid, empty_hires_flux)))
            # assigned_integrated_spectra.append(1e-30)
            # assigned_cont.append(np.column_stack((hires_wave_grid, empty_hires_flux)))

        else:
            if spectra_library == 'FW':# and timestep_teffs_final[j] > 17600:
                hires_choice.append('OB - FW')
                for i in range(len(hires_teffs)):
                    #distance_to_spec.append(abs(hires_teffs[i] - 10**timestep_temps_final[j]) + abs(hires_loggs[i] - specsyn_loggs[j]))
                    distance_to_spec.append((timestep_temps_final[j] - np.log10(hires_teffs[i]))**2 + 5*(hires_loggs[i] - specsyn_loggs[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = hires_teffs[nearest_spec_ind]
                near_spec_logg = hires_loggs[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(near_spec_logg)
                assigned_spectrum = hires_spec[nearest_spec_ind]

                assigned_flux_resampled = np.interp(hires_wave_grid, assigned_spectrum[:,0], assigned_spectrum[:,1])
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))
                assigned_integrated_spectra.append(hires_int_spec[nearest_spec_ind])
                assigned_cont.append(np.column_stack((hires_wave_grid,assigned_flux_resampled)))

                #assigned_spectra.append(assigned_spectrum)
                #assigned_cont.append(hires_cont[nearest_spec_ind])
                #assigned_integrated_spectra.append(hires_int_spec[nearest_spec_ind])

            if spectra_library == 'WM':# and timestep_teffs_final[j] > 17600:
                hires_choice.append('OB - WM')
                for i in range(len(hires_teffs)):
                    #distance_to_spec.append(abs(hires_teffs[i] - 10**timestep_temps_final[j]) + abs(hires_loggs[i] - specsyn_loggs[j]))
                    distance_to_spec.append((timestep_temps_final[j] - np.log10(hires_teffs[i]))**2 + 5*(hires_loggs[i] - specsyn_loggs[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = hires_teffs[nearest_spec_ind]
                near_spec_logg = hires_loggs[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(near_spec_logg)
                assigned_spectrum = hires_spec[nearest_spec_ind]
                assigned_spectra.append(assigned_spectrum)
                assigned_cont.append(hires_cont[nearest_spec_ind])
                assigned_integrated_spectra.append(hires_int_spec[nearest_spec_ind])

            if spectra_library == 'PoWR':
                hires_choice.append('OB - PoWR')
                for i in range(len(hires_teffs)):
                    #distance_to_spec.append(abs(hires_teffs[i] - 10**timestep_temps_final[j]) + abs(hires_loggs[i] - specsyn_loggs[j]))
                    #distance_to_spec.append(abs(hires_teffs[i] - 10**timestep_temps_final[j]) + abs(hires_loggs[i] - specsyn_loggs[j]))
                    distance_to_spec.append((timestep_temps_final[j] - hires_temps[i])**2 + 5*(hires_loggs[i] - specsyn_loggs[j])**2)
                    #distance_to_spec.append((hires_teffs[i] - timestep_teffs_final[j])**2 + 5*(hires_loggs[i] - specsyn_loggs[j])**2)
                nearest_spec_ind = np.argmin(distance_to_spec)
                near_spec_temp = hires_teffs[nearest_spec_ind]
                near_spec_logg = hires_loggs[nearest_spec_ind]
                assigned_spec_teff.append(near_spec_temp)
                assigned_spec_logg.append(near_spec_logg)
                assigned_spectrum = hires_spec[nearest_spec_ind]
                assinged_radius = timestep_radii_final[j]
                assigned_flux_scaled = assigned_spectrum[:,1]# *(((3.0856e9)**2.0)/((assinged_radius*6.96e10)**2.0)) * 1.0e20
                assigned_flux_scaled = assigned_flux_scaled * timestep_teffs_final[j] / near_spec_temp
                assigned_spectra.append(np.column_stack((hires_wave_grid,assigned_flux_scaled)))
                assigned_cont.append(hires_cont[nearest_spec_ind])
                assigned_integrated_spectra.append(hires_int_spec[nearest_spec_ind])


    return(assigned_integrated_spectra,assigned_spec_teff,assigned_spec_logg,assigned_spectra,assigned_cont, hires_choice)

def specsyn(assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, radii, No_stars):
    assigned_spec_renormed = []
    assigned_flux_renormed = []
    template_wave = assigned_spectra[0][:,0]
    for i in range(len(assigned_integrated_spectra)):
        xinte =  assigned_integrated_spectra[i] / specsyn_bbfluxes[i]
        renormed_flux = assigned_spectra[i][:,1] / xinte
        renormed_spec = np.column_stack((template_wave,renormed_flux))
        assigned_spec_renormed.append(renormed_spec)
        assigned_flux_renormed.append(renormed_flux)

    test_assigned_flux = assigned_flux_renormed
    test_radii = radii

    assigned_flux_scaled_test = []
    assigned_flux_scaled = []
    for i in range(len(test_assigned_flux)):
        assigned_flux_scaled_i = 12.566 * test_radii[i] *  test_radii[i] /1e20 * test_assigned_flux[i] * No_stars[i]# * 3.14 * 4 * 5# /1e20
        #assigned_flux_scaled_testi = 12.566 * test_radii[i] *  test_radii[i] /1e20 * test_assigned_flux[i]# * 3.14 * 4 * 5# /1e20 #FOR HWO
        assigned_flux_scaled.append(assigned_flux_scaled_i)
        #assigned_flux_scaled_test.append(assigned_flux_scaled_testi)

    assigned_flux_scaled = np.array(assigned_flux_scaled)
    population_flux = assigned_flux_scaled.sum(axis=0)
    return(population_flux, assigned_flux_scaled_test)

def specsyn_hires(assigned_integrated_spectra, specsyn_bbfluxes, assigned_spectra, assigned_cont, radii, No_stars, population_nebular, assigned_hires_choice, hires_wave_grid, spectra_library):

    assigned_flux = []
    assigned_cont_flux = []

    population_nebular_resampled = np.interp(hires_wave_grid, population_nebular[:,0], population_nebular[:,1])

    for i in range(len(assigned_integrated_spectra)):
        xinte =  assigned_integrated_spectra[i] / specsyn_bbfluxes[i]

        assigned_f = assigned_spectra[i][:,1]

        #renormed_flux = assigned_f / xinte #turn on to have bb scaling in hires spectra
        renormed_flux = assigned_f

        assigned_c = assigned_cont[i][:,1]
        #assigned_flux.append(assigned_f)
        assigned_flux.append(renormed_flux)
        assigned_cont_flux.append(assigned_c)

    if spectra_library == 'PoWR':
        assigned_flux_scaled = []
        assigned_cont_scaled = []
        assigned_flux_norm = []
        for i in range(len(assigned_flux)):
            assigned_flux_scaled_i = 12.566 * radii[i]**2 /1e20 * assigned_flux[i] * No_stars[i]# * 3.14 * 4 * 5# /1e20
            assigned_flux_scaled.append(assigned_flux_scaled_i)

            if assigned_hires_choice[i] == 'OB - PoWR':
                assigned_norm_flux_powr = assigned_cont_flux[i]
                assigned_flux_norm.append(assigned_norm_flux_powr)
            elif assigned_hires_choice[i] != 'OB - PoWR':
                assigned_cont_scaled_i = 12.566 * radii[i]**2 /1e20 * assigned_cont_flux[i] * No_stars[i]# * 3.14 * 4 * 5# /1e20
                assigned_cont_scaled.append(assigned_cont_scaled_i)
                assigned_norm_flux_not_powr = assigned_flux_scaled_i / (assigned_cont_scaled_i+1.0e-35)
                #assigned_flux_norm.append(assigned_norm_flux_not_powr)

        for i in range(len(assigned_flux)):
            assigned_cont_scaled_i = assigned_cont_flux[i]# * No_stars[i]# * 3.14 * 4 * 5# /1e20
            assigned_cont_scaled.append(assigned_cont_scaled_i)

        assigned_flux_scaled = np.array(assigned_flux_scaled)
        assigned_cont_scaled = np.array(assigned_cont_scaled)
        assigned_flux_norm = np.array(assigned_flux_norm)
        population_flux = assigned_flux_scaled.sum(axis=0)
        population_cont = assigned_cont_scaled.sum(axis=0)
        population_flux_norm = assigned_flux_norm.mean(axis=0)

        population_flux_wneb = np.add(population_flux, population_nebular_resampled)


    else:
        assigned_flux_scaled = []
        assigned_cont_scaled = []
        for i in range(len(assigned_flux)):
            assigned_flux_scaled_i = 12.566 * radii[i]**2 /1e20 * assigned_flux[i] * No_stars[i]# * 3.14 * 4 * 5# /1e20
            assigned_cont_scaled_i = 12.566 * radii[i]**2 /1e20 * assigned_cont_flux[i] * No_stars[i]# * 3.14 * 4 * 5# /1e20
            assigned_flux_scaled.append(assigned_flux_scaled_i)
            assigned_cont_scaled.append(assigned_cont_scaled_i)

        assigned_flux_scaled = np.array(assigned_flux_scaled)
        assigned_cont_scaled = np.array(assigned_cont_scaled)
        population_flux = assigned_flux_scaled.sum(axis=0)
        population_cont = assigned_cont_scaled.sum(axis=0)
        population_flux_norm = population_flux / (population_cont+1.0e-35)

        population_flux_wneb = np.add(population_flux, population_nebular_resampled)

    return(population_flux, population_flux_wneb, population_flux_norm, population_nebular_resampled)

def compute_ion_flux(wave, freq, flux_wave, flux_freq, limit, code):
    ind_ION = [i for i in range(len(wave)) if wave[i] <= limit]
    wave_ION = wave[ind_ION]
    freq_ION = freq[ind_ION]
    flux_wave_ION = flux_wave[ind_ION]
    flux_freq_ION = flux_freq[ind_ION]
    integral_photons_ION = np.trapezoid(flux_freq_ION, freq_ION)
    integral_flux_ION = np.trapezoid(flux_wave_ION, wave_ION)

    int_ION = []
    for i in range(len(wave_ION)):
        ION_comp = flux_wave_ION[i] / (6.626e-27 * wave_ION[i] )
        int_ION.append(ION_comp)
    qi = np.sum(int_ION)
    Qi = qi * 4 * np.pi *18**2
    Qilog = np.log10(Qi)

    if code == 1:
        No_photons_ION = np.log10(integral_photons_ION + 1.0e-30)+20.
        ion_flux_ION = np.log10(-1.0*integral_flux_ION + 1.0e-30)+20.
    if code == 2:
        No_photons_ION = np.log10(-1.0*integral_photons_ION + 1.0e-30)+20.
        ion_flux_ION = np.log10(integral_flux_ION + 1.0e-30)+20.
    return(No_photons_ION, ion_flux_ION, Qilog)

def ionise(wave, pop_flux, code):
    spectrum_flux = np.array(pop_flux)

    spectrum_freq = 2.997925e18/wave
    spectrum_flux_conv1 = 3.33564e-19*spectrum_flux*wave*wave
    spectrum_flux_conv = spectrum_flux_conv1/spectrum_freq/6.6262e-27

    full_lum = np.trapezoid(np.flip(spectrum_flux), np.flip(wave))
    #1=wmbasic or fastwind, 2=cmfgen
    if code == 1:
        bolo_lum = np.log10(full_lum + 1.0e-30)+20.
    if code == 2:
        bolo_lum = np.log10(-1.0*full_lum + 1.0e-30)+20.
    flux_lim_HI = 912.
    flux_lim_HeI = 504.
    flux_lim_HeII = 228.

    ionising_flux_HI = compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HI, code)
    ionising_flux_HeI = compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HeI, code)
    ionising_flux_HeII = compute_ion_flux(wave, spectrum_freq, spectrum_flux, spectrum_flux_conv, flux_lim_HeII, code)
    return(bolo_lum, ionising_flux_HI, ionising_flux_HeI, ionising_flux_HeII)

def continuum(pop_ionising_flux):

    xrange = [10.,912.,913.,1300.,1500.,1800.,2200.,
            2855.,3331.,3421.,3422.,3642.,3648.,5700.,7000.,8207.,
            8209.,14583.,14585.,22787.,22789.,32813.,32815.,
            44680.,44682.,2000000.]
    xrange = np.array(xrange)
    gamma = [0.,0.,2.11e-4,5.647,9.35,9.847,10.582,16.101,24.681,
            26.736,24.883,29.979,6.519,8.773,11.545,13.585,6.333,
            10.444,7.023,9.361,7.59,9.35,8.32,9.53,8.87,0.]
    gamma = np.array(gamma)

    continuum = 2.998e18 / xrange / xrange / 2.60e-13 * 1.0e-30* gamma * 10**(pop_ionising_flux-30.)

    return (np.column_stack((xrange, continuum)))

def calc_wind(timestep_temps, timestep_lums, timestep_masses, timestep_mdot, timestep_H, timestep_12C, timestep_14N, timestep_16O, initial_masses, No_stars, timestep_radii, timestep_cnr, minimum_wr_mass, Z):

    timestep_mdot = np.array(timestep_mdot)
    timestep_masses = np.array(timestep_masses)
    timestep_radii = np.array(timestep_radii)
    timestep_lums = np.array(timestep_lums)
    timestep_H = np.array(timestep_H)
    timestep_12C = np.array(timestep_12C)
    timestep_14N = np.array(timestep_14N)
    timestep_16O = np.array(timestep_16O)
    timestep_coher=((timestep_12C/12.)+(timestep_16O/16.))/(timestep_H/4.) #WRONG NEEDS TO BE He INSTEAD OF H ABUNDANCE
    vinfs = []

    OB_limit_mass=8.
    timestep_teffs = 10**timestep_temps
    timestep_vesc = ((2*6.67*10**-11*(timestep_masses*2*10**30))/(timestep_radii*7*10**8))**0.5 / (1.*10**3)

    vinf_vink21 = []
    mdot_vink00 = []
    mdot_leuven = []
    vinf_xshootu = []

    for i in range(len(timestep_temps)):

        if timestep_lums[i] == -20.0:
            vinf=0.
            vinf_vink21i=0.
            mdot_vink00i=timestep_mdot[i]
            mdot_leuveni=timestep_mdot[i]
            vinf_xshootui=0.

        elif timestep_temps[i] < 4.4 and timestep_temps[i] > 3.75 and timestep_mdot[i] > -3.5: #LBVs
            vinf=200.
            vinf_vink21i=200.
            mdot_vink00i=timestep_mdot[i]
            mdot_leuveni=timestep_mdot[i]
            vinf_xshootui=200.


        elif timestep_temps[i] < 3.9:# and initial_masses[i] < OB_limit_mass: #low mass stars
            vinf = 30.
            vinf_vink21i = 30.
            mdot_vink00i=timestep_mdot[i]
            mdot_leuveni=timestep_mdot[i]
            vinf_xshootui=30.

#            vinfs.append(vinf)
#            vinfs.append(vinf)

        elif timestep_temps[i] > 4.4 and timestep_H[i] < 0.4 and initial_masses[i] > minimum_wr_mass: #WR stars
            if timestep_H[i] > 0.1:
                vinf=1650.
                vinf_vink21i=1650.
                mdot_vink00i=timestep_mdot[i]
                mdot_leuveni=timestep_mdot[i]
                vinf_xshootui=1650.
#                vinfs.append(vinf)
            if timestep_cnr[i] < 10.:
                vinf=1900.
                vinf_vink21i=1900.
                mdot_vink00i=timestep_mdot[i]
                mdot_leuveni=timestep_mdot[i]
                vinf_xshootui=1900.
#                vinfs.append(vinf)
            if timestep_coher[i] < 0.5:
                vinf=1800.
                vinf_vink21i=1800.
                mdot_vink00i=timestep_mdot[i]
                mdot_leuveni=timestep_mdot[i]
                vinf_xshootui=1800.
#                vinfs.append(vinf)
            if timestep_coher[i] < 1.:
                vinf=2800.
                vinf_vink21i=2800.
                mdot_vink00i=timestep_mdot[i]
                mdot_leuveni=timestep_mdot[i]
                vinf_xshootui=2800.
#                vinfs.append(vinf)
            if timestep_coher[i] >= 1.:
                vinf=3500.
                vinf_vink21i=3500.
                mdot_vink00i=timestep_mdot[i]
                mdot_leuveni=timestep_mdot[i]
                vinf_xshootui=3500.
#                vinfs.append(vinf)

        elif timestep_temps[i] > 3.9:# and timestep_H[i] > 0.4 and initial_masses[i] > OB_limit_mass: #OB stars
            gamma0= 1. - (2.7e-5 * (10.**(timestep_lums[i])) / timestep_masses[i])
            if gamma0 <= 0.:
                gamma0=1e-10
            vinf=618.*np.sqrt(timestep_masses[i] /  (10.**(0.5*timestep_lums[i] - 2.*timestep_temps[i] + 7.52))*gamma0) * (0.58 + 2.04*(0.5*timestep_lums[i] - 2.*timestep_temps[i] + 7.52))
            #timestep_mdot[i] = np.log10((10**timestep_mdot[i])/3)
            timestep_mdot[i] = np.log10((10**timestep_mdot[i]))
            mdot_leuveni = -5.52 + 2.39*np.log10((10**timestep_lums[i])/(10**6)) -1.48*np.log10(timestep_masses[i]/45.) + 2.12*np.log10(timestep_teffs[i]/45000.) + ( 0.75 - 1.87*np.log10(timestep_teffs[i]/40000.) )*np.log10(1./1.)
            vinf_xshootui = 0.092*timestep_teffs[i] - 1040.0*1.0**0.22

            if timestep_teffs[i] < 25883.:
                log_vinf_vink21i = -7.79 - 0.07*timestep_lums[i] +2.57*timestep_temps[i]# + 0.003*np.log10(1/1)
                vinf_vink21i = 10**log_vinf_vink21i
                mdot_vink00i = -6.688 + 2.210*np.log10(((10**timestep_lums[i])/(1.*10**5))) - 1.339*np.log10(timestep_masses[i]/30.) - 1.601*np.log10((vinf_vink21i/timestep_vesc[i])/2.) + 1.07*np.log10(timestep_teffs[i]/20000.)# + 0.85*np.log10(1./1.)

            else:
                log_vinf_vink21i = 0.39 - 0.04*timestep_lums[i] +0.74*timestep_temps[i]# + 0.19*np.log10(1/1)
                vinf_vink21i = 10**log_vinf_vink21i
                mdot_vink00i = -6.697 + 2.194*np.log10(((10**timestep_lums[i])/(1.*10**5))) - 1.313*np.log10(timestep_masses[i]/30.) - 1.226*np.log10((vinf_vink21i/timestep_vesc[i])/2.) + 0.933*np.log10(timestep_teffs[i]/40000.) - 10.92*(np.log10(timestep_teffs[i]/40000.))**2# + 0.85*np.log10(1./1.)

            #vinf=618.*( (timestep_masses[i] / timestep_radii[i]) * gamma0 )**0.5 * gamma0 * (0.58 + 2.04*(0.5*timestep_lums[i] - 2.*timestep_temps[i] + 7.52))
#            vinfs.append(vinf)

        else:
            vinf=0.
            vinf_vink21i=0.
            mdot_vink00i=timestep_mdot[i]
            mdot_leuveni=timestep_mdot[i]
            vinf_xshootui=0.

        vinf_vink21.append(vinf_vink21i)
        mdot_vink00.append(mdot_vink00i)
        mdot_leuven.append(mdot_leuveni)

        vinfs.append(vinf)
        vinf_xshootu.append(vinf_xshootui)

    if Z == 'MWC':
        Z_value = 1.4285
    if Z == 'MW':
        Z_value = 1.0
    if Z == 'LMC':
        Z_value = 0.4285
    if Z == 'SMC':
        Z_value = 0.1428
    if Z == 'IZw18':
        Z_value = 0.0285
    if Z == 'XMP':
        Z_value = 0.00071428
    if Z == 'Z0':
        Z_value = 1e-5

    #vinfs_Z = np.array(vinfs)* (Z_value**0.22)
    vinfs_Z = np.array(vinfs)* (Z_value**0.13)
    vinf_xshootu = np.array(vinf_xshootu) * (Z_value**0.22)
    #vinfs_Z = np.array(vinfs)* (Z**0.13)

    timestep_mdot_weakwind = []
    for i in range(len(timestep_mdot)):
        #elif timestep_temps_final[j] < 3.65 and initial_masses[j] > minimum_wr_mass
        if initial_masses[i] > 14.5 and timestep_lums[i] < 5.2:
            timestep_mdot_weakwind.append(timestep_mdot[i])
            #print(No_stars[i])
        else:
            timestep_mdot_weakwind.append(timestep_mdot[i])

    vinf_vink21 = np.array(vinf_vink21)
    mdot_vink00 = np.array(mdot_vink00)
    mdot_leuven = np.array(mdot_leuven)

    wind_power = (10.**timestep_mdot) * vinfs_Z**2 * 3.155
    wind_mom = 2 * (10.**timestep_mdot) * vinfs_Z * 3.155e-5
    wind_mdot = 10.**timestep_mdot
    wind_vinf = vinfs_Z
    wind_mom_vink = 2 * (10.**mdot_vink00) * vinf_vink21 * 3.155e-5
    wind_mom_leuven = 2 * (10.**mdot_leuven) * vinfs_Z * 3.155e-5
    wind_mom_xshootu = 2 * (10.**np.array(timestep_mdot_weakwind)) * vinf_xshootu * 3.155e-5
    wind_power_xshootu = (10.**np.array(timestep_mdot)) * vinf_xshootu**2 * 3.155

    wind_power_calc =  wind_power * No_stars
    wind_mom_calc =  wind_mom * No_stars
    wind_mom_vink_calc = wind_mom_vink * No_stars
    wind_mom_leuven_calc = wind_mom_leuven * No_stars
    wind_mom_xshootu_calc = wind_mom_xshootu * No_stars
    wind_power_xshootu_calc = wind_power_xshootu * No_stars

    wind_power_sum = np.sum(wind_power_calc)
    wind_mom_sum = np.sum(wind_mom_calc)
    wind_mom_vink_sum = np.sum(wind_mom_vink_calc)
    wind_mom_leuven_sum = np.sum(wind_mom_leuven_calc)
    wind_mom_xshootu_sum = np.sum(wind_mom_xshootu_calc)
    wind_power_xshootu_sum = np.sum(wind_power_xshootu_calc)

    wpower = np.log10(wind_power_sum +1e-30) + 35.
    wmom = np.log10(wind_mom_sum +1e-30) + 35.
    wmom_vink = np.log10(wind_mom_vink_sum +1e-30) + 35.
    wmom_leuven = np.log10(wind_mom_leuven_sum +1e-30) + 35.
    wmom_xshootu = np.log10(wind_mom_xshootu_sum +1e-30) + 35.
    wpower_xshootu = np.log10(wind_power_xshootu_sum +1e-30) + 35.


    return(vinfs_Z, wind_power_sum, wpower, wind_mom_sum, wmom, wmom_vink, vinf_vink21, mdot_vink00, timestep_vesc, wmom_leuven, wmom_xshootu, wpower_xshootu)

def get_uv_slope(uv_wave, uv_flux):
    '''Input wave and flux in log'''

    #uv_wave = 10**uv_wave
    #uv_flux = 10**uv_flux

    uvslope_bounds_1 = [3.103119254, 3.108565024]
    uvslope_bounds_2 = [3.116939647, 3.119255889]
    uvslope_bounds_3 = [3.127752516, 3.137037455]
    uvslope_bounds_4 = [3.156851901, 3.174931594]
    uvslope_bounds_5 = [3.19368103, 3.199480915]
    uvslope_bounds_6 = [3.224533063, 3.240549248]



    # uvslope_bounds_1 = [1268.000001, 1284.000001]
    # uvslope_bounds_2 = [1309.000001, 1315.999999]
    # uvslope_bounds_3 = [1342.000001, 1371.000001]
    # uvslope_bounds_4 = [1435.0, 1496.000002]
    # uvslope_bounds_5 = [1562.000002, 1583.000001]
    # uvslope_bounds_6 = [1677.000002, 1739.999999]
    ind_uvslope_1 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_1[0] and uv_wave[i] <= uvslope_bounds_1[1]]
    ind_uvslope_2 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_2[0] and uv_wave[i] <= uvslope_bounds_2[1]]
    ind_uvslope_3 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_3[0] and uv_wave[i] <= uvslope_bounds_3[1]]
    ind_uvslope_4 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_4[0] and uv_wave[i] <= uvslope_bounds_4[1]]
    ind_uvslope_5 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_5[0] and uv_wave[i] <= uvslope_bounds_5[1]]
    ind_uvslope_6 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_6[0] and uv_wave[i] <= uvslope_bounds_6[1]]

    ind_uvslope_full = np.concatenate((ind_uvslope_1, ind_uvslope_2, ind_uvslope_3, ind_uvslope_4, ind_uvslope_5, ind_uvslope_6))

    #Leitherer+1999 mode
    # uvslope_bounds_1 = [np.log10(1280.000001), np.log10(1320.000001)]
    # uvslope_bounds_2 = [np.log10(1780.000001), np.log10(1820.999999)]
    # ind_uvslope_1 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_1[0] and uv_wave[i] <= uvslope_bounds_1[1]]
    # ind_uvslope_2 = [i for i in range(len(uv_wave)) if uv_wave[i] >= uvslope_bounds_2[0] and uv_wave[i] <= uvslope_bounds_2[1]]
    # ind_uvslope_full = np.concatenate((ind_uvslope_1, ind_uvslope_2))


    wave_uv_slope = uv_wave[ind_uvslope_full]
    flux_uv_slope = uv_flux[ind_uvslope_full]

    popt_uvslope, pcov_uvslope = curve_fit(f, wave_uv_slope, flux_uv_slope)
    x_uvslope = np.linspace(3.05, 3.3, 1000)
    #x_uvslope = np.linspace(1122.0, 1995.0, 1000)
    y_uvslope = popt_uvslope[0]*x_uvslope + popt_uvslope[1]
    beta_uvslope = popt_uvslope[0]

    return(x_uvslope, y_uvslope, beta_uvslope)

def get_ew(uv_wave, uv_flux, HI_ionflux):
    '''Input wave, flux and ionising flux'''

    #Ha_continuum_flux_fn = interp1d(uv_wave, uv_flux, kind='linear', fill_value='extrapolate')
    #Ha_continuum_flux = Ha_continuum_flux_fn(6563.)

    Ha_continuum_flux = np.interp(6563., uv_wave, uv_flux)
    Hb_continuum_flux = np.interp(4861., uv_wave, uv_flux)
    Pb_continuum_flux = np.interp(12818., uv_wave, uv_flux)
    Bg_continuum_flux = np.interp(21655., uv_wave, uv_flux)

    Ha_luminosity = 1.36e-12 * 10**(HI_ionflux-30.)
    Hb_luminosity = 4.76e-13 * 10**(HI_ionflux-30.)
    Pb_luminosity = 7.73e-14 * 10**(HI_ionflux-30.)
    Bg_luminosity = 1.31e-14 * 10**(HI_ionflux-30.)

    Ha_ew = Ha_luminosity / Ha_continuum_flux * 1.0e10
    #Hb_ew = Ha_luminosity / Ha_continuum_flux * 1.0e10
    Hb_ew = Hb_luminosity / Hb_continuum_flux * 1.0e10
    Pb_ew = Pb_luminosity / Pb_continuum_flux * 1.0e10
    Bg_ew = Bg_luminosity / Bg_continuum_flux * 1.0e10

    Ha_ew_unit = np.log10(Ha_ew + 1.0e-35)
    Hb_ew_unit = np.log10(Hb_ew + 1.0e-35)
    Pb_ew_unit = np.log10(Pb_ew + 1.0e-35)
    Bg_ew_unit = np.log10(Bg_ew + 1.0e-35)

    return(Ha_ew_unit, Hb_ew_unit, Pb_ew_unit, Bg_ew_unit, Ha_luminosity, Ha_continuum_flux)

def colours(population_flux, wave_grid):

    V_wave = [50.,4750.,4800.,4850.,4900.,4950.,5000.,5050.,5100.,5150.,5200.,5250.,5300.,5350.,5400.,5450.,5500.,5550.,5600.,5650.,5700.,5750.,5800.,5850.,5900.,5950.,6000.,6050.,6100.,6150.,6200.,6250.,6300.,6350.,6400.,6450.,6500.,6550.,6600.,6650.,6700.,6750.,6800.,6850.,6900.,6950.,7000.,7050.,7100.,7150.,7200.,7250.,7300.,7350.,7400.,2000000.]
    V_profile = [0.000,0.000,0.030,0.084,0.163,0.301,0.458,0.630,0.780,0.895,0.967,0.997,1.000,0.988,0.958,0.919,0.877,0.819,0.765,0.711,0.657,0.602,0.545,0.488,0.434,0.386,0.331,0.289,0.250,0.214,0.181,0.151,0.120,0.093,0.069,0.051,0.036,0.027,0.021,0.018,0.016,0.014,0.012,0.011,0.010,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001,0.000,0.000]

    U_wave = [50.,3050.,3100.,3150.,3200.,3250.,3300.,3350.,3400.,3450.,3500.,3550.,3600.,3650.,3700.,3750.,3800.,3850.,3900.,3950.,4000.,4050.,4100.,4150.,4200.,2000000.]
    U_profile = [0.000,0.000,0.020,0.077,0.135,0.204,0.282,0.385,0.493,0.600,0.705,0.820,0.900,0.959,0.993,1.000,0.975,0.850,0.645,0.400,0.223,0.125,0.057,0.005,0.000,0.000]

    I_wave = [50.,7000.,7100.,7200.,7300.,7400.,7500.,7600.,7700.,7800.,7900.,8000.,8100.,8200.,8300.,8400.,8500.,8600.,8700.,8800.,8900.,9000.,9100.,9200.,2000000.]
    I_profile = [0.00000,0.00000,0.02400,0.23200,0.55500,0.78500,0.91000,0.96500,0.98500,0.99000,0.99500,1.00000,1.00000,0.99000,0.98000,0.95000,0.91000,0.86000,0.75000,0.56000,0.33000,0.15000,0.03000,0.00000,0.00000]

    B_wave = [50.,3600.,3650.,3700.,3750.,3800.,3850.,3900.,3950.,4000.,4050.,4100.,4150.,4200.,4250.,4300.,4350.,4400.,4450.,4500.,4550.,4600.,4650.,4700.,4750.,4800.,4850.,4900.,4950.,5000.,5050.,5100.,5150.,5200.,5250.,5300.,5350.,5400.,5450.,5500.,5550.,2000000.]
    B_profile = [0.000,0.000,0.006,0.030,0.060,0.134,0.302,0.567,0.841,0.959,0.983,0.996,1.000,0.996,0.987,0.974,0.957,0.931,0.897,0.849,0.800,0.748,0.698,0.648,0.597,0.545,0.497,0.447,0.397,0.345,0.297,0.252,0.207,0.166,0.129,0.095,0.069,0.043,0.024,0.009,0.000,0.000]

    V_interp = np.interp(wave_grid, V_wave, V_profile)
    U_interp = np.interp(wave_grid, U_wave, U_profile)
    I_interp = np.interp(wave_grid, I_wave, I_profile)
    B_interp = np.interp(wave_grid, B_wave, B_profile)

    V_data = population_flux * V_interp
    U_data = population_flux * U_interp
    I_data = population_flux * I_interp
    B_data = population_flux * B_interp

    V_value = np.trapezoid(wave_grid, V_data)
    U_value = np.trapezoid(wave_grid, U_data)
    I_value = np.trapezoid(wave_grid, I_data)
    B_value = np.trapezoid(wave_grid, B_data)

    V_mag = ( -2.5 * np.log10(-1.0*V_value) )
    U_mag = ( -2.5 * np.log10(-1.0*U_value) )
    I_mag = ( -2.5 * np.log10(-1.0*I_value) )
    B_mag = ( -2.5 * np.log10(-1.0*B_value) )

    absV_mag = ( -2.5 * np.log10(-1.0*V_value) ) + 36.552


    #UB_mag = U_mag - B_mag - 1.102

    return(V_mag, U_mag, I_mag, B_mag, absV_mag)

def get_SN_rate(population_SN_ind, No_stars):
    population_new_SN_mass_inds = []
    for i in range(len(population_SN_ind)): #population_SN_ind gives the list of all indices for masses that have gone SN so far
        if i == len(population_SN_ind)-1:
            population_new_SN_mass_inds.append([])
            break
        pop_SN_ind_test = population_SN_ind[i+1][:-len(population_SN_ind[i])] #this just finds all the mass indices that have gone supernova in the latest timestep
        population_new_SN_mass_inds.append(pop_SN_ind_test)

    #This just counts the number of timesteps which have passed since the last set of SN
    ind_SN_times = []
    counter = 0
    for i in range(len(population_new_SN_mass_inds)):
        if len(population_new_SN_mass_inds[i]) < 1:
            ind_SN_times.append(1)
            counter = counter + 1
        else:
            if counter == 0:
                ind_SN_times.append(1)
            else:
                ind_SN_times.append(counter)
            counter = 0

    #For each mass index that goes SN in a given timestep, divide the total number of stars by the time since the last set of SN to get the rate
    population_SN_rate = []
    population_SN_rate_csf = []
    for i in range(len(population_new_SN_mass_inds)):
        pop_SN_calc = population_new_SN_mass_inds[i]

        SN_rate_step = []
        SN_rate_step_csf = []
        for j in pop_SN_calc:
            SN_rate_mass = No_stars[j] / ( 1e5 * (ind_SN_times[i] + 1))
            SN_rate_mass_csf = No_stars[j] / ( 1e5 )


            SN_rate_step.append(SN_rate_mass)
            SN_rate_step_csf.append(SN_rate_mass_csf)
        SN_rate_step_sum = np.sum(SN_rate_step)
        SN_rate_step_sum_csf = np.sum(SN_rate_step_csf)
        population_SN_rate.append(SN_rate_step_sum)
        population_SN_rate_csf.append(SN_rate_step_sum_csf)

    population_SN_rate = np.array(population_SN_rate) + 1.0e-30
    population_SN_rate_log = np.log10(population_SN_rate)
    population_SN_rate_csf = np.array(population_SN_rate_csf) + 1.0e-30
    population_SN_rate_log_csf = np.log10(population_SN_rate_csf)

    return(population_SN_rate_log, population_SN_rate_log_csf)

def compute_radii(temps, lums):
#    4 pi R^2 sigma T^4
    sigma = 5.670 *10**(-8)
    radii = []
    lums_sol = 10**np.array(lums) * 3.828*10**26
    teffs = 10**temps

    for i in range(len(teffs)):
        if temps[i] == 0.0:
            R=0.1
        else:
            R = (lums_sol[i] / ( 4 * np.pi * sigma * teffs[i]**4 ))**0.5
            R = R / (7*10**8)
        radii.append(R)

    return(radii)

def planck(wave, teff):

    pi = 3.1415927
    h = 6.626196e-27
    c = 2.997925e10
    boltz = 1.380622e-16
    cgswave = 1.0e-8 * wave

    c1 = h * c / boltz / teff / cgswave

    for i in range(len(c1)):
        if c1[i] > 80:
            c1[i] = 1.0e-30


    else:
        flux = 2* pi * h * c*c / cgswave**5 / (np.exp(c1)-1)*1e-8

    planck_spectrum = np.column_stack((wave,flux))

    return(planck_spectrum)

def f(x, A, B):
    return A*x + B

def interpolate_tracks(z_model, inp):

    # Copy so the age correction below does not modify the loaded tracks
    evo_tracks = np.array(z_model.evo_tracks, copy=True)

    #split tracks into individual sub-arrays
    split_factor = len(z_model.mass_grid)
    track_ages = np.array_split(evo_tracks[:,1], split_factor)
    track_masses = np.array_split(evo_tracks[:,2], split_factor)
    track_lums = np.array_split(evo_tracks[:,3], split_factor)
    track_temps = np.array_split(evo_tracks[:,4], split_factor)
    track_H_abundances = np.array_split(evo_tracks[:,5], split_factor)
    track_He_abundances = np.array_split(evo_tracks[:,6], split_factor)
    track_12C_abundances = np.array_split(evo_tracks[:,7], split_factor)
    track_14N_abundances = np.array_split(evo_tracks[:,8], split_factor)
    track_16O_abundances = np.array_split(evo_tracks[:,9], split_factor)
    track_core_temps = np.array_split(evo_tracks[:,10], split_factor)
    track_mass_loss_rates = np.array_split(evo_tracks[:,11], split_factor)

    #set initial age for all tracks to 0.0
    for i in track_ages:
        i[0] = 1.0e-3

    for i in range(len(track_temps)):
        track_ages[i] = np.append(track_ages[i], np.array([track_ages[i][-1] + 100., (track_ages[i][-1] + 100.) * 10**6]))
        track_lums[i] = np.append(track_lums[i], np.array([-20., -20.]))
        track_temps[i] = np.append(track_temps[i], np.array([track_temps[i][-1], track_temps[i][-1]]))
        track_masses[i] = np.append(track_masses[i], np.array([track_masses[i][-1], track_masses[i][-1]]))
        track_H_abundances[i] = np.append(track_H_abundances[i], np.array([track_H_abundances[i][-1], track_H_abundances[i][-1]]))
        track_He_abundances[i] = np.append(track_He_abundances[i], np.array([track_He_abundances[i][-1], track_He_abundances[i][-1]]))
        track_12C_abundances[i] = np.append(track_12C_abundances[i], np.array([track_12C_abundances[i][-1], track_12C_abundances[i][-1]]))
        track_14N_abundances[i] = np.append(track_14N_abundances[i], np.array([track_14N_abundances[i][-1], track_14N_abundances[i][-1]]))
        track_16O_abundances[i] = np.append(track_16O_abundances[i], np.array([track_16O_abundances[i][-1], track_16O_abundances[i][-1]]))
        track_core_temps[i] = np.append(track_core_temps[i], np.array([track_core_temps[i][-1], track_core_temps[i][-1]]))
        track_mass_loss_rates[i] = np.append(track_mass_loss_rates[i], np.array([track_mass_loss_rates[i][-1], track_mass_loss_rates[i][-1]]))

    for i in range(len(track_mass_loss_rates)):
        for j in range(len(track_mass_loss_rates[0])):
            if track_mass_loss_rates[i][j] > -1:
                track_mass_loss_rates[i][j] = -1000

    # Interpolate each parameter in mass between adjacent tracks
    track_params = dict(ages=track_ages, lums=track_lums, temps=track_temps, H_abundances=track_H_abundances,
                        He_abundances=track_He_abundances, **{'12C_abundances': track_12C_abundances,
                                                              '14N_abundances': track_14N_abundances,
                                                              '16O_abundances': track_16O_abundances},
                        core_temps=track_core_temps, mass_loss_rates=track_mass_loss_rates)
    grid_params_adjinterp_total = {}
    for name, track_param in track_params.items():
        grid_masses_adjinterp_total, grid_params_adjinterp_total[name] = interpolate_param(track_param, track_masses,
                                                                                           inp.run_speed_mode)

    # One array per interpolated track
    grids = SimpleNamespace(grid_masses=rearrange_grid_array(grid_masses_adjinterp_total),
                            **{f'grid_{name}': rearrange_grid_array(grid_param)
                               for name, grid_param in grid_params_adjinterp_total.items()})

    return grids

def interp_time(values, ages, times_steps):

    """
    Linearly interpolate an array at the input ages from the internal time grid.

    The interpolation is done along the first axis of ``values``, as stored. The SEDs and spectra of the outputs are
    stored as log10 of the flux, so they are interpolated in log. The values at the internal time steps are returned
    unchanged, including the ``-inf`` of null fluxes (which would otherwise become ``nan`` in the weighted sum).

    Parameters
    ----------
    values : array_like
        Array with one entry per internal time step along the first axis.
    ages : float or array_like
        Ages in yr where the array is interpolated. They must be within ``times_steps``.
    times_steps : numpy.ndarray
        Internal time grid of the population synthesis, in yr (increasing).

    Returns
    -------
    numpy.ndarray
        Array with one entry per input age along the first axis and the remaining axes of ``values``. A scalar age
        returns one entry.

    Raises
    ------
    ValueError
        If any age is outside the internal time grid.

    Examples
    --------
    Interpolate the stellar SED at the output spectra ages:

    >>> sed = interp_time(res.SED_stellar, inp.times_spectra, inp.times_steps)
    """

    t = times_steps
    ages = np.atleast_1d(np.asarray(ages, dtype=float))
    outside = (ages < t[0]) | (ages > t[-1])
    if outside.any():
        raise ValueError(f'The ages {ages[outside] / 1e6} Myr are outside the pyStarburst99 time grid '
                         f'[{t[0] / 1e6}, {t[-1] / 1e6}] Myr and cannot be interpolated')

    values = np.asarray(values, dtype=float)
    idx = np.clip(np.searchsorted(t, ages, side='right') - 1, 0, len(t) - 2)
    w = ((ages - t[idx]) / (t[idx + 1] - t[idx])).reshape(-1, *[1] * (values.ndim - 1))
    lo, hi = values[idx], values[idx + 1]
    with np.errstate(invalid='ignore'):
        return np.where(w == 0, lo, np.where(w == 1, hi, (1 - w) * lo + w * hi))

def prepare_outputs(pop, spec_lib, star_formation_option):
    """
    Output arrays of the population for the star formation model.

    For an instantaneous burst (``"ISF"``) the population properties at each time step are the outputs. For continuous
    star formation (``"CSF"``) each time step adds a new burst, so the outputs are the cumulative sums over the time
    steps, and the SEDs, spectra, fluxes, winds and supernova rate are converted back to log10.

    Parameters
    ----------
    pop : types.SimpleNamespace
        Population properties from ``Compiler.population_synthesis``.
    spec_lib : types.SimpleNamespace
        Spectral libraries from ``Compiler.spectral_grids``. Its ``wave_grid`` is used for the UV slope, equivalent
        widths and colours of the ``"CSF"`` model.
    star_formation_option : {"ISF", "CSF"}
        Instantaneous burst or continuous star formation.

    Returns
    -------
    types.SimpleNamespace
        Output arrays with one entry per internal time step along the first axis: ``SED_stellar``, ``SED_total``,
        ``hires_flux``, ``hires_flux_wneb``, ``hires_flux_norm``, ``Lbol``, ``HI_flux``, ``HeI_flux``, ``HeII_flux``,
        ``wind_power``, ``wind_momentum``, ``uv_slope``, ``Ha_ew``, ``Hb_ew``, ``Pb_ew``, ``Bg_ew``, ``colours``
        (V, U, I, B and M_V) and ``SN_rate``.

    Notes
    -----
    The SEDs, high-resolution fluxes (except ``hires_flux_norm``), ionising fluxes and winds are log10 values, as
    saved by pySB99.py.
    """

    if star_formation_option == 'ISF':
        pop_SED_stellar = np.array(pop.population_flux_iterations_send)
        pop_SED_total = np.array(pop.population_flux_total_iterations_send)

        pop_hires_flux = np.array(pop.population_hires_flux_iterations_send)
        pop_hires_flux_wneb = np.array(pop.population_hires_flux_wneb_iterations_send)
        pop_hires_flux_norm = np.array(pop.population_hires_flux_norm_iterations)

        pop_Lbol = np.array(pop.population_ion_L_flux_iterations)
        pop_HI_flux = np.array(pop.population_ion_HI_flux_iterations)
        pop_HeI_flux = np.array(pop.population_ion_HEI_flux_iterations)
        pop_HeII_flux = np.array(pop.population_ion_HEII_flux_iterations)

        pop_windpower = np.array(pop.population_windpowers)
        pop_windmom = np.array(pop.population_windmoms)

        pop_uv_slope = np.array(pop.population_uv_slopes_beta)

        pop_Ha_ew = np.array(pop.population_Ha_ew)
        pop_Hb_ew = np.array(pop.population_Hb_ew)
        pop_Pb_ew = np.array(pop.population_Pb_ew)
        pop_Bg_ew = np.array(pop.population_Bg_ew)

        pop_colours = np.column_stack((pop.population_Vmag, pop.population_Umag, pop.population_Imag,
                                       pop.population_Bmag, pop.population_absVmag))

        pop_SN_rate = pop.SN_rates

    else:  # CSF
        csf_population_flux = np.cumsum(pop.population_flux_iterations, axis=0)
        csf_population_flux_total = np.cumsum(pop.population_flux_total_iterations, axis=0)
        pop_SED_stellar = np.log10(csf_population_flux) + 20.
        pop_SED_total = np.log10(csf_population_flux_total) + 20.

        pop_hires_flux = np.log10(np.cumsum(pop.population_hires_flux_iterations, axis=0)) + 20.
        pop_hires_flux_wneb = np.log10(np.cumsum(pop.population_hires_flux_wneb_iterations, axis=0)) + 20.
        pop_hires_flux_norm = np.cumsum(pop.population_hires_flux_norm_iterations, axis=0)

        pop_Lbol = np.log10(np.cumsum(10**(np.array(pop.population_ion_L_flux_iterations) - 20.))) + 20.
        pop_HI_flux = np.log10(np.cumsum(10**(np.array(pop.population_ion_HI_flux_iterations) - 20.))) + 20.
        pop_HeI_flux = np.log10(np.cumsum(10**(np.array(pop.population_ion_HEI_flux_iterations) - 20.))) + 20.
        pop_HeII_flux = np.log10(np.cumsum(10**(np.array(pop.population_ion_HEII_flux_iterations) - 20.))) + 20.

        pop_windpower = np.log10(np.cumsum(pop.population_windpowers_calc)) + 35.
        pop_windmom = np.log10(np.cumsum(pop.population_windmoms_calc)) + 35.

        pop_uv_slope = np.array([get_uv_slope(np.log10(spec_lib.wave_grid), np.log10(csf_population_flux_total[i, :]) + 20.)[2]
                                 for i in range(len(csf_population_flux_total))])

        csf_ews = [get_ew(spec_lib.wave_grid, csf_population_flux_total[i, :], pop_HI_flux[i])
                   for i in range(len(pop_HI_flux))]
        pop_Ha_ew, pop_Hb_ew, pop_Pb_ew, pop_Bg_ew = [np.array([ews[j] for ews in csf_ews]) for j in range(4)]

        pop_colours = np.array([colours(csf_population_flux_total[i], spec_lib.wave_grid)
                                for i in range(len(csf_population_flux_total))])

        pop_SN_rate = np.log10(np.cumsum(10**np.array(pop.SN_rates_csf_calc)))

    return SimpleNamespace(SED_stellar=pop_SED_stellar, SED_total=pop_SED_total,
                           hires_flux=pop_hires_flux, hires_flux_wneb=pop_hires_flux_wneb,
                           hires_flux_norm=pop_hires_flux_norm,
                           Lbol=pop_Lbol, HI_flux=pop_HI_flux, HeI_flux=pop_HeI_flux, HeII_flux=pop_HeII_flux,
                           wind_power=pop_windpower, wind_momentum=pop_windmom, uv_slope=pop_uv_slope,
                           Ha_ew=pop_Ha_ew, Hb_ew=pop_Hb_ew, Pb_ew=pop_Pb_ew, Bg_ew=pop_Bg_ew,
                           colours=pop_colours, SN_rate=pop_SN_rate)