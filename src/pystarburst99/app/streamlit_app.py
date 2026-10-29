import math
import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")  # The plots are saved from Streamlit's script thread, not a GUI

import streamlit as st

import numpy as np

from pystarburst99.ins_outs import load_cfg, pySB99_cfg, TRACK_CHOICES, SPECTRA_LIBRARIES, STAR_FORMATION_OPTIONS, RUN_SPEED_MODES
from pystarburst99.plotting.plots import PySB_Plotter
from pystarburst99.workflow import TOML_PATH, Compiler


def select(label, options, key, **kwargs):
    options = list(options)
    index = options.index(D[key]) if D.get(key) in options else 0
    return st.selectbox(label, options, index=index, **kwargs)


def floats(text):
    return [float(x) for x in text.replace(",", " ").split()]


def browse_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)  # Dialog in front of the browser
        folder = filedialog.askdirectory(initialdir=st.session_state["parent_folder"], parent=root)
        root.destroy()
    except Exception as err:  # No tkinter or no display
        st.toast(f"Folder dialog not available: {err}")
        return
    if folder:
        st.session_state["parent_folder"] = folder


def progress_callback(bar, log):
    state = {"message": None, "pct": -1}

    def callback(tracker):
        pct = int(100 * tracker.fraction)
        if tracker.message != state["message"]:
            state["message"] = tracker.message
            log.text(tracker.message.strip())
        elif pct == state["pct"]:
            return  # Skip redraws which change nothing
        state["pct"] = pct

        in_loop = tracker.total and tracker.counter < tracker.total
        label = f"{tracker.stage.lstrip('- ')}: {tracker.counter}/{tracker.total}" if in_loop else tracker.message.strip()
        bar.progress(pct / 100, text=label)

    return callback


def plotter(compiler):
    """PySB_Plotter with the results of a compiler run (same data as in Compiler.save_results)."""
    inp, res, pop = compiler.inputs, compiler.results, compiler.pop
    data = SimpleNamespace(times=inp.times_steps, Z=inp.Z, rot=inp.rotation, output_age=inp.output_age,
                           sed_wave=compiler.spec_lib.wave_grid, SED_stellar=res.SED_stellar, SED_total=res.SED_total,
                           hires_wave=compiler.z_model.hires_wave_grid, hires_flux_wneb=res.hires_flux_wneb,
                           hires_flux_norm=res.hires_flux_norm, Lbol=res.Lbol, HI_flux=res.HI_flux,
                           HeI_flux=res.HeI_flux, HeII_flux=res.HeII_flux, wind_power=res.wind_power,
                           wind_momentum=res.wind_momentum, uv_slope=res.uv_slope, Ha_ew=res.Ha_ew,
                           MV=res.colours[:, 4], SN_rate=res.SN_rate, iso_teff=pop.population_uncorr_teffs,
                           iso_lum=pop.population_lums, iso_H=pop.population_H_abundances)
    return PySB_Plotter(data)


# The widgets start at the app defaults, then at the [default_settings] values
APP_DEFAULTS = dict(track_choice="GENEC", rotation=False, SED_library="FW", spectra_library="WM",
                    star_formation_option="ISF", log_age_start=6.0, log_age_end=8.0, log_age_step=0.1,
                    n_spectra=None, log_intervals=True, overwrite_results=True)
D = {**load_cfg(TOML_PATH)["default_settings"], **APP_DEFAULTS}

# The app passes every input, so the Compiler reads an empty [default_settings]
APP_TOML = Path(tempfile.gettempdir()) / "pystarburst99_app.toml"
APP_TOML.write_text("[default_settings]\n")

METALLICITIES = [k for k, v in pySB99_cfg.items() if isinstance(v, dict) and "folder" in v]

# Native folder dialog: only when the app runs on a machine with a display (not on Streamlit Cloud)
HAS_DISPLAY = not sys.platform.startswith("linux") or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# Plot type: (PySB_Plotter method, keyword arguments)
PLOT_TYPES = {"SED": ("sed", {}),
              "High-resolution spectrum": ("hires_spectrum", {}),
              "Normalised high-resolution spectrum": ("hires_spectrum", {"norm": True}),
              "Isochrone (HR diagram)": ("isochrone", {}),
              "Ionising photons HI": ("evolution", {"quantity": "HI"}),
              "Ionising photons HeI": ("evolution", {"quantity": "HeI"}),
              "Ionising photons HeII": ("evolution", {"quantity": "HeII"}),
              "Bolometric luminosity": ("evolution", {"quantity": "Lbol"}),
              "Wind power": ("evolution", {"quantity": "wind_power"}),
              "Wind momentum": ("evolution", {"quantity": "wind_momentum"}),
              "UV β slope": ("evolution", {"quantity": "uv_slope"}),
              "Hα equivalent width": ("evolution", {"quantity": "ew"}),
              "Absolute V magnitude": ("evolution", {"quantity": "MV"}),
              "Supernova rate": ("evolution", {"quantity": "SN_rate"})}

# Start the page design
st.set_page_config(page_title="pyStarburst99", layout="wide")
st.title("pyStarburst99")
st.caption("Stellar population synthesis based on Starburst99")

# Model selection
with st.container(border=True):
    st.subheader("Model")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        Z = select("Metallicity", METALLICITIES, "Z",
                   format_func=lambda z: f"{z} (Z = {pySB99_cfg[z]['Z_value']})")
        run_speed_mode = select("Run speed mode", RUN_SPEED_MODES, "run_speed_mode")
    z_cfg = pySB99_cfg[Z]
    with c2:
        track_choice = select("Evolutionary tracks", TRACK_CHOICES, "track_choice")
        st.space(20)
        rotation = st.checkbox("Rotation", value=bool(D.get("rotation", False)), disabled=track_choice == "Pauli")
    with c3:
        SED_library = select("SED library", [k for k in z_cfg["SED"] if k != "FW_VMS"], "SED_library")
        spectra_library = select("High-resolution library", SPECTRA_LIBRARIES, "spectra_library")
    with c4:
        mass_loss_rate_option = select("PoWR mass-loss option", z_cfg["hires_PoWR"], "mass_loss_rate_option",
                                       disabled=spectra_library != "PoWR")
        st.space(20)
        beta_tools = st.checkbox("Beta tools", value=bool(D.get("beta_tools", False)))

# Star formation and IMF
with st.container(border=True):
    st.subheader("Star formation and IMF")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        star_formation_option = select("Star formation", STAR_FORMATION_OPTIONS, "star_formation_option")
        if star_formation_option == "ISF":
            M_total = st.number_input("Total mass [M☉]", value=float(D.get("M_total") or 1e6), format="%.3e")
            SFR = None
        else:  # CSF: the Compiler computes M_total from the SFR
            SFR = st.number_input("SFR [M☉/yr]", value=float(D.get("SFR") or 1.0), format="%.3e")
            M_total = None
    with c2:
        IMF_exponents = st.text_input("IMF exponents", help="Comma-separated",
                                      value=", ".join(map(str, D.get("IMF_exponents", [1.3, 2.3]))))
        IMF_mass_limits = st.text_input("IMF mass limits [M☉]", help="Comma-separated, one more than the exponents",
                                        value=", ".join(map(str, D.get("IMF_mass_limits", [0.1, 0.5, 120.0]))))
    with c3:
        maximum_SN_mass = st.number_input("Maximum SN mass [M☉]", value=float(D.get("maximum_SN_mass", 120.0)))
    with c4:
        st.space(20)
        input_imf_option = st.checkbox("Stochastic IMF", value=bool(D.get("input_imf_option", False)))
        stochastic_imf_file = st.text_input("Stochastic IMF file (.npy)", value=D.get("stochastic_imf_file") or "",
                                            disabled=not input_imf_option)

# Time grid
with st.container(border=True):
    st.subheader("Time grid")
    c1, c2, c3, c4 = st.columns(4)
    start_default = D.get("log_age_start")
    with c1:
        time_step_length = st.number_input("Time step [yr]", value=float(D.get("time_step_length", 1e5)),
                                           min_value=1.0, format="%.3e")

        log_intervals = st.checkbox("Logarithmic age spacing", value=bool(D.get("log_intervals", False)))

        start_at_zero = st.checkbox("Start at 0 yr", disabled=log_intervals,
                                    value=start_default is not None and math.isinf(start_default))
    with c2:

        log_age_start = st.number_input("log(age start / yr)", step=0.1, format="%.2f",
                                        value=float(start_default) if start_default is not None
                                        and math.isfinite(start_default) else 6.0,
                                        disabled=start_at_zero and not log_intervals)
        log_age_end = st.number_input("log(age end / yr)", value=float(D.get("log_age_end", 7.0)), step=0.1,
                                      format="%.2f")
    with c3:
        spacing = st.radio("Ages from", ["Step", "Number of spectra"], horizontal=True,
                           index=int(D.get("n_spectra") is not None))
        if spacing == "Step":
            log_age_step = st.number_input("Step [dex, or log(step / yr) if linear]", step=0.1, format="%.2f",
                                           value=float(D.get("log_age_step") or 5.0))
            n_spectra = None
        else:
            n_spectra = st.number_input("Number of spectra", value=int(D.get("n_spectra") or 50), min_value=2, step=1)
            log_age_step = None
    with c4:
        age_decimals = st.number_input("Age decimals", value=D.get("age_decimals"), min_value=0, step=1,
                                       placeholder="auto")

if start_at_zero and not log_intervals:
    log_age_start = -math.inf

# Outputs
with st.container(border=True):
    st.subheader("Output")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.session_state.setdefault("parent_folder", str(Path.cwd()))
        parent_folder = st.text_input("Parent folder", key="parent_folder")
        if HAS_DISPLAY:
            st.button("Browse…", on_click=browse_folder)
    with c2:
        folder_name = st.text_input("Folder name", value=Z)
        plot_results_check = st.checkbox("Save plots", value=bool(D.get("plot_results_check", False)))

    with c3:
        output_age = st.number_input("Example output age [Myr]", value=float(D.get("output_age", 1.0)),
                                     min_value=0.0)
        overwrite_results = st.checkbox("Overwrite results", value=bool(D.get("overwrite_results", False)))

    output_folder = Path(parent_folder).expanduser() / folder_name
    st.caption(f"The grids will be saved in `{output_folder}`")

# Compilation launcher
if st.button("Run compilation", type="primary"):
    bar = st.progress(0.0, text="Starting")
    log = st.status("Compilation log", expanded=True)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            compiler = Compiler(Z=Z, output_folder=str(output_folder), track_choice=track_choice, rotation=rotation,
                                SED_library=SED_library, spectra_library=spectra_library,
                                mass_loss_rate_option=mass_loss_rate_option,
                                star_formation_option=star_formation_option, output_age=output_age,
                                plot_results_check=plot_results_check, input_imf_option=input_imf_option,
                                stochastic_imf_file=stochastic_imf_file or None, M_total=M_total, SFR=SFR,
                                IMF_exponents=floats(IMF_exponents), IMF_mass_limits=floats(IMF_mass_limits),
                                maximum_SN_mass=maximum_SN_mass, time_step_length=time_step_length,
                                run_speed_mode=run_speed_mode, log_intervals=log_intervals,
                                log_age_start=log_age_start, log_age_end=log_age_end, log_age_step=log_age_step,
                                n_spectra=n_spectra, age_decimals=age_decimals, beta_tools=beta_tools,
                                overwrite_results=overwrite_results, toml_path=APP_TOML,
                                verbose=False,  # The app shows the progress instead of the terminal
                                progress_callback=progress_callback(bar, log))
            compiler.population_synthesis()
            compiler.save_results()

    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as err:
        log.update(label="Compilation failed", state="error")
        st.error(str(err))

    else:
        log.update(label="Compilation complete", state="complete", expanded=False)
        st.session_state["compiler"] = compiler  # Available for later results/plots widgets
        st.success(f"Grids saved in `{output_folder}`")

    for w in caught:
        if issubclass(w.category, UserWarning):
            st.warning(str(w.message))

# Results
compiler = st.session_state.get("compiler")
if compiler is not None and compiler.results is not None:
    with st.container(border=True):
        st.subheader("Results")
        inp = compiler.inputs
        st.caption(f"Last run: Z = {inp.Z}, {inp.track_choice} tracks, {inp.SED_library} SEDs, "
                   f"{inp.spectra_library} high-resolution spectra, {inp.star_formation_option} · `{inp.output_folder}`")

        c1, c2 = st.columns([1, 2])
        with c1:
            plot_type = st.selectbox("Plot type", list(PLOT_TYPES))
        with c2:
            ages = (inp.times_spectra / 1e6).tolist()  # Output grid ages [Myr]
            default_age = ages[int(np.argmin(np.abs(np.array(ages) - inp.output_age)))]
            age = st.select_slider("Age [Myr]", options=ages, value=default_age, format_func=lambda a: f"{a:.4g}")

        # The plots show the closest internal time step
        t_plot = inp.times_steps[np.argmin(np.abs(inp.times_steps - age * 1e6))]
        if not np.isclose(t_plot, age * 1e6, rtol=1e-6):
            st.caption(f"Plotting the closest internal time step: {t_plot / 1e6:.4g} Myr")

        method, kwargs = PLOT_TYPES[plot_type]
        p = plotter(compiler)
        if method == "evolution":
            p.evolution(in_fig=None, **kwargs)
            p.ax.axvline(np.log10(t_plot), color="grey", linestyle="--", linewidth=1)  # Selected age
        else:
            getattr(p, method)(age=age, in_fig=None, **kwargs)

        fig_col, _ = st.columns([3, 2])
        fig_col.pyplot(p.fig)
        p.reset_figure()