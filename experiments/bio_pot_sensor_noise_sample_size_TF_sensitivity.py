'''This script quantifies how measurement noise (SNR) and sample size affect the accuracy of
frequency-response (transfer function) estimation for the Differential EMG Sensor (Biopotential
Sensor), relative to AC simulation ground truth.

The LF and HF recordings are analyzed *separately* rather than merged into one continuous
spectrum (as bio_pot_sensor_TF_ACsim_Estimated_Comparison.py does for its own plots): LF uses a
much larger nperseg (4096*16) than HF (4096), so the same nSamples value gives wildly different
Welch segment counts for the two (e.g. at 131072 samples: 2 LF segments vs. 32 HF segments).
Merging them into one curve would conflate two processes with very different data requirements,
and could confound the noise analysis right at the LF/HF crossover frequency. Each band therefore
gets its own estimator config, its own frequency range, and its own sample-size sweep values
(chosen relative to that band's own nperseg).

Two sweeps are run per band (LF, HF):
    1. SNR sweep, at a fixed sample size (all available data): uncorrelated Gaussian noise is
       added to the band's time series at a range of SNRs.
    2. Sample-size sweep, with no added noise: increasingly large prefixes of the (clean) time
       series are used.

At each sweep point the estimation routine (estimate_tf_PSD + sample_logarithmically, restricted
to the band's own frequency range -- no merging) is re-run, and the frequency-resolved
magnitude/phase error against the AC-sim response is computed (same log-frequency interpolation
as bio_pot_sensor_TF_ACsim_Estimated_Comparison.py's "Combined Difference Plot" section) -- kept
as a full error-vs-frequency curve rather than collapsed to a single summary number.
'''

import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.tf_estimation import estimate_tf_PSD, sample_logarithmically
from src.utils import (manual_unwrap, add_correlated_noise,
                       save_freq_error_sweep_data, load_freq_error_sweep_data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH_SNR_LF = 'data/experimental_results/algo_outputs/bio_pot_sensor_TF_LF_freq_error_vs_SNR.json'
DATA_PATH_SNR_HF = 'data/experimental_results/algo_outputs/bio_pot_sensor_TF_HF_freq_error_vs_SNR.json'
DATA_PATH_NSAMPLES_LF = 'data/experimental_results/algo_outputs/bio_pot_sensor_TF_LF_freq_error_vs_nSamples.json'
DATA_PATH_NSAMPLES_HF = 'data/experimental_results/algo_outputs/bio_pot_sensor_TF_HF_freq_error_vs_nSamples.json'
# Full 8-color Okabe-Ito colorblind-safe palette (black, orange, sky blue, bluish green, yellow,
# blue, vermillion, reddish purple) -- enough distinct colors for up to 8 sweep values (e.g. SNRs)
OKABE_ITO_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442',
                    '#0072B2', '#D55E00', '#CC79A7']
LINE_STYLES = ['-', '--', ':', '-.']
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def shade_from_zero(ax, freqs, values, color):
    '''Shades the region between zero and a single curve's value at each frequency, so the
    shaded area itself reads as that curve's (trace's) deviation from a perfect match.'''
    ax.fill_between(freqs, 0, values, color=color, alpha=0.15, linewidth=0, zorder=0)


def load_ac_sim_response(tex_label):
    '''
    Loads the AC-simulation magnitude/phase response for one transfer function of the bio-pot
    sensor, matching the CSV layout used in bio_pot_sensor_TF_ACsim_Estimated_Comparison.py.

    Parameters:
    ----------
    tex_label: transfer-function label used in the CSV filenames, e.g. 'oa', 'oc', 'od', 'db'

    Returns:
    -------
    freq: 1D array of AC-sim frequency points (Hz)
    mag_db: 1D array of AC-sim magnitude (dB) at those frequencies
    phase_deg: 1D array of AC-sim phase (degrees) at those frequencies
    '''

    base = f'data/op_and_freq_resp_data/bio_pot_sensor/H_{tex_label}'
    mag_data = pd.read_csv(f'{base}_mag_ac_sim.csv', skiprows=1, header=None).to_numpy()
    phase_data = pd.read_csv(f'{base}_phase_ac_sim.csv', skiprows=1, header=None).to_numpy()
    freq = phase_data[:, 0]
    mag_db = mag_data[:, 1]
    phase_deg = phase_data[:, 1]
    return freq, mag_db, phase_deg


def estimate_response(data, Adj, band_params):
    '''
    Estimates the transfer function from a single dataset (LF or HF alone -- no merging across
    datasets), restricted to [freq_lLim, freq_uLim] and log-sampled.

    Parameters:
    ----------
    data: nSamples X nNodes time-domain data
    Adj: nNodes X nNodes adjacency matrix of the graphical representation of the LDIM
    band_params: dict with keys 'nFFT', 'f_samp', 'nperseg', 'noverlap', 'win', 'freq_lLim',
        'freq_uLim', 'n_log_pts'

    Returns:
    -------
    f_log: 1D array of log-sampled frequency points spanning [freq_lLim, freq_uLim]
    H_log: nNodes X nNodes X len(f_log) estimated (complex) transfer function array
    '''

    p = band_params
    H_full, _, f_full = estimate_tf_PSD(data, np.copy(data), Adj, p['nFFT'], p['f_samp'],
                                        p['nperseg'], p['noverlap'], p['win'],
                                        p['freq_lLim'], p['freq_uLim'])
    idx_lo = np.searchsorted(f_full, p['freq_lLim'], side='right')
    idx_hi = np.searchsorted(f_full, p['freq_uLim'], side='right')
    f_band = f_full[idx_lo:idx_hi]
    H_band = H_full[:, :, idx_lo:idx_hi]
    f_log, H_log = sample_logarithmically(f_band, H_band, num_log_points=p['n_log_pts'])
    return f_log, H_log


def compute_freq_error(f_est, H_est, spec):
    '''
    Computes the frequency-resolved magnitude (dB) and phase (degrees) error between an estimated
    transfer function and its AC-simulation ground truth, interpolating the estimate (on a
    log-frequency axis) onto the AC sim's own frequency points -- matches the difference
    computation in the "Combined Difference Plot (Estimated - AC Sim)" section of
    bio_pot_sensor_TF_ACsim_Estimated_Comparison.py.

    Parameters:
    ----------
    f_est: 1D array of frequency points at which H_est is defined
    H_est: nNodes X nNodes X len(f_est) estimated (complex) transfer function array
    spec: dict with keys 'node_idx' (tuple (out_node,in_node) into H_est), 'unwrap_est_phase'
        (bool), 'ac_freq', 'ac_mag', 'ac_phase' (AC-sim arrays), 'ind_1','ind_2' (valid AC-sim
        index range within this spec's [freq_lLim, freq_uLim] band)

    Returns:
    -------
    ac_freq_valid: 1D array of AC-sim frequency points the errors are defined at
    mag_diff_db: 1D array of magnitude error (dB), same length as ac_freq_valid
    phase_diff_deg: 1D array of phase error (degrees), same length as ac_freq_valid
    '''

    out_node, in_node = spec['node_idx']
    H_node = H_est[out_node, in_node, :]
    mag_db_est_full = 20 * np.log10(np.abs(H_node))
    if spec['unwrap_est_phase']:
        phase_deg_est_full = manual_unwrap(np.angle(H_node)) * (180.0 / np.pi)
    else:
        phase_deg_est_full = np.angle(H_node) * (180.0 / np.pi)

    ac_freq_valid = spec['ac_freq'][spec['ind_1']:spec['ind_2']]
    ac_mag_valid = spec['ac_mag'][spec['ind_1']:spec['ind_2']]
    ac_phase_valid = spec['ac_phase'][spec['ind_1']:spec['ind_2']]

    mag_db_est_interp = np.interp(np.log10(ac_freq_valid), np.log10(f_est), mag_db_est_full)
    phase_deg_est_interp = np.interp(np.log10(ac_freq_valid), np.log10(f_est), phase_deg_est_full)
    ac_phase_unwrapped_deg = np.rad2deg(np.unwrap(np.deg2rad(ac_phase_valid)))

    mag_diff_db = mag_db_est_interp - ac_mag_valid
    phase_diff_deg = phase_deg_est_interp - ac_phase_unwrapped_deg
    return ac_freq_valid, mag_diff_db, phase_diff_deg


def build_tf_specs(freq_lLim, freq_uLim):
    '''
    Builds the per-transfer-function spec list (AC-sim data + node indices), restricted to a
    single band's [freq_lLim, freq_uLim] range, matching
    bio_pot_sensor_TF_ACsim_Estimated_Comparison.py's tf_plot_specs.

    Parameters:
    ----------
    freq_lLim, freq_uLim: this band's frequency range (Hz)

    Returns:
    -------
    tf_specs: list of dicts, one per transfer function, each with keys 'tex_label', 'node_idx',
        'unwrap_est_phase', 'title', 'ac_freq', 'ac_mag', 'ac_phase', 'ind_1', 'ind_2'
    '''

    tf_defs = [
        {'tex_label': 'oc', 'node_idx': (4, 2), 'unwrap_est_phase': True, 'title': r'$\hat{\mathcal{H}}_{oc}$'},
        {'tex_label': 'od', 'node_idx': (4, 3), 'unwrap_est_phase': False, 'title': r'$\hat{\mathcal{H}}_{od}$'},
        {'tex_label': 'oa', 'node_idx': (4, 0), 'unwrap_est_phase': True, 'title': r'$\hat{\mathcal{H}}_{oa}$'},
        {'tex_label': 'db', 'node_idx': (3, 1), 'unwrap_est_phase': False, 'title': r'$\hat{\mathcal{H}}_{db}$'},
    ]
    tf_specs = []
    for tf_def in tf_defs:
        ac_freq, ac_mag, ac_phase = load_ac_sim_response(tf_def['tex_label'])
        ind_1 = np.searchsorted(ac_freq, freq_lLim, side='right')
        ind_2 = np.searchsorted(ac_freq, freq_uLim, side='right')
        tf_specs.append({**tf_def, 'ac_freq': ac_freq, 'ac_mag': ac_mag, 'ac_phase': ac_phase,
                         'ind_1': ind_1, 'ind_2': ind_2})
    return tf_specs


def sweep_and_score(data, Adj, band_params, tf_specs):
    '''
    Runs the estimation routine once on the given (possibly noisy/truncated) single-band data,
    and computes the frequency-resolved error for every transfer function in tf_specs against
    its AC-sim ground truth.

    Returns:
    -------
    errors: dict mapping tf label -> (ac_freq_valid, mag_diff_db, phase_diff_deg)
    '''

    f_est, H_est = estimate_response(data, Adj, band_params)
    errors = {}
    for spec in tf_specs:
        ac_freq_valid, mag_diff_db, phase_diff_deg = compute_freq_error(f_est, H_est, spec)
        errors[spec['tex_label']] = (ac_freq_valid, mag_diff_db, phase_diff_deg)
    return errors


def run_sweep(data, Adj, band_params, tf_specs, x_values, make_data_for_x, x_label_fmt):
    '''
    Runs a full sweep (SNR or sample-size) for a single band's data/config, computing
    frequency-resolved error for every transfer function at each swept value.

    Parameters:
    ----------
    data: nSamples_total X nNodes clean time-domain data for this band
    Adj: nNodes X nNodes adjacency matrix
    band_params: dict of estimation params for this band (see estimate_response)
    tf_specs: tf specs for this band's frequency range (see build_tf_specs)
    x_values: list of swept parameter values (e.g. SNR in dB, or sample size)
    make_data_for_x: function(data, x) -> data to estimate from at this sweep point (e.g. add
        noise at a given SNR, or truncate to a given sample count)
    x_label_fmt: format string used only for the progress printout, applied as x_label_fmt.format(x)

    Returns:
    -------
    freq_by_tf, mag_by_tf, phase_by_tf: dicts mapping tf label to (respectively) its frequency
        grid, list of magnitude-error arrays, and list of phase-error arrays -- one list entry
        per entry in x_values
    '''

    tf_labels = [spec['tex_label'] for spec in tf_specs]
    freq_by_tf = {label: None for label in tf_labels}
    mag_by_tf = {label: [] for label in tf_labels}
    phase_by_tf = {label: [] for label in tf_labels}

    for x in x_values:
        data_x = make_data_for_x(data, x)
        errors = sweep_and_score(data_x, Adj, band_params, tf_specs)
        for label in tf_labels:
            ac_freq_valid, mag_diff_db, phase_diff_deg = errors[label]
            freq_by_tf[label] = ac_freq_valid
            mag_by_tf[label].append(mag_diff_db)
            phase_by_tf[label].append(phase_diff_deg)
        print(f"{x_label_fmt.format(x)}: " + ", ".join(
            f"{label} RMS mag={np.sqrt(np.mean(mag_by_tf[label][-1]**2)):.2f}dB" for label in tf_labels))

    return freq_by_tf, mag_by_tf, phase_by_tf


def plot_freq_error_traces(freq, x_values, mag_diff_list, phase_diff_list, trace_label_fmt,
                           save_path, title=None, show_legend=False, xlabel='Frequency (Hz)'):
    '''
    Plots magnitude and phase error vs. frequency (stacked subplots) for a single transfer
    function, with one trace per swept value (SNR or sample size). Styled with the Okabe-Ito
    palette plus distinct linestyles/markers per trace, and sparse markers staggered per trace so
    they don't overlap.

    These figures are meant to be used as subfigures sharing one legend per sweep/study, so the
    legend is only drawn when show_legend=True (call this once per group of figures, not on
    every one) -- it's anchored to the bottom axis in axes coordinates so it always sits a fixed,
    small gap below the x-axis label.

    Parameters:
    ----------
    freq: 1D array of frequency points (shared by every trace)
    x_values: list of swept parameter values, one per trace
    mag_diff_list: list of 1D magnitude-error arrays (dB), one per entry in x_values
    phase_diff_list: list of 1D phase-error arrays (deg), one per entry in x_values
    trace_label_fmt: format string for the legend, applied as trace_label_fmt.format(x)
    save_path: path to save the resulting PDF
    title: optional title placed above the magnitude subplot (e.g. the transfer-function label)
    show_legend: if True, draw the shared legend below this figure
    xlabel: x-axis label (frequency is always plotted on a log axis)
    '''

    FONTSIZE = 8
    n_markers_target = 6
    step = max(1, len(freq) // n_markers_target)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(1.7, 2.5))
    for idx, x in enumerate(x_values):
        color = OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)]
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        offset = (idx * step) // len(x_values)
        markevery = (offset, step)
        label = trace_label_fmt.format(x)

        shade_from_zero(ax_mag, freq, mag_diff_list[idx], color)
        shade_from_zero(ax_phase, freq, phase_diff_list[idx], color)

        ax_mag.semilogx(freq, mag_diff_list[idx], linestyle=linestyle, color=color,
                       marker=marker, markevery=markevery, markersize=3, markeredgewidth=0,
                       label=label)
        ax_phase.semilogx(freq, phase_diff_list[idx], linestyle=linestyle, color=color,
                         marker=marker, markevery=markevery, markersize=3, markeredgewidth=0,
                         label=label)

    if title is not None:
        ax_mag.set_title(title, fontsize=FONTSIZE)
    ax_mag.set_ylabel(r'$\Delta$ Mag', fontsize=FONTSIZE, labelpad=2)
    ax_mag.tick_params(axis='both', labelsize=FONTSIZE)
    ax_mag.tick_params(axis='x', labelbottom=False)
    ax_mag.grid(True)

    ax_phase.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax_phase.set_ylabel(r'$\Delta$ Ph', fontsize=FONTSIZE, labelpad=2)
    ax_phase.tick_params(axis='both', labelsize=FONTSIZE)
    ax_phase.grid(True)

    # figsize is identical for every call (legend or not), and the legend is anchored in FIGURE
    # coordinates (not axes-relative), positioned within the figure's own bottom margin rather
    # than allowed to spill below the canvas -- so together with the fixed figsize and no
    # bbox_inches='tight' cropping on save, every output PDF ends up exactly the same size,
    # whether or not it carries the legend. Reserving extra bottom margin (shrinking the axes)
    # is how that room gets made, as opposed to growing the overall figure. The margin scales
    # with the actual number of legend rows so it hugs the legend tightly instead of always
    # reserving worst-case room (which left a gap for shorter legends).
    ncol = 2
    if show_legend:
        n_rows = -(-len(x_values) // ncol)  # ceil division
        bottom_margin = 0.16 + 0.06 * n_rows
    else:
        bottom_margin = 0.24
    fig.subplots_adjust(left=0.32, right=0.95, top=0.86, bottom=bottom_margin, hspace=0.4)
    if show_legend:
        ax_phase.legend(fontsize=FONTSIZE - 2, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                        bbox_transform=fig.transFigure, ncol=ncol, handlelength=1.2,
                        columnspacing=0.8, labelspacing=0.3, borderaxespad=0.3)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                         help='Skip all sweeps and replot from previously saved data')
    args = parser.parse_args()

    band_freq_ranges = {'LF': (1, 1e2), 'HF': (1e2, 10e3)}
    tf_specs_by_band = {name: build_tf_specs(*freq_range) for name, freq_range in band_freq_ranges.items()}

    data_paths = {
        ('LF', 'SNR'): DATA_PATH_SNR_LF, ('HF', 'SNR'): DATA_PATH_SNR_HF,
        ('LF', 'nSamples'): DATA_PATH_NSAMPLES_LF, ('HF', 'nSamples'): DATA_PATH_NSAMPLES_HF,
    }
    results = {}

    if args.load:
        for key, path in data_paths.items():
            results[key] = load_freq_error_sweep_data(path)
    else:
        #######################################################################################################################
        # Load time series data
        time_series_path = "data/time_series_data/bio_pot_sensor/healthy"
        list_of_nodes = ['a', 'b', 'c', 'd', 'o']

        data_Normal_LF = np.column_stack([
            pd.read_csv(f'{time_series_path}/V{node}_10mHz5kHz.csv').iloc[:, 1:].values.flatten()
            for node in list_of_nodes
        ])
        data_Normal_HF = np.column_stack([
            pd.read_csv(f'{time_series_path}/V{node}_100mHz50kHz.csv').iloc[:, 1:].values.flatten()
            for node in list_of_nodes
        ])
        data_by_band = {'LF': data_Normal_LF, 'HF': data_Normal_HF}

        Adj = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0]])

        band_params_by_name = {
            'LF': {'nFFT': 4096, 'f_samp': 1 / 100e-6, 'nperseg': 4096 * 8, 'noverlap': 0, 'win': 'hann',
                  'freq_lLim': band_freq_ranges['LF'][0], 'freq_uLim': band_freq_ranges['LF'][1], 'n_log_pts': 100},
            'HF': {'nFFT': 4096, 'f_samp': 1 / 10e-6, 'nperseg': 4096, 'noverlap': 0, 'win': 'hann',
                  'freq_lLim': band_freq_ranges['HF'][0], 'freq_uLim': band_freq_ranges['HF'][1], 'n_log_pts': 100},
        }
        SNR_dB_list = [30, 15, 0, -15]
        # Sample-size lists are chosen per band relative to that band's own nperseg, so both span
        # the same story (1 Welch segment -- i.e. a raw periodogram, no averaging -- up to nearly
        # all available data) even though the absolute sample counts differ by ~16x.
        nSamplesList_by_band = {
            'LF': [262144, 524288, 1000000],   # 8,16,32 segments (nperseg=65536)
            'HF': [32768, 131072, 524288, 1000000],     # 1,8,32,128,244 segments (nperseg=4096)
        }
        seed = 42

        for band_name in ['LF', 'HF']:
            data = data_by_band[band_name]
            band_params = band_params_by_name[band_name]
            tf_specs = tf_specs_by_band[band_name]

            print("----------------------------------------______")
            print(f"[{band_name}] Sweeping SNR at max available sample size")

            def make_noisy(data, snr_db, seed=seed):
                rng = np.random.default_rng(seed)
                return add_correlated_noise(data, snr_db, 0.0, rng)

            snr_sweep_out = run_sweep(
                data, Adj, band_params, tf_specs, SNR_dB_list, make_noisy, x_label_fmt="SNR={:g} dB")
            # Stored as (x_values, *run_sweep_out) to match load_freq_error_sweep_data's return
            # shape, since both paths populate the same `results` dict.
            results[(band_name, 'SNR')] = (SNR_dB_list, *snr_sweep_out)
            save_freq_error_sweep_data(data_paths[(band_name, 'SNR')], SNR_dB_list, *snr_sweep_out)

            print("----------------------------------------______")
            print(f"[{band_name}] Sweeping sample size, no noise")
            nSamplesList = nSamplesList_by_band[band_name]
            nsamples_sweep_out = run_sweep(
                data, Adj, band_params, tf_specs, nSamplesList,
                lambda data, n: data[:n, :], x_label_fmt="nSamples={:g}")
            results[(band_name, 'nSamples')] = (nSamplesList, *nsamples_sweep_out)
            save_freq_error_sweep_data(data_paths[(band_name, 'nSamples')], nSamplesList, *nsamples_sweep_out)

    #######################################################################################################################
    # Plot results: one figure per transfer function per sweep per band
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    save_dir = 'data/experimental_results/plots/TF_Noise_SampleSize_Sensitivity'
    os.makedirs(save_dir, exist_ok=True)

    for band_name in ['LF', 'HF']:
        tf_specs = tf_specs_by_band[band_name]
        SNR_x, freq_snr, mag_snr, phase_snr = results[(band_name, 'SNR')]
        N_x, freq_n, mag_n, phase_n = results[(band_name, 'nSamples')]

        for i, spec in enumerate(tf_specs):
            label = spec['tex_label']
            is_last = (i == len(tf_specs) - 1)
            plot_freq_error_traces(
                freq_snr[label], SNR_x, mag_snr[label], phase_snr[label],
                trace_label_fmt='SNR={:g}dB', title=spec['title'], show_legend=is_last,
                save_path=f'{save_dir}/bio_pot_sensor_TF_{label}_{band_name}_freq_error_vs_SNR.pdf')
            plot_freq_error_traces(
                freq_n[label], N_x, mag_n[label], phase_n[label],
                trace_label_fmt='N={:g}', title=spec['title'], show_legend=is_last,
                save_path=f'{save_dir}/bio_pot_sensor_TF_{label}_{band_name}_freq_error_vs_nSamples.pdf')


    plt.show()


# Entry point of the script
if __name__ == "__main__":
    main()
