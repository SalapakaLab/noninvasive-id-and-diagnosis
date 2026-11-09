'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions or the Differential EMG Sensor (biopotential sensor) circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_simple_math_TF_biopot_sensor_python, get_math_TF_biopot_sensor_MATLAB
from src.tf_estimation import estimate_tf_PSD, merge_LF_HF_frequency_resp, sample_logarithmically
from src.utils import manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt


def main():
    # User defined variables
    freq_uLim = 10e3
    freq_mLim = 1e2
    freq_lLim = 1

    # Read DC Operating point data from the CSV file
    file_path = "data/op_and_freq_resp_data/bio_pot_sensor/bio_pot_sensor_dc_data.csv"
    try:
        dcOpData = pd.read_csv(file_path)
        # print(dcOpData.head())
        # Set 'Device' as the index for easier lookup (like a dictionary)
        dcOpData.set_index('Device', inplace=True)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    Ra = 1e3
    Ca = 10e-6
    Rc = 1e3
    Cc = 10e-6
    Ro = 10e3
    Co = 32e-9
    Aa = 412.49
    Ab = 33.77
    Ac = 412.49
    Ao = 552.97
    circuit_params = {}
    circuit_params['Ra'] = Ra
    circuit_params['Ca'] = Ca
    circuit_params['Rc'] = Rc
    circuit_params['Cc'] = Cc
    circuit_params['Ro'] = Ro
    circuit_params['Co'] = Co



    #######################################################################################################################
    # Mathematical Transfer functions
    H_oa_MATH_mtlb, H_oc_MATH_mtlb, H_od_MATH_mtlb, H_db_MATH_mtlb = get_math_TF_biopot_sensor_MATLAB(circuit_params)

    # Get Bode Data (Math Model)
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase for MATLAB model
    mag_oa_mtlb, phase_oa_mtlb, _ = ct.frequency_response(H_oa_MATH_mtlb, omega_MATH)
    mag_oc_mtlb, phase_oc_mtlb, _ = ct.frequency_response(H_oc_MATH_mtlb, omega_MATH)
    mag_od_mtlb, phase_od_mtlb, _ = ct.frequency_response(H_od_MATH_mtlb, omega_MATH)
    mag_db_mtlb, phase_db_mtlb, _ = ct.frequency_response(H_db_MATH_mtlb, omega_MATH)

    mag_oa_db_mtlb = 20 * np.log10(mag_oa_mtlb)
    phase_oa_deg_mtlb = np.rad2deg(manual_unwrap(phase_oa_mtlb))

    mag_oc_db_mtlb = 20 * np.log10(mag_oc_mtlb)
    phase_oc_deg_mtlb = np.rad2deg(manual_unwrap(phase_oc_mtlb))

    mag_od_db_mtlb = 20 * np.log10(mag_od_mtlb)
    phase_od_deg_mtlb = np.rad2deg(phase_od_mtlb)

    mag_db_db_mtlb = 20 * np.log10(mag_db_mtlb)
    phase_db_deg_mtlb = np.rad2deg(phase_db_mtlb)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_Hoc_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_oc_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_Hoc = mag_data_Hoc_pd.to_numpy()
    phase_data_Hoc_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_oc_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_Hoc = phase_data_Hoc_pd.to_numpy()
    mag_Hoc = mag_data_Hoc[:,1]
    phase_Hoc = phase_data_Hoc[:,1]
    frequencies_ac_sim_Hoc = phase_data_Hoc[:,0]
    ind_1_Hoc = np.searchsorted(frequencies_ac_sim_Hoc, freq_lLim, side='right')
    ind_2_Hoc = np.searchsorted(frequencies_ac_sim_Hoc, freq_uLim, side='right')

    mag_data_Hod_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_od_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_Hod = mag_data_Hod_pd.to_numpy()
    phase_data_Hod_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_od_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_Hod = phase_data_Hod_pd.to_numpy()
    mag_Hod = mag_data_Hod[:,1]
    phase_Hod = phase_data_Hod[:,1]
    frequencies_ac_sim_Hod = phase_data_Hod[:,0]
    ind_1_Hod = np.searchsorted(frequencies_ac_sim_Hod, freq_lLim, side='right')
    ind_2_Hod = np.searchsorted(frequencies_ac_sim_Hod, freq_uLim, side='right')

    mag_data_Hoa_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_oa_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_Hoa = mag_data_Hoa_pd.to_numpy()
    phase_data_Hoa_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_oa_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_Hoa = phase_data_Hoa_pd.to_numpy()
    mag_Hoa = mag_data_Hoa[:,1]
    phase_Hoa = phase_data_Hoa[:,1]
    frequencies_ac_sim_Hoa = phase_data_Hoa[:,0]
    ind_1_Hoa = np.searchsorted(frequencies_ac_sim_Hoa, freq_lLim, side='right')
    ind_2_Hoa = np.searchsorted(frequencies_ac_sim_Hoa, freq_uLim, side='right')

    mag_data_Hdb_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_db_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_Hdb = mag_data_Hdb_pd.to_numpy()
    phase_data_Hdb_pd = pd.read_csv('data/op_and_freq_resp_data/bio_pot_sensor/H_db_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_Hdb = phase_data_Hdb_pd.to_numpy()
    mag_Hdb = mag_data_Hdb[:,1]
    phase_Hdb = phase_data_Hdb[:,1]
    frequencies_ac_sim_Hdb = phase_data_Hdb[:,0]
    ind_1_Hdb = np.searchsorted(frequencies_ac_sim_Hdb, freq_lLim, side='right')
    ind_2_Hdb = np.searchsorted(frequencies_ac_sim_Hdb, freq_uLim, side='right')





    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/bio_pot_sensor/healthy"
    list_of_nodes = ['a','b','c','d','o']
    
    # Load all data using list comprehension and stack into columns
    data_Normal_LF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_10mHz5kHz.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    
    data_Normal_HF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_100mHz50kHz.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])

    # Creating a dummy set of data because the estimator function expects two sets of data
    dummy_LF_data = np.copy(data_Normal_LF)
    dummy_HF_data = np.copy(data_Normal_HF)

    # Estimation alorithm parameters
    f_samp_HF = 1/10e-6
    nperseg_HF, noverlap_HF, win_HF = 4096, 0, 'hann'
    f_samp_LF = 1/100e-6
    nperseg_LF, noverlap_LF, win_LF = 4096*4, 0, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0]])

    # Estimate frequency response
    H_healthy_est_HF, _, f_HF = estimate_tf_PSD(data_Normal_HF, dummy_HF_data, Adj, n_FFT, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, freq_mLim, freq_uLim)
    H_healthy_est_LF, _, f_LF = estimate_tf_PSD(data_Normal_LF, dummy_LF_data, Adj, n_FFT, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, freq_lLim, freq_mLim)

    # Logaritming sampling of the frequency array
    n_log_pts = 100
    H_healthy_combined_linear, f_combined_linear = merge_LF_HF_frequency_resp(H_healthy_est_HF, H_healthy_est_LF, f_HF, f_LF, freq_lLim, freq_mLim, freq_uLim)
    f_combined, H_healthy_combined = sample_logarithmically(f_combined_linear, H_healthy_combined_linear, num_log_points=n_log_pts)

    H_oa_healthy_est = H_healthy_combined[4,0,:]
    H_oc_healthy_est = H_healthy_combined[4,2,:]
    H_od_healthy_est = H_healthy_combined[4,3,:]
    H_db_healthy_est = H_healthy_combined[3,1,:]



    #######################################################################################################################
    # %% Plot the bode plots for visualization
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    # %% Plot the bode plots: (1) LDIM vs AC Sim, and (2) Estimated vs AC Sim, for each transfer function
    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5

    tf_plot_specs = [
        {
            'tex_label': 'oc', 'title': r'$\hat{\mathcal{H}}_{oc}$',
            'mag_mtlb': mag_oc_db_mtlb, 'phase_mtlb': phase_oc_deg_mtlb, 'H_TF_mtlb': H_oc_MATH_mtlb,
            'ac_freq': frequencies_ac_sim_Hoc, 'ind_1': ind_1_Hoc, 'ind_2': ind_2_Hoc,
            'ac_mag': mag_Hoc, 'ac_phase': phase_Hoc, 'node_idx': (4, 2),
            'H_est': H_oc_healthy_est, 'unwrap_est_phase': True,
        },
        {
            'tex_label': 'od', 'title': r'$\hat{\mathcal{H}}_{od}$',
            'mag_mtlb': mag_od_db_mtlb, 'phase_mtlb': phase_od_deg_mtlb, 'H_TF_mtlb': H_od_MATH_mtlb,
            'ac_freq': frequencies_ac_sim_Hod, 'ind_1': ind_1_Hod, 'ind_2': ind_2_Hod,
            'ac_mag': mag_Hod, 'ac_phase': phase_Hod, 'node_idx': (4, 3),
            'H_est': H_od_healthy_est, 'unwrap_est_phase': False,
        },
        {
            'tex_label': 'oa', 'title': r'$\hat{\mathcal{H}}_{oa}$',
            'mag_mtlb': mag_oa_db_mtlb, 'phase_mtlb': phase_oa_deg_mtlb, 'H_TF_mtlb': H_oa_MATH_mtlb,
            'ac_freq': frequencies_ac_sim_Hoa, 'ind_1': ind_1_Hoa, 'ind_2': ind_2_Hoa,
            'ac_mag': mag_Hoa, 'ac_phase': phase_Hoa, 'node_idx': (4, 0),
            'H_est': H_oa_healthy_est, 'unwrap_est_phase': True,
        },
        {
            'tex_label': 'db', 'title': r'$\hat{\mathcal{H}}_{db}$',
            'mag_mtlb': mag_db_db_mtlb, 'phase_mtlb': phase_db_deg_mtlb, 'H_TF_mtlb': H_db_MATH_mtlb,
            'ac_freq': frequencies_ac_sim_Hdb, 'ind_1': ind_1_Hdb, 'ind_2': ind_2_Hdb,
            'ac_mag': mag_Hdb, 'ac_phase': phase_Hdb, 'node_idx': (3, 1),
            'H_est': H_db_healthy_est, 'unwrap_est_phase': False,
        },
    ]

    # Evaluate the LDIM (MATLAB) TFs exactly at the AC sim's own frequency points,
    # so the LDIM-AC error can be computed pointwise (mirrors eight_stag_amp's diff plot)
    mag_diff_db_dict = {}
    phase_diff_deg_dict = {}
    ac_freq_valid_dict = {}

    for spec in tf_plot_specs:
        is_last = (spec is tf_plot_specs[-1])
        ac_freq_valid = spec['ac_freq'][spec['ind_1']:spec['ind_2']]
        ac_mag_valid = spec['ac_mag'][spec['ind_1']:spec['ind_2']]
        ac_phase_valid = spec['ac_phase'][spec['ind_1']:spec['ind_2']]

        # LDIM - AC Sim error, evaluated at the AC sim's own frequency points
        mag_at_ac, phase_at_ac, _ = ct.frequency_response(spec['H_TF_mtlb'], ac_freq_valid * 2 * np.pi)
        mag_db_at_ac = 20 * np.log10(mag_at_ac)
        math_phase_unwrapped = np.unwrap(phase_at_ac)
        ac_phase_unwrapped = np.unwrap(np.deg2rad(ac_phase_valid))
        mag_diff_db_dict[spec['tex_label']] = mag_db_at_ac - ac_mag_valid
        phase_diff_deg_dict[spec['tex_label']] = (math_phase_unwrapped - ac_phase_unwrapped) * (180.0 / np.pi)
        ac_freq_valid_dict[spec['tex_label']] = ac_freq_valid

        # %% LDIM vs AC Simulation
        plt.figure(figsize=(1.7, 1.7))
        plt.subplot(2, 1, 1)
        plt.semilogx(freqs_hz_MATH, spec['mag_mtlb'], linewidth=3, linestyle='-', label="LDIM")
        plt.semilogx(ac_freq_valid, ac_mag_valid, linewidth=1.5, linestyle='--', label="AC Sim.")
        plt.title(spec['title'], fontsize=FONTSIZE)
        plt.ylabel('Mag.', fontsize=FONTSIZE)
        plt.xlim([freq_lLim, freq_uLim])
        if spec['title'] == r'$\hat{\mathcal{H}}_{db}$':
            plt.ylim([-5, 5])
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=FONTSIZE)

        plt.subplot(2, 1, 2)
        plt.semilogx(freqs_hz_MATH, spec['phase_mtlb'], linewidth=3, linestyle='-', label="LDIM")
        plt.semilogx(ac_freq_valid, ac_phase_valid, linewidth=1.5, linestyle='--', label="AC Sim.")
        plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
        plt.ylabel('Ph.', fontsize=FONTSIZE)
        plt.xlim([freq_lLim, freq_uLim])
        if spec['title'] == r'$\hat{\mathcal{H}}_{db}$':
            plt.ylim([-5, 5])
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        if is_last:
            plt.legend(
                fontsize=FONTSIZE,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.05),
                ncol=2,
                handlelength=1.0,
                handletextpad=0.4,
                columnspacing=1.5,
            )
        plt.tight_layout(pad=0, h_pad=0)
        plt.savefig(f"data/experimental_results/plots/TF_vs_AC/diffEMG_LDIM_vs_AC_H_{spec['tex_label']}.pdf", dpi=300, bbox_inches='tight')
        plt.show()

        # %% Estimated vs AC Simulation
        angles_estimate = np.angle(spec['H_est']) * (180 / np.pi)
        if spec['unwrap_est_phase']:
            angles_estimate = manual_unwrap(np.angle(spec['H_est'])) * (180 / np.pi)

        plt.figure(figsize=(1.7, 1.7))
        plt.subplot(2, 1, 1)
        plt.semilogx(ac_freq_valid, ac_mag_valid, linewidth=1.5, linestyle='-', label="AC Sim.")
        plt.semilogx(f_combined, 20 * (np.log10(np.abs(spec['H_est']))), linewidth=1.5, linestyle='-', label="Est.")
        plt.title(spec['title'], fontsize=FONTSIZE)
        plt.ylabel('Mag.', fontsize=FONTSIZE)
        plt.xlim([freq_lLim, freq_uLim])
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=FONTSIZE)

        plt.subplot(2, 1, 2)
        plt.semilogx(ac_freq_valid, ac_phase_valid, linewidth=1.5, linestyle='-', label="AC Sim.")
        plt.semilogx(f_combined, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
        plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
        plt.ylabel('Ph.', fontsize=FONTSIZE)
        plt.xlim([freq_lLim, freq_uLim])
        plt.grid(True)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        if is_last:
            plt.legend(
                fontsize=FONTSIZE,
                loc='lower center',
                bbox_to_anchor=(0.5, 1.05),
                ncol=2,
                handlelength=1.0,
                handletextpad=0.4,
                columnspacing=1.5,
            )
        plt.tight_layout(pad=0, h_pad=0)
        plt.savefig(f"data/experimental_results/plots/AC_vs_Est/diffEMG_Est_vs_AC_H_{spec['tex_label']}.pdf", dpi=300, bbox_inches='tight')
        plt.show()

    # %% Combined Difference Plot (LDIM - AC Sim), all transfer functions overlaid
    # Okabe-Ito colorblind-safe palette (distinct in hue and luminance, so it also
    # separates in grayscale print). Each curve also gets its own linestyle and a
    # sparse, staggered marker so curves stay distinguishable where they overlap.
    diff_colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']
    diff_linestyles = ['-', '--', ':', '-.']
    diff_markers = ['o', 's', '^', 'D']
    n_markers_target = 8

    def shade_from_zero(ax, freqs, values, color):
        '''Shades the region between zero and a single curve's value at each frequency,
        so the shaded area itself reads as that curve's deviation from a perfect match.'''
        ax.fill_between(freqs, 0, values, color=color, alpha=0.15, linewidth=0, zorder=0)

    fig_diff, (axd_mag, axd_phase) = plt.subplots(2, 1, figsize=(3.4, 2.5))
    for i, spec in enumerate(tf_plot_specs):
        label = spec['tex_label']
        color = diff_colors[i % len(diff_colors)]
        linestyle = diff_linestyles[i % len(diff_linestyles)]
        marker = diff_markers[i % len(diff_markers)]
        # Stagger the marker start index per curve so markers don't stack at
        # identical x-positions (all curves share the same AC-sim frequency grid)
        n_pts = len(ac_freq_valid_dict[label])
        step = max(1, n_pts // n_markers_target)
        offset = (i * step) // len(tf_plot_specs)
        markevery = (offset, step)

        shade_from_zero(axd_mag, ac_freq_valid_dict[label], mag_diff_db_dict[label], color)
        shade_from_zero(axd_phase, ac_freq_valid_dict[label], phase_diff_deg_dict[label], color)

        axd_mag.semilogx(ac_freq_valid_dict[label], mag_diff_db_dict[label],
                          linewidth=LINEWIDTH, linestyle=linestyle, color=color,
                          marker=marker, markevery=markevery, markersize=MARKERSIZE, markeredgewidth=0,
                          label=rf'$\Delta\hat{{\mathcal{{H}}}}_{{{label}}}$')
        axd_phase.semilogx(ac_freq_valid_dict[label], phase_diff_deg_dict[label],
                            linewidth=LINEWIDTH, linestyle=linestyle, color=color,
                            marker=marker, markevery=markevery, markersize=MARKERSIZE, markeredgewidth=0,
                            label=rf'$\Delta\hat{{\mathcal{{H}}}}_{{{label}}}$')

    axd_mag.set_ylabel('Mag. diff (dB)', fontsize=FONTSIZE)
    axd_mag.tick_params(axis='both', labelsize=FONTSIZE)
    axd_mag.tick_params(axis='x', labelbottom=False)
    axd_mag.set_xlim([freq_lLim, freq_uLim])
    axd_mag.grid(True)

    axd_phase.set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    axd_phase.set_ylabel('Ph. diff (deg)', fontsize=FONTSIZE)
    axd_phase.set_xlim([freq_lLim, freq_uLim])
    axd_phase.tick_params(axis='both', labelsize=FONTSIZE)
    axd_phase.grid(True)
    axd_phase.legend(fontsize=FONTSIZE, loc='upper right', ncol=2, handlelength=1)

    plt.tight_layout(pad=0.5, h_pad=1.0)
    plt.savefig('data/experimental_results/plots/TF_vs_AC/diffEMG_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()



# Entry point of the script
if __name__ == "__main__":
    main()  