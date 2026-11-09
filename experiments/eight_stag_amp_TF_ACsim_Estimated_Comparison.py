'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions or the eight stage stagger tuned amplifier circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_params, get_math_TF_8_stagger_tunned_amplifier
from src.tf_estimation import estimate_tf_PSD, merge_LF_HF_frequency_resp, sample_logarithmically
from src.utils import manual_unwrap_eight_stag_TF, manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def main():
    # User defined variables
    freq_uLim = 1e12
    freq_mLim = 10e9
    freq_lLim = 10e5
    freq_uLim_est = 1e12
    freq_lLim_est = 1e10
    plot_Xlim = [freq_lLim, 1e10]
    plot_est_Xlim = [freq_lLim_est, freq_uLim_est]

    # Read DC Operating point data from the CSV file
    file_path = "data/op_and_freq_resp_data/eight_stagger_tuned_RF_amp/Eight_St_RF_amp_dc_data.csv"
    try:
        dcOpData = pd.read_csv(file_path)
        # print(dcOpData)
        # Set 'Device' as the index for easier lookup (like a dictionary)
        dcOpData.set_index('Device', inplace=True)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    R_vg = 1.5e3
    C_dc = 950e-15
    circuit_params = {}
    for stage in range(1,9):
        circuit_params[f'R_vg{stage}'] = R_vg
        circuit_params[f'C_{stage}'] = C_dc
        circuit_params[f'gm{stage}'], circuit_params[f'rds{stage}'] = get_params(f'M{stage-1}',dcOpData)
    circuit_params['L_1'] = 925e-12
    circuit_params['L_2'] = 590e-12
    circuit_params['L_3'] = 750e-12
    circuit_params['L_4'] = 900e-12
    circuit_params['L_5'] = 980e-12
    circuit_params['L_6'] = 535e-12
    circuit_params['L_7'] = 400e-12
    circuit_params['L_8'] = 1.2e-9





    #######################################################################################################################
    # Mathematical Transfer functions
    H_21_MATH, H_32_MATH, H_43_MATH, H_54_MATH, H_65_MATH, H_76_MATH, H_87_MATH = get_math_TF_8_stagger_tunned_amplifier(circuit_params)

    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase for all stage transfer functions
    H_TFs = [H_21_MATH, H_32_MATH, H_43_MATH, H_54_MATH, H_65_MATH, H_76_MATH, H_87_MATH]
    H_labels = ['H_21', 'H_32', 'H_43', 'H_54', 'H_65', 'H_76', 'H_87']
    
    mag_MATH_dict = {}
    phase_MATH_dict = {}
    mag_db_MATH_dict = {}
    phase_deg_MATH_dict = {}
    
    for i, (H_TF, label) in enumerate(zip(H_TFs, H_labels)):
        mag, phase, _ = ct.frequency_response(H_TF, omega_MATH)
        mag_MATH_dict[label] = mag
        phase_MATH_dict[label] = phase
        mag_db_MATH_dict[label] = 20 * np.log10(mag)
        phase_deg_MATH_dict[label] = np.rad2deg(phase)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    # Load AC simulation data for all transfer functions
    ac_data_dict = {}
    
    for label in H_labels:
        mag_file = f'data/op_and_freq_resp_data/eight_stagger_tuned_RF_amp/{label}_mag_ac_sim.csv'
        phase_file = f'data/op_and_freq_resp_data/eight_stagger_tuned_RF_amp/{label}_phase_ac_sim.csv'
        
        mag_data_pd = pd.read_csv(mag_file, skiprows=1, header=None)
        phase_data_pd = pd.read_csv(phase_file, skiprows=1, header=None)
        
        mag_data = mag_data_pd.to_numpy()
        phase_data = phase_data_pd.to_numpy()
        
        frequencies_ac_sim = phase_data[:, 0]
        ind_1 = np.searchsorted(frequencies_ac_sim, freq_lLim, side='right')
        ind_2 = np.searchsorted(frequencies_ac_sim, freq_uLim, side='right')
        
        ac_data_dict[label] = {
            'mag': mag_data[:, 1],
            'phase': phase_data[:, 1],
            'freq': frequencies_ac_sim,
            'ind_1': ind_1,
            'ind_2': ind_2
        }





    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/eight_stagger_tuned_RF_amp/Healthy"

    # Load all data
    data_Normal_LF = pd.read_csv(f'{time_series_path}/V_4M1T.csv').iloc[:, 1::2].values
    data_Normal_HF = pd.read_csv(f'{time_series_path}/V_20M10T.csv').iloc[:, 1::2].values

    # Creating a dummy set of data because the estimator function expects two sets of data
    dummy_LF_data = np.copy(data_Normal_LF)
    dummy_HF_data = np.copy(data_Normal_HF)

    # Estimation alorithm parameters
    f_samp_HF = 1/50e-15
    nperseg_HF, noverlap_HF, win_HF = 4096*4, 0, 'hann'
    f_samp_LF = 1/500e-15
    nperseg_LF, noverlap_LF, win_LF = 4096*16, 0, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0]])

    # Estimate frequency response
    H_healthy_est_HF, _, f_HF = estimate_tf_PSD(data_Normal_HF, dummy_HF_data, Adj, n_FFT, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, freq_lLim, freq_mLim)
    H_healthy_est_LF, _, f_LF = estimate_tf_PSD(data_Normal_LF, dummy_LF_data, Adj, n_FFT, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, freq_mLim, freq_uLim)

    # Logaritming sampling of the frequency array
    n_log_pts = 50
    H_healthy_combined_linear, f_combined_linear = merge_LF_HF_frequency_resp(H_healthy_est_HF, H_healthy_est_LF, f_HF, f_LF, freq_lLim, freq_mLim, freq_uLim)
    f_combined, H_healthy_combined = sample_logarithmically(f_combined_linear, H_healthy_combined_linear, num_log_points=n_log_pts)

    ind_1_est = np.searchsorted(f_combined, freq_lLim_est, side='right')
    ind_2_est = np.searchsorted(f_combined, freq_uLim_est, side='right')




    #######################################################################################################################
    # %% Plot the bode plots for visualization

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5

    # Plot each transfer function to compare the mathematical model, AC simulation
    for label in H_labels:
        if label == H_labels[0] or label == H_labels[2] or label == H_labels[4] or label == H_labels[6]:
            plt.figure(figsize=(2.15, 1.5))
        else:
            plt.figure(figsize=(2.0, 1.5))
        

        # Magnitude Plot
        plt.subplot(2, 1, 1)
        # Make the bottom line thicker (linewidth=3) and solid
        plt.semilogx(freqs_hz_MATH, mag_db_MATH_dict[label], 
                    linewidth=3, linestyle='-', label="LDIM")
        ac_data = ac_data_dict[label]
        # Make the top line thinner (linewidth=1.5) and dashed
        plt.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
             ac_data['mag'][ac_data['ind_1']:ac_data['ind_2']], 
             linewidth=1.5, linestyle='--', label="AC Sim.")
        plt.title(rf'$\hat{{\mathcal{{H}}}}_{{{label[2]}{label[3]}}}$', fontsize=FONTSIZE)
        plt.ylabel('Mag.', fontsize = FONTSIZE)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.tick_params(axis='x', labelbottom=False)
        plt.xlim(plot_Xlim)
        plt.grid(True)

        # Phase Plot
        plt.subplot(2, 1, 2)
        plt.semilogx(freqs_hz_MATH, phase_deg_MATH_dict[label], 
                    linewidth=3, linestyle='-', label="LDIM")
        plt.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
                     ac_data['phase'][ac_data['ind_1']:ac_data['ind_2']], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
        plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
        plt.ylabel('Ph.', fontsize=FONTSIZE)
        plt.ylim([-150, 10])
        plt.xlim(plot_Xlim)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.grid(True)
        plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
        plt.savefig(f'data/experimental_results/plots/TF_vs_AC/eight_stag_amp_LDIM_vs_AC_{label}.pdf', dpi=300, bbox_inches='tight')


    # %% Plot Estimated vs AC Simulation
    for label in H_labels:
        if label == H_labels[0] or label == H_labels[2] or label == H_labels[4] or label == H_labels[6]:
            plt.figure(figsize=(2.15, 1.5))
        else:
            plt.figure(figsize=(2.0, 1.5))
        
        out_node = int(label[3]) - 1
        in_node = int(label[2]) - 1
        angles_estimate = manual_unwrap((np.angle(H_healthy_combined[in_node, out_node, :]))) * (180/np.pi)

        # Magnitude Plot
        plt.subplot(2, 1, 1)
        ac_data = ac_data_dict[label]
        # Make the top line thinner (linewidth=1.5) and dashed
        plt.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
             ac_data['mag'][ac_data['ind_1']:ac_data['ind_2']], 
             linewidth=1.5, linestyle='-', label="AC Sim.")
        # Make the top line thinner (linewidth=1.5) and dashed
        plt.semilogx(f_combined[ind_1_est:ind_2_est], 20*(np.log10(np.abs(H_healthy_combined[in_node, out_node, ind_1_est:ind_2_est]))), 
             linewidth=1.5, linestyle='-', label="Est.")
        plt.title(rf'$\hat{{\mathcal{{H}}}}_{{{label[2]}{label[3]}}}$', fontsize=FONTSIZE)
        if label == H_labels[0] or label == H_labels[2] or label == H_labels[4] or label == H_labels[6]:
            plt.ylabel('Mag.', fontsize = FONTSIZE)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.tick_params(axis='x', labelbottom=False)
        plt.xlim(plot_est_Xlim)
        plt.ylim([-55, 10])
        plt.grid(True)

        # Phase Plot
        plt.subplot(2, 1, 2)
        plt.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
                     ac_data['phase'][ac_data['ind_1']:ac_data['ind_2']], 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
        plt.semilogx(f_combined[ind_1_est:ind_2_est], angles_estimate[ind_1_est:ind_2_est], 
                     linewidth=1.5, linestyle='-', label="Est.")
        plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
        if label == H_labels[0] or label == H_labels[2] or label == H_labels[4] or label == H_labels[6]:
            plt.ylabel('Ph.', fontsize = FONTSIZE)
        plt.ylim([-360, -40])
        plt.xlim(plot_est_Xlim)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.grid(True)
        plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
        plt.savefig(f'data/experimental_results/plots/AC_vs_Est/eight_stag_amp_Est_vs_AC_{label}.pdf', dpi=300, bbox_inches='tight')


    # Single combined plot for all transfer functions (AC Sim vs LDIM)
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(3.4, 4.0))
    # Okabe-Ito colorblind-safe palette (distinct in hue and luminance, so it also
    # separates in grayscale print). Black is reserved for the LDIM/AC-sim legend
    # proxies below, so it is left out of the per-curve palette.
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    for i, label in enumerate(H_labels):
        # Assign a unique color to each transfer function
        color = colors[i % len(colors)]
        tf_label = f'$H_{{{label[2]}{label[3]}}}$'
        ac_data = ac_data_dict[label]

        # Magnitude Plot
        ax_mag.semilogx(freqs_hz_MATH, mag_db_MATH_dict[label], 
                        linewidth=2.5, linestyle='-', color=color, label=tf_label)
        # AC Sim (Dashed line)
        ax_mag.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
                        ac_data['mag'][ac_data['ind_1']:ac_data['ind_2']], 
                        linewidth=1.5, linestyle='--', color=color)

        # Phase Plot
        # LDIM
        ax_phase.semilogx(freqs_hz_MATH, phase_deg_MATH_dict[label], 
                        linewidth=2.5, linestyle='-', color=color)
        # AC Sim
        ax_phase.semilogx(ac_data['freq'][ac_data['ind_1']:ac_data['ind_2']], 
                        ac_data['phase'][ac_data['ind_1']:ac_data['ind_2']], 
                        linewidth=1.5, linestyle='--', color=color)

    ax_mag.set_ylabel('Mag. (dB)', fontsize=FONTSIZE)
    ax_mag.tick_params(axis='both', labelsize=FONTSIZE)
    ax_mag.tick_params(axis='x', labelbottom=False)  # Hide x labels for top plot
    ax_mag.set_xlim(plot_Xlim)
    ax_mag.grid(True)

    ax_phase.set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    ax_phase.set_ylabel('Ph. (deg)', fontsize=FONTSIZE)
    ax_phase.set_ylim([-150, 10])
    ax_phase.set_xlim(plot_Xlim)
    ax_phase.tick_params(axis='both', labelsize=FONTSIZE)
    ax_phase.grid(True)

    #Create the Combined Legend
    handles, lgd_labels = ax_mag.get_legend_handles_labels()
    ldim_line = mlines.Line2D([], [], color='black', linewidth=2.5, linestyle='-', label='LDIM')
    ac_line = mlines.Line2D([], [], color='black', linewidth=1.5, linestyle='--', label='AC Sim.')
    handles.extend([ldim_line, ac_line])
    lgd_labels.extend(['LDIM', 'AC Sim.'])
    fig.legend(
        handles, lgd_labels,
        fontsize=FONTSIZE,      
        loc='center left',     
        bbox_to_anchor=(0.83, 0.5), # Anchored in the blank space on the right
        ncol=1,
        handlelength=1.5
    )
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.83, 1])
    plt.savefig('data/experimental_results/plots/TF_vs_AC/eight_stag_amp_LDIM_vs_AC_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Evaluate the mathematical TFs exactly at the AC sim's own frequency points
    mag_diff_db_dict = {}
    phase_diff_deg_dict = {}
    ac_freq_valid_dict = {}
    for H_TF, label in zip(H_TFs, H_labels):
        ac_data = ac_data_dict[label]
        ac_freq = ac_data['freq']
        ind1 = ac_data['ind_1']
        ind2 = ac_data['ind_2']
        # Use only the valid frequency range from the AC sim to avoid extrapolation
        ac_freq_valid = ac_freq[ind1:ind2]
        ac_mag_valid = ac_data['mag'][ind1:ind2]
        ac_phase_valid = ac_data['phase'][ind1:ind2]

        mag_at_ac, phase_at_ac, _ = ct.frequency_response(H_TF, ac_freq_valid * 2 * np.pi)
        mag_db_at_ac = 20 * np.log10(mag_at_ac)
        math_phase_unwrapped = np.unwrap(phase_at_ac)

        ac_phase_rad = np.deg2rad(ac_phase_valid)
        ac_phase_unwrapped = np.unwrap(ac_phase_rad)

        # Differences: LDIM (math) minus AC sim, in dB and degrees
        mag_diff_db_dict[label] = mag_db_at_ac - ac_mag_valid
        phase_diff_deg_dict[label] = (math_phase_unwrapped - ac_phase_unwrapped) * (180.0 / np.pi)
        ac_freq_valid_dict[label] = ac_freq_valid

    # Combined Difference Plot (LDIM - AC Sim)
    # The 7 curves are nearly identical, so color alone isn't enough to tell them apart
    # where they overlap. Each curve also gets its own linestyle and a sparse marker
    # (redundant encoding), which keeps them distinguishable even in grayscale print.
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    def shade_from_zero(ax, freqs, values, color):
        '''Shades the region between zero and a single curve's value at each frequency,
        so the shaded area itself reads as that curve's deviation from a perfect match.'''
        ax.fill_between(freqs, 0, values, color=color, alpha=0.15, linewidth=0, zorder=0)

    fig_diff, (axd_mag, axd_phase) = plt.subplots(2, 1, figsize=(3.5, 2.5))
    n_markers_target = 8
    for i, label in enumerate(H_labels):
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        # All 7 curves share the same AC-sim frequency grid, so a fractional
        # markevery would place every curve's markers at identical x-positions.
        # Stagger the start index per curve instead, so markers fall at
        # different frequencies and don't stack on top of each other.
        n_pts = len(ac_freq_valid_dict[label])
        step = max(1, n_pts // n_markers_target)
        offset = (i * step) // len(H_labels)
        markevery = (offset, step)

        shade_from_zero(axd_mag, ac_freq_valid_dict[label], mag_diff_db_dict[label], color)
        shade_from_zero(axd_phase, ac_freq_valid_dict[label], phase_diff_deg_dict[label], color)

        # Plot magnitude difference (dB)
        axd_mag.semilogx(ac_freq_valid_dict[label], mag_diff_db_dict[label],
                          linewidth=LINEWIDTH, linestyle=linestyle, color=color,
                          marker=marker, markevery=markevery, markersize=MARKERSIZE, markeredgewidth=0,
                          label=rf'$\Delta\hat{{\mathcal{{H}}}}_{{{label[2]}{label[3]}}}$')
        # Plot phase difference (deg)
        axd_phase.semilogx(ac_freq_valid_dict[label], phase_diff_deg_dict[label],
                            linewidth=LINEWIDTH, linestyle=linestyle, color=color,
                            marker=marker, markevery=markevery, markersize=MARKERSIZE, markeredgewidth=0,
                            label=rf'$\Delta\hat{{\mathcal{{H}}}}_{{{label[2]}{label[3]}}}$')

    axd_mag.set_ylabel('Mag. diff (dB)', fontsize=FONTSIZE)
    axd_mag.tick_params(axis='both', labelsize=FONTSIZE)
    axd_mag.tick_params(axis='x', labelbottom=False)
    axd_mag.set_xlim(plot_Xlim)
    axd_mag.set_ylim([-0.05, 0.05])
    axd_mag.grid(True)

    axd_phase.set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    axd_phase.set_ylabel('Ph. diff (deg)', fontsize=FONTSIZE)
    axd_phase.set_ylim([-0.1, 1])
    axd_phase.set_xlim(plot_Xlim)
    axd_phase.tick_params(axis='both', labelsize=FONTSIZE)
    axd_phase.grid(True)

    # Legend (use magnitude axis handles)
    handles, lgd_labels = axd_mag.get_legend_handles_labels()
    # fig_diff.legend(handles, lgd_labels, fontsize=FONTSIZE, loc='center left', bbox_to_anchor=(0.83, 0.5), ncol=1, handlelength=1.5)
    axd_phase.legend(fontsize=FONTSIZE-1, loc='upper left', ncol=2, handlelength=1)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.83, 1])
    plt.savefig('data/experimental_results/plots/TF_vs_AC/eight_stag_amp_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()



# Entry point of the script
if __name__ == "__main__":
    main()  