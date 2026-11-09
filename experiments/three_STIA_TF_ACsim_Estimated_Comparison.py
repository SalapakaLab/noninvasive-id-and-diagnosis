'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions or the three STIA circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_math_TF_three_STIA_low_freq
from src.tf_estimation import estimate_tf_PSD, sample_logarithmically, merge_LF_MF_HF_frequency_resp
from src.utils import manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    # User defined variables
    freq_uLim = 1e8
    freq_lLim = 1e3
    freq_uLim_est = 1e11
    freq_muLim_est = 1e10
    freq_mlLim_est = 1e7
    freq_lLim_est = 1e6
    plot_est_Xlim = [freq_lLim_est, freq_uLim_est]


    # DC Operating point data from Cadence Simulation
    # Make sure the device names are entered properly.
    gm_2 = 23.46e-3
    rds_2 = 1/(492e-6)
    gm_3 = 11.79e-3
    rds_3 = 1/(101.9e-6)

    circuit_params = {}
    circuit_params['gm2'] = gm_2
    circuit_params['rds2'] = rds_2
    circuit_params['gm3'] = gm_3
    circuit_params['rds3'] = rds_3
    circuit_params['C_L'] = 50e-15
    circuit_params['R_d2'] = 101
    circuit_params['R_d3'] = 110
    circuit_params['L_d2'] = 1.2e-9
    circuit_params['L_d3'] = 530e-12





    #######################################################################################################################
    # Mathematical Transfer functions
    H_ba_MATH, H_cb_MATH = get_math_TF_three_STIA_low_freq(circuit_params)

    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase
    mag_ba_MATH, phase_ba_MATH, _ = ct.frequency_response(H_ba_MATH, omega_MATH)
    mag_cb_MATH, phase_cb_MATH, _ = ct.frequency_response(H_cb_MATH, omega_MATH)

    # Convert to dB and degrees
    mag_ba_db_MATH = 20 * np.log10(mag_ba_MATH)
    phase_ba_deg_MATH = np.rad2deg(phase_ba_MATH)

    mag_cb_db_MATH = 20 * np.log10(mag_cb_MATH)
    phase_cb_deg_MATH = np.rad2deg(phase_cb_MATH)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/three_STIA/healthy/H_ba_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HBA = mag_data_HBA_pd.to_numpy()
    phase_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/three_STIA/healthy/H_ba_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HBA = phase_data_HBA_pd.to_numpy()
    mag_HBA = mag_data_HBA[:,1]
    phase_HBA = phase_data_HBA[:,1]
    frequencies_ac_sim_HBA = phase_data_HBA[:,0]
    ind_1_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_lLim, side='right')
    ind_2_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_uLim, side='right')

    mag_data_HCB_pd = pd.read_csv('data/op_and_freq_resp_data/three_STIA/healthy/H_cb_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HCB = mag_data_HCB_pd.to_numpy()
    phase_data_HCB_pd = pd.read_csv('data/op_and_freq_resp_data/three_STIA/healthy/H_cb_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HCB = phase_data_HCB_pd.to_numpy()
    mag_HCB = mag_data_HCB[:,1]
    phase_HCB = phase_data_HCB[:,1]
    frequencies_ac_sim_HCB = phase_data_HCB[:,0]
    ind_1_HCB = np.searchsorted(frequencies_ac_sim_HCB, freq_lLim, side='right')
    ind_2_HCB = np.searchsorted(frequencies_ac_sim_HCB, freq_uLim, side='right')





    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/three_STIA/healthy"
    list_of_nodes = ['a','b','c']
    
    # Load all data using list comprehension and stack into columns
    data_Normal_LF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_100Hz100MHz.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    data_Normal_MF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_1M1T.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    data_Normal_HF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_100M100T.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    print("Data shape:", data_Normal_LF.shape)
    print("First few rows of data:\n", data_Normal_LF[:5])

    # Creating a dummy set of data because the estimator function expects two sets of data
    dummy_data_LF = np.copy(data_Normal_LF)
    dummy_data_MF = np.copy(data_Normal_MF)
    dummy_data_HF = np.copy(data_Normal_HF)

    # Estimation alorithm parameters
    f_samp_LF = 1/5e-9
    f_samp_MF = 1/500e-15
    f_samp_HF = 1/5e-15
    nperseg_LF, noverlap_LF, win_LF = 4096*16, 4096, 'hann'
    nperseg_MF, noverlap_MF, win_MF = 4096*16, 4096, 'hann'
    nperseg_HF, noverlap_HF, win_HF = 4096*16, 4096, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0]])
    
    # Estimate frequency response
    H_est_LF, _, f_est_LF = estimate_tf_PSD(data_Normal_LF, dummy_data_LF, Adj, n_FFT, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, freq_lLim_est, freq_mlLim_est)
    H_est_MF, _, f_est_MF = estimate_tf_PSD(data_Normal_MF, dummy_data_MF, Adj, n_FFT, f_samp_MF, nperseg_MF, noverlap_MF, win_MF, freq_mlLim_est, freq_muLim_est)
    H_est_HF, _, f_est_HF = estimate_tf_PSD(data_Normal_HF, dummy_data_HF, Adj, n_FFT, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, freq_muLim_est, freq_uLim)

    # Logaritming sampling of the frequency array
    n_log_pts = 200
    H_healthy_combined_linear, f_combined_linear = merge_LF_MF_HF_frequency_resp(H_est_HF, H_est_MF, H_est_LF, f_est_HF, f_est_MF, f_est_LF, freq_lLim_est, freq_mlLim_est, freq_muLim_est, freq_uLim_est)
    f_est, H_est = sample_logarithmically(f_combined_linear, H_healthy_combined_linear, num_log_points=n_log_pts)

    H_ba_est = H_est[1,0,:]
    H_cb_est = H_est[2,1,:]




    #######################################################################################################################
    # %% Plot the bode plots for visualization
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    # %% Plot Math Model vs AC Simulation
    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_ba_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], mag_HBA[ind_1_HBA:ind_2_HBA], linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{21}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.ylim([6.2, 8])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_ba_deg_MATH+360, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], phase_HBA[ind_1_HBA:ind_2_HBA], linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    # plt.yticks([178, 180])
    plt.ylim([170, 190])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    plt.tight_layout(pad=0, h_pad=0)
    # plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/three_STIA_LDIM_vs_AC_H_21.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_cb_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HCB[ind_1_HCB:ind_2_HCB], mag_HCB[ind_1_HCB:ind_2_HCB], linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize = FONTSIZE)
    plt.ylim([1.5, 2.5])
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_cb_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HCB[ind_1_HCB:ind_2_HCB], phase_HCB[ind_1_HCB:ind_2_HCB], linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.ylim([170, 190])
    plt.legend(
                fontsize=FONTSIZE,      
                loc='lower center',         
                bbox_to_anchor=(0.5, 1.05), 
                ncol=2,                     
                # frameon=False,
                handlelength=1.0,      
                handletextpad=0.4,   
                columnspacing=1.5            
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    # plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/three_STIA_LDIM_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %% Combined Difference Plot (LDIM - AC Sim)
    # Evaluate the LDIM (mathematical) TFs exactly at the AC sim's own frequency points,
    # so the LDIM-AC error can be computed pointwise. H_21's LDIM phase needs the same
    # +360 deg correction applied to it above (to align it with the AC sim's phase
    # convention) before the difference is meaningful.
    diff_specs = [
        {'tex_label': '21', 'H_TF': H_ba_MATH, 'ac_freq': frequencies_ac_sim_HBA, 'ind_1': ind_1_HBA, 'ind_2': ind_2_HBA, 'ac_mag': mag_HBA, 'ac_phase': phase_HBA, 'math_phase_offset_deg': 360,
         'H_est': H_ba_est, 'est_phase_offset_deg': 360, 'node_idx': (1, 0)},
        {'tex_label': '32', 'H_TF': H_cb_MATH, 'ac_freq': frequencies_ac_sim_HCB, 'ind_1': ind_1_HCB, 'ind_2': ind_2_HCB, 'ac_mag': mag_HCB, 'ac_phase': phase_HCB, 'math_phase_offset_deg': 0,
         'H_est': H_cb_est, 'est_phase_offset_deg': 360, 'node_idx': (2, 1)},
    ]

    mag_diff_db_dict = {}
    phase_diff_deg_dict = {}
    ac_freq_valid_dict = {}
    for spec in diff_specs:
        label = spec['tex_label']
        ac_freq_valid = spec['ac_freq'][spec['ind_1']:spec['ind_2']]
        ac_mag_valid = spec['ac_mag'][spec['ind_1']:spec['ind_2']]
        ac_phase_valid = spec['ac_phase'][spec['ind_1']:spec['ind_2']]

        mag_at_ac, phase_at_ac, _ = ct.frequency_response(spec['H_TF'], ac_freq_valid * 2 * np.pi)
        mag_db_at_ac = 20 * np.log10(mag_at_ac)
        math_phase_deg = np.rad2deg(phase_at_ac) + spec['math_phase_offset_deg']
        math_phase_unwrapped = np.unwrap(np.deg2rad(math_phase_deg))
        ac_phase_unwrapped = np.unwrap(np.deg2rad(ac_phase_valid))

        mag_diff_db_dict[label] = mag_db_at_ac - ac_mag_valid
        phase_diff_deg_dict[label] = (math_phase_unwrapped - ac_phase_unwrapped) * (180.0 / np.pi)
        ac_freq_valid_dict[label] = ac_freq_valid

    # Okabe-Ito colorblind-safe palette (distinct in hue and luminance, so it also
    # separates in grayscale print); distinct linestyle + staggered marker per curve
    diff_colors = ['#E69F00', '#56B4E9']
    diff_linestyles = ['-', '--']
    diff_markers = ['o', 's']
    n_markers_target = 8

    def shade_from_zero(ax, freqs, values, color):
        '''Shades the region between zero and a single curve's value at each frequency,
        so the shaded area itself reads as that curve's deviation from a perfect match.'''
        ax.fill_between(freqs, 0, values, color=color, alpha=0.15, linewidth=0, zorder=0)

    fig_diff, (axd_mag, axd_phase) = plt.subplots(2, 1, figsize=(3.4, 2.5))
    for i, spec in enumerate(diff_specs):
        label = spec['tex_label']
        color = diff_colors[i % len(diff_colors)]
        linestyle = diff_linestyles[i % len(diff_linestyles)]
        marker = diff_markers[i % len(diff_markers)]
        # Stagger the marker start index per curve so markers don't stack at
        # identical x-positions (both curves share the same AC-sim frequency grid)
        n_pts = len(ac_freq_valid_dict[label])
        step = max(1, n_pts // n_markers_target)
        offset = (i * step) // len(diff_specs)
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
    axd_phase.legend(fontsize=FONTSIZE, loc='upper left', ncol=2, handlelength=1)

    plt.tight_layout(pad=0.5, h_pad=1.0)
    plt.savefig('data/experimental_results/plots/TF_vs_AC/three_STIA_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %% Plot Estimated vs AC Simulation
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HBA, mag_HBA, linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_ba_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{21}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.xlim(plot_est_Xlim)
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_ba_est))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HBA, phase_HBA, linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim(plot_est_Xlim)
    plt.grid(True)
    plt.yticks([0, 100, 180])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    plt.tight_layout(pad=0, h_pad=0)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_STIA_Est_vs_AC_H_21.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HCB, mag_HCB, linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_cb_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize = FONTSIZE)
    plt.ylim([-25, 5])
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.xlim(plot_est_Xlim)
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_cb_est))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HCB, phase_HCB, linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim(plot_est_Xlim)
    plt.grid(True)
    plt.ylim([-40, 240])
    plt.legend(
                fontsize=FONTSIZE,      
                loc='lower center',         
                bbox_to_anchor=(0.5, 1.05), 
                ncol=2,                     
                # frameon=False,
                handlelength=1.0,      
                handletextpad=0.4,     
                columnspacing=1.5               
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_STIA_Est_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()




# Entry point of the script
if __name__ == "__main__":
    main()  