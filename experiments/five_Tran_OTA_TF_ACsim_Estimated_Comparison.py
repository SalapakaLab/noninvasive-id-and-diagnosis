'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions or the five transistoir OTA circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_params, get_math_TF_5_tran_OTA_python
from src.tf_estimation import estimate_tf_PSD, sample_logarithmically
from src.utils import manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_time_series(data):
    nChannels = data.shape[1]
    for i in range(nChannels):
        plt.figure(figsize=(10, 4))
        plt.plot(data[:, i], label=f'V_{i+1}')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.show()



def main():
    # User defined variables
    freq_uLim = 1e7
    freq_lLim = 1e4


    # Read DC Operating point data from the CSV file
    file_path = "data/op_and_freq_resp_data/five_tran_ota/FiveTranOTA45nm_dc_data.csv"
    try:
        dcOpData = pd.read_csv(file_path)
        # print(dcOpData)
        # Set 'Device' as the index for easier lookup (like a dictionary)
        dcOpData.set_index('Device', inplace=True)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    # Make sure the device manes are entered properly.
    gm_1, rds_1 = get_params('M1',dcOpData)
    gm_2, rds_2 = get_params('M2',dcOpData)
    gm_3, rds_3 = get_params('M3',dcOpData)
    gm_4, rds_4 = get_params('M4',dcOpData)
    gm_5, rds_5 = get_params('M5',dcOpData)

    circuit_params = {}
    circuit_params['gm_1'] = gm_1
    circuit_params['rds_1'] = rds_1
    circuit_params['gm_2'] = gm_2
    circuit_params['rds_2'] = rds_2
    circuit_params['gm_3'] = gm_3
    circuit_params['rds_3'] = rds_3
    circuit_params['gm_4'] = gm_4
    circuit_params['rds_4'] = rds_4
    circuit_params['gm_5'] = gm_5
    circuit_params['rds_5'] = rds_5
    circuit_params['C_L'] = 100e-12





    #######################################################################################################################
    # Mathematical Transfer functions
    H_41_MATH, H_42_MATH, H_43_MATH = get_math_TF_5_tran_OTA_python(circuit_params)

    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase
    mag_41_MATH, phase_41_MATH, _ = ct.frequency_response(H_41_MATH, omega_MATH)
    mag_42_MATH, phase_42_MATH, _ = ct.frequency_response(H_42_MATH, omega_MATH)
    mag_43_MATH, phase_43_MATH, _ = ct.frequency_response(H_43_MATH, omega_MATH)

    # Convert to dB and degrees
    mag_41_db_MATH = 20 * np.log10(mag_41_MATH)
    phase_41_deg_MATH = np.rad2deg(phase_41_MATH) # control library might wrap phase

    mag_42_db_MATH = 20 * np.log10(mag_42_MATH)
    phase_42_deg_MATH = np.rad2deg(phase_42_MATH)




    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_H41_pd =  pd.read_csv('data/op_and_freq_resp_data/five_tran_ota/H41_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_H41 = mag_data_H41_pd.to_numpy()
    phase_data_H41_pd =  pd.read_csv('data/op_and_freq_resp_data/five_tran_ota/H41_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_H41 = phase_data_H41_pd.to_numpy()
    mag_H41 = mag_data_H41[:,1]
    phase_H41 = phase_data_H41[:,1]
    frequencies_ac_sim_H41 = phase_data_H41[:,0]
    ind_1_H41 = np.searchsorted(frequencies_ac_sim_H41,freq_lLim, side='right')
    ind_2_H41 = np.searchsorted(frequencies_ac_sim_H41, freq_uLim, side='right')

    mag_data_H42_pd =  pd.read_csv('data/op_and_freq_resp_data/five_tran_ota/H42_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_H42 = mag_data_H42_pd.to_numpy()
    phase_data_H42_pd =  pd.read_csv('data/op_and_freq_resp_data/five_tran_ota/H42_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_H42 = phase_data_H42_pd.to_numpy()
    mag_H42 = mag_data_H42[:,1]
    phase_H42 = phase_data_H42[:,1]
    frequencies_ac_sim_H42 = phase_data_H42[:,0]
    ind_1_H42 = np.searchsorted(frequencies_ac_sim_H42,freq_lLim, side='right')
    ind_2_H42 = np.searchsorted(frequencies_ac_sim_H42, freq_uLim, side='right')




    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/five_tran_ota/healthy"

    # Load all data
    data_Normal = pd.read_csv(f'{time_series_path}/V_100Hz100MHz.csv').iloc[:, [1, 3, 7]].values
    print("Data shape:", data_Normal.shape)
    print("First few rows of data:\n", data_Normal[:5])
    # plot_time_series(data_Normal)

    # Creating a dummy set of data because the estimator function expects two sets of data
    dummy_data = np.copy(data_Normal)

    # Estimation alorithm parameters
    f_samp = 1/5e-9
    nperseg, noverlap, win = 4096*8, 0, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 0]])
    
    # Estimate frequency response
    H_est_log, _, f_est_log = estimate_tf_PSD(data_Normal, dummy_data, Adj, n_FFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim)

    # Logaritming sampling of the frequency array
    n_log_pts = 100
    f_est, H_est = sample_logarithmically(f_est_log, H_est_log, num_log_points=n_log_pts)
    H_41_est = H_est[2,0,:] 
    H_42_est = H_est[2,1,:]





    #######################################################################################################################
    # %% Plot the bode plots for visualization


    # %% Plot Math Model vs AC Simulation
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5

    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_41_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_H41[ind_1_H41:ind_2_H41], mag_H41[ind_1_H41:ind_2_H41], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{31}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    # plt.ylim([-35,30])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.yticks([10, 30])
    # plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    # plt.gca().xaxis.set_minor_locator(ticker.NullLocator())

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_41_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_H41[ind_1_H41:ind_2_H41], phase_H41[ind_1_H41:ind_2_H41], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    # plt.ylim([-600,0])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/five_tran_OTA_LDIM_vs_AC_H_31.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_42_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_H42[ind_1_H42:ind_2_H42], mag_H42[ind_1_H42:ind_2_H42], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.yticks([10, 30])
    plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_42_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_H42[ind_1_H42:ind_2_H42], phase_H42[ind_1_H42:ind_2_H42], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    # plt.ylim([-150,-250])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.legend(
                fontsize=FONTSIZE,      
                loc='lower center',         # Anchor the bottom-center of the legend box...
                bbox_to_anchor=(0.5, 1.05), # ...to just slightly above the Phase plot's X-axis
                ncol=2,                     # Set to 2 (or more) so the legend spreads horizontally!
                # frameon=False,
                handlelength=1.0,      # <--- Reduces the length of the line
                handletextpad=0.4,     # <--- Reduces the gap between the line and the text
                columnspacing=1.5      # <--- (Optional) Reduces the gap between the two columns          
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/five_tran_OTA_LDIM_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %% Plot Estimated vs AC Simulation
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_H41, mag_H41, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_41_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{31}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.ylim([2,38])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.yticks([10, 20, 30])
    # plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    # plt.gca().xaxis.set_minor_locator(ticker.NullLocator())

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = ((np.angle(H_41_est))) * (180/np.pi)
    plt.semilogx(frequencies_ac_sim_H41, phase_H41, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    # plt.ylim([-600,0])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/five_tran_OTA_Est_vs_AC_H_31.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_H42, mag_H42, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_42_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.yticks([10, 20, 30])
    plt.ylim([2,38])
    plt.tick_params(axis='x', labelbottom=False)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_42_est))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_H42, phase_H42, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.ylim([70,220])
    plt.xlim([freq_lLim,freq_uLim])
    plt.grid(True)
    plt.legend(
                fontsize=FONTSIZE,      
                loc='lower center',         # Anchor the bottom-center of the legend box...
                bbox_to_anchor=(0.5, 1.05), # ...to just slightly above the Phase plot's X-axis
                ncol=2,                     # Set to 2 (or more) so the legend spreads horizontally!
                # frameon=False,
                handlelength=1.0,      # <--- Reduces the length of the line
                handletextpad=0.4,     # <--- Reduces the gap between the line and the text
                columnspacing=1.5      # <--- (Optional) Reduces the gap between the two columns          
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/five_tran_OTA_Est_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %% Combined Difference Plot (LDIM - AC Sim)
    # Evaluate the LDIM (mathematical) TFs exactly at the AC sim's own frequency points,
    # so the LDIM-AC error can be computed pointwise
    diff_specs = [
        {
            'tex_label': '31', 'H_TF': H_41_MATH,
            'ac_freq': frequencies_ac_sim_H41, 'ind_1': ind_1_H41, 'ind_2': ind_2_H41,
            'ac_mag': mag_H41, 'ac_phase': phase_H41,
            'H_est': H_41_est, 'unwrap_est_phase': False, 'est_phase_offset_deg': 0, 'node_idx': (2, 0),
        },
        {
            'tex_label': '32', 'H_TF': H_42_MATH,
            'ac_freq': frequencies_ac_sim_H42, 'ind_1': ind_1_H42, 'ind_2': ind_2_H42,
            'ac_mag': mag_H42, 'ac_phase': phase_H42,
            'H_est': H_42_est, 'unwrap_est_phase': True, 'est_phase_offset_deg': 360, 'node_idx': (2, 1),
        },
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
        math_phase_unwrapped = np.unwrap(phase_at_ac)
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
    plt.savefig('data/experimental_results/plots/TF_vs_AC/five_tran_OTA_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()





# Entry point of the script
if __name__ == "__main__":
    main()  