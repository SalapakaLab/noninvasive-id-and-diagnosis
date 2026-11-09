'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions or the MSTA TIA circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_math_TF_three_MSTA_TIA
from src.tf_estimation import compute_MSTA_TIA_tf_fast_vectorized, sample_logarithmically, compute_partial_Wiener_coeffs_PSD
from src.utils import manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt


def main():
    # User defined variables
    freq_uLim = 1e6
    freq_lLim = 10

    freq_uLim_est = 1e6
    freq_lLim_est = 100


    # DC Operating point data from Cadence Simulation
    # Make sure the device manes are entered properly.
    gm_1 = 21.59e-3
    rds_1 = 1/(159.8e-6)
    gm_2 = 22.5e-3
    rds_2 = 1/(354.7e-6)
    gm_3 = 20.73e-3
    rds_3 = 1/(207.2e-6)
    RM4 = 1/(1.86e-3)

    circuit_params = {}
    circuit_params['gm1'] = gm_1
    circuit_params['rds1'] = rds_1
    circuit_params['gm2'] = gm_2
    circuit_params['rds2'] = rds_2
    circuit_params['gm3'] = gm_3
    circuit_params['rds3'] = rds_3
    circuit_params['RM4'] = RM4
    circuit_params['CL'] = 50e-15
    circuit_params['Cin'] = 200e-15
    circuit_params['RF'] = 3.5e3
    circuit_params['Rd2'] = 110
    circuit_params['Rd3'] = 120
    circuit_params['L2'] = 2.4e-9
    circuit_params['L3'] = 1.5e-9





    #######################################################################################################################
    # Mathematical Transfer functions
    H_ba_MATH, H_cb_MATH, H_ac_MATH = get_math_TF_three_MSTA_TIA(circuit_params)

    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase
    mag_ba_MATH, phase_ba_MATH, _ = ct.frequency_response(H_ba_MATH, omega_MATH)
    mag_cb_MATH, phase_cb_MATH, _ = ct.frequency_response(H_cb_MATH, omega_MATH)
    mag_ac_MATH, phase_ac_MATH, _ = ct.frequency_response(H_ac_MATH, omega_MATH)

    # Convert to dB and degrees
    mag_ba_db_MATH = 20 * np.log10(mag_ba_MATH)
    phase_ba_deg_MATH = np.rad2deg(phase_ba_MATH)

    mag_cb_db_MATH = 20 * np.log10(mag_cb_MATH)
    phase_cb_deg_MATH = np.rad2deg(phase_cb_MATH)

    mag_ac_db_MATH = 20 * np.log10(mag_ac_MATH)
    phase_ac_deg_MATH = np.rad2deg(phase_ac_MATH)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_ba_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HBA = mag_data_HBA_pd.to_numpy()
    phase_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_ba_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HBA = phase_data_HBA_pd.to_numpy()
    mag_HBA = mag_data_HBA[:,1]
    phase_HBA = phase_data_HBA[:,1]
    frequencies_ac_sim_HBA = phase_data_HBA[:,0]
    ind_1_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_lLim, side='right')
    ind_2_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_uLim, side='right')

    mag_data_HCB_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_cb_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HCB = mag_data_HCB_pd.to_numpy()
    phase_data_HCB_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_cb_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HCB = phase_data_HCB_pd.to_numpy()
    mag_HCB = mag_data_HCB[:,1]
    phase_HCB = phase_data_HCB[:,1]
    frequencies_ac_sim_HCB = phase_data_HCB[:,0]
    ind_1_HCB = np.searchsorted(frequencies_ac_sim_HCB, freq_lLim, side='right')
    ind_2_HCB = np.searchsorted(frequencies_ac_sim_HCB, freq_uLim, side='right')

    mag_data_HAC_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_ac_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HAC = mag_data_HAC_pd.to_numpy()
    phase_data_HAC_pd = pd.read_csv('data/op_and_freq_resp_data/three_MSTA_TIA/H_ac_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HAC = phase_data_HAC_pd.to_numpy()
    mag_HAC = mag_data_HAC[:,1]
    phase_HAC = phase_data_HAC[:,1]
    frequencies_ac_sim_HAC = phase_data_HAC[:,0]
    ind_1_HAC = np.searchsorted(frequencies_ac_sim_HAC, freq_lLim, side='right')
    ind_2_HAC = np.searchsorted(frequencies_ac_sim_HAC, freq_uLim, side='right')





    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/three_MSTA_TIA"
    
    # Load all data
    data_Normal = pd.read_csv(f'{time_series_path}/V_10Hz10MHz_fsamp_50ns.csv').iloc[:, 1::2].values
    print("Data shape:", data_Normal.shape)
    print("First few rows of data:\n", data_Normal[:5])


    # Estimation alorithm parameters
    # f_samp = 1/5e-12
    f_samp = 1/50e-9
    nperseg, noverlap, win = 4096*16, 4096*8, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 1],
                    [1, 0, 0],
                    [0, 1, 0]])
    n_log_pts = 100
    
    # Estimate frequency response    
    # Package the loaded SPICE data into a dictionary keyed by (j, i)
    # Format: (j, i): (frequencies, magnitude_db, phase_degrees)
    my_ac_ground_truth = {
        (1, 0): (frequencies_ac_sim_HBA, 
                mag_HBA, 
                phase_HBA),
                
        (2, 1): (frequencies_ac_sim_HCB, 
                mag_HCB, 
                phase_HCB),
                
        (0, 2): (frequencies_ac_sim_HAC, 
                mag_HAC, 
                phase_HAC)
    }

    H_est, f_est, r1_all, r2_all = compute_MSTA_TIA_tf_fast_vectorized(
        timeseries_data=data_Normal, 
        f_samp=f_samp, 
        nFFT=n_FFT, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        win=win, 
        n_log_pts=n_log_pts,
        ac_ground_truth=None
    )

    H_21_est = H_est[1,0,:]
    H_32_est = H_est[2,1,:]
    H_13_est = H_est[0,2,:]




    #######################################################################################################################
    # %% Estimation without using the revolving door criterion.
    W_21, f_est_log = compute_partial_Wiener_coeffs_PSD(data_Normal,f_samp, 1, 0, [], n_FFT, nperseg, noverlap, win)
    W_32, f_est_log = compute_partial_Wiener_coeffs_PSD(data_Normal,f_samp, 2, 1, [], n_FFT, nperseg, noverlap, win)
    W_13, f_est_log = compute_partial_Wiener_coeffs_PSD(data_Normal,f_samp, 0, 2, [], n_FFT, nperseg, noverlap, win)
    f_est_no_rev_door, H_21_est_no_rev_door = sample_logarithmically(f_est_log, W_21[:,0], num_log_points=n_log_pts)
    _, H_32_est_no_rev_door = sample_logarithmically(f_est_log, W_32[:,0], num_log_points=n_log_pts)
    _, H_13_est_no_rev_door = sample_logarithmically(f_est_log, W_13[:,0], num_log_points=n_log_pts)




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
    plt.figure(figsize=(1.85, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_ba_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], mag_HBA[ind_1_HBA:ind_2_HBA], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{21}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_ba_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], phase_HBA[ind_1_HBA:ind_2_HBA]-360, 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([-181,-179])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/three_MSTA_TIA_LDIM_vs_AC_H_21.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(2, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_cb_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HCB[ind_1_HCB:ind_2_HCB], mag_HCB[ind_1_HCB:ind_2_HCB], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_cb_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HCB[ind_1_HCB:ind_2_HCB], phase_HCB[ind_1_HCB:ind_2_HCB], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/three_MSTA_TIA_LDIM_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.75, 1.4))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_ac_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HAC[ind_1_HAC:ind_2_HAC], mag_HAC[ind_1_HAC:ind_2_HAC], 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.title(r'$\hat{\mathcal{H}}_{13}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([20,21])
    plt.tick_params(axis='x', labelbottom=False)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_ac_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HAC[ind_1_HAC:ind_2_HAC], phase_HAC[ind_1_HAC:ind_2_HAC]+360, 
                     linewidth=1.5, linestyle='--', label="AC Sim.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim,freq_uLim])
    plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.legend(
                fontsize=FONTSIZE-1,      
                loc='lower center',         # Anchor the bottom-center of the legend box...
                bbox_to_anchor=(0.5, 1.05), # ...to just slightly above the Phase plot's X-axis
                ncol=2,                     # Set to 2 (or more) so the legend spreads horizontally!
                # frameon=False,
                handlelength=1.0,      # <--- Reduces the length of the line
                handletextpad=0.4,     # <--- Reduces the gap between the line and the text
                columnspacing=1.5      # <--- (Optional) Reduces the gap between the two columns          
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/three_MSTA_TIA_LDIM_vs_AC_H_13.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %% Combined Difference Plot (LDIM - AC Sim)
    # Evaluate the LDIM (mathematical) TFs exactly at the AC sim's own frequency points,
    # so the LDIM-AC error can be computed pointwise. Each AC sim phase needs the same
    # +-360 deg correction applied to it above (to undo the AC sim tool's phase wrapping)
    # before the difference is meaningful.
    diff_specs = [
        {'tex_label': '21', 'H_TF': H_ba_MATH, 'ac_freq': frequencies_ac_sim_HBA, 'ind_1': ind_1_HBA, 'ind_2': ind_2_HBA, 'ac_mag': mag_HBA, 'ac_phase': phase_HBA, 'phase_offset_deg': -360,
         'H_est': H_21_est, 'est_phase_offset_deg': 360},
        {'tex_label': '32', 'H_TF': H_cb_MATH, 'ac_freq': frequencies_ac_sim_HCB, 'ind_1': ind_1_HCB, 'ind_2': ind_2_HCB, 'ac_mag': mag_HCB, 'ac_phase': phase_HCB, 'phase_offset_deg': 0,
         'H_est': H_32_est, 'est_phase_offset_deg': 360},
        {'tex_label': '13', 'H_TF': H_ac_MATH, 'ac_freq': frequencies_ac_sim_HAC, 'ind_1': ind_1_HAC, 'ind_2': ind_2_HAC, 'ac_mag': mag_HAC, 'ac_phase': phase_HAC, 'phase_offset_deg': 360,
         'H_est': H_13_est, 'est_phase_offset_deg': 0},
    ]

    mag_diff_db_dict = {}
    phase_diff_deg_dict = {}
    ac_freq_valid_dict = {}
    for spec in diff_specs:
        label = spec['tex_label']
        ac_freq_valid = spec['ac_freq'][spec['ind_1']:spec['ind_2']]
        ac_mag_valid = spec['ac_mag'][spec['ind_1']:spec['ind_2']]
        ac_phase_valid = spec['ac_phase'][spec['ind_1']:spec['ind_2']] + spec['phase_offset_deg']

        mag_at_ac, phase_at_ac, _ = ct.frequency_response(spec['H_TF'], ac_freq_valid * 2 * np.pi)
        mag_db_at_ac = 20 * np.log10(mag_at_ac)
        math_phase_unwrapped = np.unwrap(phase_at_ac)
        ac_phase_unwrapped = np.unwrap(np.deg2rad(ac_phase_valid))

        mag_diff_db_dict[label] = mag_db_at_ac - ac_mag_valid
        phase_diff_deg_dict[label] = (math_phase_unwrapped - ac_phase_unwrapped) * (180.0 / np.pi)
        ac_freq_valid_dict[label] = ac_freq_valid

    # Okabe-Ito colorblind-safe palette (distinct in hue and luminance, so it also
    # separates in grayscale print); distinct linestyle + staggered marker per curve
    diff_colors = ['#E69F00', '#56B4E9', '#009E73']
    diff_linestyles = ['-', '--', ':']
    diff_markers = ['o', 's', '^']
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
        # identical x-positions (all curves share the same AC-sim frequency grid)
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
    axd_phase.legend(fontsize=FONTSIZE, loc='upper left', ncol=3, handlelength=1)

    plt.tight_layout(pad=0.5, h_pad=1.0)
    plt.savefig('data/experimental_results/plots/TF_vs_AC/three_MSTA_TIA_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()






    plt.figure(figsize=(1.85, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HBA, mag_HBA, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_21_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{21}$', fontsize=FONTSIZE)
    # plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_21_est))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HBA, phase_HBA, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    # plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([-181,-179])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_21.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(2, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HCB, mag_HCB, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_32_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_32_est))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HCB, phase_HCB, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_32.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.75, 1.4))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HAC, mag_HAC, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, 20*(np.log10(np.abs(H_13_est))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{13}$', fontsize=FONTSIZE)
    # plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([20,21])
    plt.tick_params(axis='x', labelbottom=False)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_13_est))) * (180/np.pi)
    plt.semilogx(frequencies_ac_sim_HAC, phase_HAC, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    # plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.legend(
                fontsize=FONTSIZE-1,      
                loc='lower center',         
                bbox_to_anchor=(0.5, 1.05), 
                ncol=2,                    
                # frameon=False,
                handlelength=1.0,      
                handletextpad=0.4,    
                columnspacing=1.5        
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_13.pdf', dpi=300, bbox_inches='tight')
    plt.show()




    FONTSIZE = 7
    plt.figure(figsize=(1.15, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HBA, mag_HBA, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_21_est_no_rev_door))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{21}$', fontsize=FONTSIZE)
    # plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_21_est_no_rev_door))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HBA, phase_HBA, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    # plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([-181,-179])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_21_no_rev_door.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.15, 1.5))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HCB, mag_HCB, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_32_est_no_rev_door))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE)
    plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([5,10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tick_params(axis='x', labelbottom=False)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_32_est_no_rev_door))) * (180/np.pi) + 360
    plt.semilogx(frequencies_ac_sim_HCB, phase_HCB, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0.5, h_pad=1.0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_32_no_rev_door.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # %%
    plt.figure(figsize=(1.15, 1.4))
    plt.subplot(2, 1, 1)
    plt.semilogx(frequencies_ac_sim_HAC, mag_HAC, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_13_est_no_rev_door))), linewidth=1.5, linestyle='-', label="Est.")
    plt.title(r'$\hat{\mathcal{H}}_{13}$', fontsize=FONTSIZE)
    # plt.ylabel('Mag.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([20,21])
    plt.tick_params(axis='x', labelbottom=False)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout()
    # plt.tight_layout(pad=0, h_pad=0)

    # Phase Plot
    plt.subplot(2, 1, 2)
    angles_estimate = manual_unwrap((np.angle(H_13_est_no_rev_door))) * (180/np.pi)
    plt.semilogx(frequencies_ac_sim_HAC, phase_HAC, 
                     linewidth=1.5, linestyle='-', label="AC Sim.")
    plt.semilogx(f_est_no_rev_door, angles_estimate, linewidth=1.5, linestyle='-', label="Est.")
    plt.xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    # plt.ylabel('Ph.', fontsize=FONTSIZE)
    plt.xlim([freq_lLim_est,freq_uLim_est])
    # plt.ylim([179,181])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.legend(
                fontsize=FONTSIZE-1,      
                loc='lower center',         
                bbox_to_anchor=(0.5, 1.05), 
                ncol=2,                     
                # frameon=False,
                handlelength=1.0,      
                handletextpad=0.4,     
                columnspacing=1.5         
            )
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    # plt.tight_layout(pad=0, h_pad=0)
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 0.85, 1])
    plt.savefig(f'data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_H_13_no_rev_door.pdf', dpi=300, bbox_inches='tight')
    plt.show()




    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5 

    # Increased the height slightly to 2.5 to accommodate the legend between the rows
    fig, axs = plt.subplots(2, 3, figsize=(3.5, 1.8), sharex=True)

    # Column 1: H_21
    axs[0, 0].semilogx(frequencies_ac_sim_HBA, mag_HBA, lw=LINEWIDTH, ls='-', label="AC Sim.")
    axs[0, 0].semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_21_est_no_rev_door))), lw=LINEWIDTH, ls='-', label="Est.")
    axs[0, 0].set_title(r'$\hat{\mathcal{H}}_{21}$', fontsize=FONTSIZE, pad=3)
    axs[0, 0].set_ylabel('Mag.', fontsize=FONTSIZE) 

    # Phase
    angles_est_21 = manual_unwrap((np.angle(H_21_est_no_rev_door))) * (180/np.pi) + 360
    axs[1, 0].semilogx(frequencies_ac_sim_HBA, phase_HBA, lw=LINEWIDTH, ls='-')
    axs[1, 0].semilogx(f_est_no_rev_door, angles_est_21, lw=LINEWIDTH, ls='-')
    axs[1, 0].set_ylabel('Ph.', fontsize=FONTSIZE)
    axs[1, 0].set_xlabel('Freq. (Hz)', fontsize=FONTSIZE)

    # Column 2: H_32
    axs[0, 1].semilogx(frequencies_ac_sim_HCB, mag_HCB, lw=LINEWIDTH, ls='-')
    axs[0, 1].semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_32_est_no_rev_door))), lw=LINEWIDTH, ls='-')
    axs[0, 1].set_title(r'$\hat{\mathcal{H}}_{32}$', fontsize=FONTSIZE, pad=3)

    # Phase
    angles_est_32 = manual_unwrap((np.angle(H_32_est_no_rev_door))) * (180/np.pi) + 360
    axs[1, 1].semilogx(frequencies_ac_sim_HCB, phase_HCB, lw=LINEWIDTH, ls='-')
    axs[1, 1].semilogx(f_est_no_rev_door, angles_est_32, lw=LINEWIDTH, ls='-')
    axs[1, 1].set_xlabel('Freq. (Hz)', fontsize=FONTSIZE)

    # Column 3: H_13
    axs[0, 2].semilogx(frequencies_ac_sim_HAC, mag_HAC, lw=LINEWIDTH, ls='-')
    axs[0, 2].semilogx(f_est_no_rev_door, 20*(np.log10(np.abs(H_13_est_no_rev_door))), lw=LINEWIDTH, ls='-')
    axs[0, 2].set_title(r'$\hat{\mathcal{H}}_{13}$', fontsize=FONTSIZE, pad=3)

    # Phase 
    angles_est_13 = manual_unwrap((np.angle(H_13_est_no_rev_door))) * (180/np.pi)
    axs[1, 2].semilogx(frequencies_ac_sim_HAC, phase_HAC, lw=LINEWIDTH, ls='-')
    axs[1, 2].semilogx(f_est_no_rev_door, angles_est_13, lw=LINEWIDTH, ls='-')
    axs[1, 2].set_xlabel('Freq. (Hz)', fontsize=FONTSIZE)

    # Global Formatting & Layout
    for ax in axs.flat:
        ax.set_xlim([freq_lLim_est, freq_uLim_est])
        ax.grid(True, linewidth=0.5)
        ax.tick_params(axis='both', labelsize=FONTSIZE-1, pad=1)

    plt.tight_layout(pad=0.2, w_pad=0.5, h_pad=2.5)

    # Extract legend handles from the first subplot
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # Attach the legend to the center-top plot (axs[0, 1]) and push it down below the x-axis
    axs[0, 1].legend(
        handles, labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.25), # Pushes it between the two rows
        ncol=2, 
        fontsize=FONTSIZE, 
        frameon=False, 
        handlelength=1.5, 
        handletextpad=0.4, 
        columnspacing=1.0
    )

    plt.savefig('data/experimental_results/plots/AC_vs_Est/three_MSTA_TIA_Est_vs_AC_no_rev_door_all_combined.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()





# Entry point of the script
if __name__ == "__main__":
    main()  