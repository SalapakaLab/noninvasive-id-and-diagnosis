'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions for the hardware circuit'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_math_TF_Hardware_Circuit_python
from src.tf_estimation import estimate_tf_PSD, sample_logarithmically
from src.utils import manual_unwrap
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt


def main():
    # User defined variables
    freq_uLim = 1e5
    freq_lLim = 100


    # DC Operating point data from Cadence Simulation
    # Make sure the device names are entered properly.
    gm_2 = 52.46e-3
    rds_2 = 1/(391.6e-9)
    gm_4 = 44.15e-3
    rds_4 = 1/(277.4e-9)
    gm_5 = 52.46e-3
    rds_5 = 1/(391.7e-9)

    circuit_params = {}
    circuit_params['R1'] = 32e3
    circuit_params['R2'] = 8e3
    circuit_params['R3'] = 597
    circuit_params['R4'] = 100
    circuit_params['R5'] = 41.7e3
    circuit_params['R6'] = 3e3
    circuit_params['R7'] = 300
    circuit_params['R8'] = 100
    circuit_params['R9'] = 32e3
    circuit_params['R10'] = 8e3
    circuit_params['R11'] = 41.7e3
    circuit_params['R12'] = 3e3
    circuit_params['R13'] = 597
    circuit_params['R14'] = 100
    circuit_params['R15'] = 32e3
    circuit_params['R16'] = 8e3
    circuit_params['R17'] = 32e3
    circuit_params['R18'] = 100
    circuit_params['R19'] = 8e3
    circuit_params['R20'] = 41.7e3
    circuit_params['R21'] = 3e3
    circuit_params['R22'] = 597
    circuit_params['R23'] = 100
    circuit_params['R24'] = 32e3
    circuit_params['R25'] = 8e3
    circuit_params['C1'] = 1e-6
    circuit_params['C2'] = 1e-6
    circuit_params['C3'] = 1e-6
    circuit_params['C4'] = 1e-6
    circuit_params['C5'] = 1e-6
    circuit_params['C6'] = 1e-6
    circuit_params['C7'] = 1e-6
    circuit_params['C8'] = 1e-6
    circuit_params['gm2'] = gm_2
    circuit_params['gm4'] = gm_4
    circuit_params['gm5'] = gm_5
    circuit_params['rds2'] = rds_2
    circuit_params['rds4'] = rds_4
    circuit_params['rds5'] = rds_5





    #######################################################################################################################
    # Mathematical Transfer functions
    H_31_MATH, H_32_MATH, H_43_MATH = get_math_TF_Hardware_Circuit_python(circuit_params)

    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase
    mag_31_MATH, phase_31_MATH, _ = ct.frequency_response(H_31_MATH, omega_MATH)
    mag_32_MATH, phase_32_MATH, _ = ct.frequency_response(H_32_MATH, omega_MATH)
    mag_43_MATH, phase_43_MATH, _ = ct.frequency_response(H_43_MATH, omega_MATH)

    # Convert to dB and degrees
    mag_31_db_MATH = 20 * np.log10(mag_31_MATH)
    phase_31_deg_MATH = np.rad2deg(phase_31_MATH)

    mag_32_db_MATH = 20 * np.log10(mag_32_MATH)
    phase_32_deg_MATH = np.rad2deg(phase_32_MATH)

    mag_43_db_MATH = 20 * np.log10(mag_43_MATH)
    phase_43_deg_MATH = np.rad2deg(phase_43_MATH)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_H31_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_31_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_H31 = mag_data_H31_pd.to_numpy()
    phase_data_H31_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_31_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_H31 = phase_data_H31_pd.to_numpy()
    mag_H31 = mag_data_H31[:,1]
    phase_H31 = phase_data_H31[:,1]
    frequencies_ac_sim_H31 = phase_data_H31[:,0]
    ind_1_H31 = np.searchsorted(frequencies_ac_sim_H31, freq_lLim, side='right')
    ind_2_H31 = np.searchsorted(frequencies_ac_sim_H31, freq_uLim, side='right')

    mag_data_H32_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_32_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_H32 = mag_data_H32_pd.to_numpy()
    phase_data_H32_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_32_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_H32 = phase_data_H32_pd.to_numpy()
    mag_H32 = mag_data_H32[:,1]
    phase_H32 = phase_data_H32[:,1]
    frequencies_ac_sim_H32 = phase_data_H32[:,0]
    ind_1_H32 = np.searchsorted(frequencies_ac_sim_H32, freq_lLim, side='right')
    ind_2_H32 = np.searchsorted(frequencies_ac_sim_H32, freq_uLim, side='right')

    mag_data_H43_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_43_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_H43 = mag_data_H43_pd.to_numpy()
    phase_data_H43_pd = pd.read_csv('data/op_and_freq_resp_data/hardware_circuit/H_43_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_H43 = phase_data_H43_pd.to_numpy()
    mag_H43 = mag_data_H43[:,1]
    phase_H43 = phase_data_H43[:,1]
    frequencies_ac_sim_H43 = phase_data_H43[:,0]
    ind_1_H43 = np.searchsorted(frequencies_ac_sim_H43, freq_lLim, side='right')
    ind_2_H43 = np.searchsorted(frequencies_ac_sim_H43, freq_uLim, side='right')





    #######################################################################################################################
    # Get estimated transfer functions from data
    # Load time series data
    time_series_path = "data/time_series_data/hardware_circuit/healthy"

    # Load all data
    data_Normal = pd.read_excel(f'{time_series_path}/Healthy_Hardware_Data_1MS_s.xlsx').iloc[:, 1::2].values
    print("Data shape:", data_Normal.shape)
    print("First few rows of data:\n", data_Normal[:5])
    # Scaling factors/gains used in data acquisition system. We need to compensate 
    # for scaling of the instrumentation amplifiers and other filters in the measurement circuit.
    # The filters have unity gain between 15Hz and 100kHz, therefore we need to compensate for the
    # instrumentation amplifiers' gain.
    data_scale_fac_1 = (991*1.99)
    data_scale_fac_2 = (991*1.99)
    data_scale_fac_3 = (991*1.99)
    data_scale_fac_4 = (496*1.99)
    data_Normal[:,0] = data_Normal[:,0]*(1/data_scale_fac_1)
    data_Normal[:,1] = data_Normal[:,1]*(1/data_scale_fac_2)
    data_Normal[:,2] = data_Normal[:,2]*(1/data_scale_fac_3)
    data_Normal[:,3] = data_Normal[:,3]*(1/data_scale_fac_4)


    # Creating a dummy set of data because the estimator function expects two sets of data
    dummy_data = np.copy(data_Normal)

    # Estimation algorithm parameters
    f_samp = 1/1e-6
    nperseg, noverlap, win = 4096*8, 0, 'hann'
    n_FFT = 4096
    Adj = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0]])
    
    
    # Estimate frequency response
    H_est_log, _, f_est_log = estimate_tf_PSD(data_Normal, dummy_data, Adj, n_FFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim)
    
    # Logaritming sampling of the frequency array
    n_log_pts = 200
    f_est, H_est = sample_logarithmically(f_est_log, H_est_log, num_log_points=n_log_pts)

    H_31_est = H_est[2,0,:]
    H_32_est = H_est[2,1,:]
    H_43_est = H_est[3,2,:]



    #######################################################################################################################
    # %% Plot the Combined bode plots for visualization

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    FONTSIZE = 8

    # Create a 2 (rows) x 3 (columns) subplot grid.
    fig, axes = plt.subplots(2, 3, figsize=(5.1, 2))

    # COLUMN 0: H_31
    # Magnitude
    axes[0, 0].semilogx(f_est, 20*(np.log10(np.abs(H_31_est))), label="Est.")
    axes[0, 0].semilogx(freqs_hz_MATH, mag_31_db_MATH, linewidth=2.5, label="LDIM")
    axes[0, 0].semilogx(frequencies_ac_sim_H31[ind_1_H31:ind_2_H31], mag_H31[ind_1_H31:ind_2_H31], linestyle='--', label="AC Sim.")
    axes[0, 0].set_title('$H_{31}$', fontsize=FONTSIZE)
    axes[0, 0].set_ylabel('Mag.', fontsize=FONTSIZE)

    # Phase
    angles_estimate_31 = manual_unwrap((np.angle(H_31_est))) * (180/np.pi)
    axes[1, 0].semilogx(f_est, angles_estimate_31, label="Est.")
    axes[1, 0].semilogx(freqs_hz_MATH, phase_31_deg_MATH, linewidth=2.5, label="LDIM")
    axes[1, 0].semilogx(frequencies_ac_sim_H31[ind_1_H31:ind_2_H31], phase_H31[ind_1_H31:ind_2_H31], linestyle='--', label="AC Sim.")
    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)
    axes[1, 0].set_ylabel('Ph.', fontsize=FONTSIZE)


    # COLUMN 1: H_32
    # Magnitude
    axes[0, 1].semilogx(f_est, 20*(np.log10(np.abs(H_32_est))), label="Est.")
    axes[0, 1].semilogx(freqs_hz_MATH, mag_32_db_MATH, linewidth=2.5, label="LDIM")
    axes[0, 1].semilogx(frequencies_ac_sim_H32[ind_1_H32:ind_2_H32], mag_H32[ind_1_H32:ind_2_H32], linestyle='--', label="AC Sim.")
    axes[0, 1].set_title('$\hat{{\mathcal{{H}}}}_{32}$', fontsize=FONTSIZE)

    # Phase
    angles_estimate_32 = manual_unwrap((np.angle(H_32_est))) * (180/np.pi)
    axes[1, 1].semilogx(f_est, angles_estimate_32, label="Est.")
    axes[1, 1].semilogx(freqs_hz_MATH, phase_32_deg_MATH, linewidth=2.5, label="LDIM")
    axes[1, 1].semilogx(frequencies_ac_sim_H32[ind_1_H32:ind_2_H32], phase_H32[ind_1_H32:ind_2_H32], linestyle='--', label="AC Sim.")
    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)


    # COLUMN 2: H_43
    # Magnitude
    axes[0, 2].semilogx(f_est, 20*(np.log10(np.abs(H_43_est))), label="Est.")
    axes[0, 2].semilogx(freqs_hz_MATH, mag_43_db_MATH, linewidth=2.5, label="LDIM")
    axes[0, 2].semilogx(frequencies_ac_sim_H43[ind_1_H43:ind_2_H43], mag_H43[ind_1_H43:ind_2_H43], linestyle='--', label="AC Sim.")
    axes[0, 2].set_title('$\hat{{\mathcal{{H}}}}_{43}$', fontsize=FONTSIZE)

    # Phase
    angles_estimate_43 = manual_unwrap((np.angle(H_43_est))) * (180/np.pi)
    axes[1, 2].semilogx(f_est, angles_estimate_43, label="Est.")
    axes[1, 2].semilogx(freqs_hz_MATH, phase_43_deg_MATH, linewidth=2.5, label="LDIM")
    axes[1, 2].semilogx(frequencies_ac_sim_H43[ind_1_H43:ind_2_H43], phase_H43[ind_1_H43:ind_2_H43], linestyle='--', label="AC Sim.")
    axes[1, 2].set_xlabel('Frequency (Hz)', fontsize=FONTSIZE)


    # Apply limits, grids, and ticks to ALL subplots
    for ax in axes.flat:
        ax.set_xlim([freq_lLim, freq_uLim])
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=FONTSIZE)

    # Hide X-axis tick labels for the top row (Magnitude plots) to save space
    for ax in axes[0, :]:
        ax.tick_params(axis='x', labelbottom=False)

    # Grab the handles and labels from the first plot
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Place a single figure-level legend at the bottom center
    fig.legend(
        handles, labels,
        fontsize=FONTSIZE-1,      
        loc='upper center',
        bbox_to_anchor=(0.55, 0.66),
        ncol=3,
        handlelength=1.0,      
        handletextpad=0.4,     
        columnspacing=1.5      
    )

    # rect=[left, bottom, right, top]
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0, rect=[0, 0.1, 1, 1])
    plt.savefig('data/experimental_results/plots/Hardware_TF_AC_Est/hardware_circuit_TF_ac_est_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()




# Entry point of the script
if __name__ == "__main__":
    main()  