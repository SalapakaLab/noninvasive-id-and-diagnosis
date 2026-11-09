'''This script implements the experiment related to comparison of frequency response from AC
simulation to that from Mathematical transfer functions for the LDO'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.math_models import get_params, get_math_TF_LDO_python
import pandas as pd
import numpy as np
import control as ct
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    # User defined variables
    freq_uLim = 1e6
    freq_lLim = 1000


    # Read DC Operating point data from the CSV file
    file_path = "data/op_and_freq_resp_data/LDO/LDO_dc_data.csv"
    try:
        dcOpData = pd.read_csv(file_path)
        # print(dcOpData.head())
        # Set 'Device' as the index for easier lookup (like a dictionary)
        dcOpData.set_index('Device', inplace=True)
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    # Make sure the device names are entered properly.
    gm_1, rds_1 = get_params('M1',dcOpData)
    gm_2, rds_2 = get_params('M2',dcOpData)
    gm_3, rds_3 = get_params('M3',dcOpData)
    gm_4, rds_4 = get_params('M4',dcOpData)
    gm_5, rds_5 = get_params('M5',dcOpData)
    gm_6, rds_6 = get_params('M6',dcOpData)
    gm_7, rds_7 = get_params('M7',dcOpData)
    gm_8, rds_8 = get_params('M8',dcOpData)
    gm_9, rds_9 = get_params('M9',dcOpData)
    gm_10, rds_10 = get_params('M10',dcOpData)
    gm_11, rds_11 = get_params('M11',dcOpData)

    circuit_params = {}
    circuit_params['gm1'] = gm_1
    circuit_params['rds1'] = rds_1
    circuit_params['gm2'] = gm_2
    circuit_params['rds2'] = rds_2
    circuit_params['gm3'] = gm_3
    circuit_params['rds3'] = rds_3
    circuit_params['gm4'] = gm_4
    circuit_params['rds4'] = rds_4
    circuit_params['gm5'] = gm_5
    circuit_params['rds5'] = rds_5
    circuit_params['gm6'] = gm_6
    circuit_params['rds6'] = rds_6
    circuit_params['gm7'] = gm_7
    circuit_params['rds7'] = rds_7
    circuit_params['gm8'] = gm_8
    circuit_params['rds8'] = rds_8
    circuit_params['gm9'] = gm_9
    circuit_params['rds9'] = rds_9
    circuit_params['gm10'] = gm_10
    circuit_params['rds10'] = rds_10
    circuit_params['gm11'] = gm_11
    circuit_params['rds11'] = rds_11
    circuit_params['Ra'] = 40e3
    circuit_params['Rb'] = 40e3
    circuit_params['Rt'] = 1.5e3
    circuit_params['Rc'] = 500
    circuit_params['Cc'] = 1e-12
    circuit_params['R1'] = 1e3
    circuit_params['R2'] = 9e3





    #######################################################################################################################
    # Mathematical Transfer functions
    H_QX_MATH, H_QA_MATH, H_QY_MATH, H_BX_MATH, H_BA_MATH, H_BY_MATH, H_YQ_MATH, H_YB_MATH = get_math_TF_LDO_python(circuit_params)
    # Get Bode Data (Math Model)
    # create frequency vector in rad/s
    omega_MATH = np.logspace(np.log10(freq_lLim * 2 * np.pi), np.log10(freq_uLim * 2 * np.pi), 1000)
    freqs_hz_MATH = omega_MATH / (2 * np.pi)

    # Calculate magnitude and phase
    mag_QX_MATH, phase_QX_MATH, _ = ct.frequency_response(H_QX_MATH, omega_MATH)
    mag_QA_MATH, phase_QA_MATH, _ = ct.frequency_response(H_QA_MATH, omega_MATH)
    mag_QY_MATH, phase_QY_MATH, _ = ct.frequency_response(H_QY_MATH, omega_MATH)
    mag_BX_MATH, phase_BX_MATH, _ = ct.frequency_response(H_BX_MATH, omega_MATH)
    mag_BA_MATH, phase_BA_MATH, _ = ct.frequency_response(H_BA_MATH, omega_MATH)
    # mag_BQ_MATH, phase_BQ_MATH, _ = ct.frequency_response(H_BQ_MATH, omega_MATH)
    mag_BY_MATH, phase_BY_MATH, _ = ct.frequency_response(H_BY_MATH, omega_MATH)
    mag_YQ_MATH, phase_YQ_MATH, _ = ct.frequency_response(H_YQ_MATH, omega_MATH)
    mag_YB_MATH, phase_YB_MATH, _ = ct.frequency_response(H_YB_MATH, omega_MATH)

    # Convert to dB and degrees
    mag_QX_db_MATH = 20 * np.log10(mag_QX_MATH)
    phase_QX_deg_MATH = np.rad2deg(phase_QX_MATH)

    mag_QA_db_MATH = 20 * np.log10(mag_QA_MATH)
    phase_QA_deg_MATH = np.rad2deg(phase_QA_MATH)

    mag_QY_db_MATH = 20 * np.log10(mag_QY_MATH)
    phase_QY_deg_MATH = np.rad2deg(phase_QY_MATH)

    mag_BX_db_MATH = 20 * np.log10(mag_BX_MATH)
    phase_BX_deg_MATH = np.rad2deg(phase_BX_MATH)

    mag_BA_db_MATH = 20 * np.log10(mag_BA_MATH)
    phase_BA_deg_MATH = np.rad2deg(phase_BA_MATH)

    mag_BY_db_MATH = 20 * np.log10(mag_BY_MATH)
    phase_BY_deg_MATH = np.rad2deg(phase_BY_MATH)

    mag_YQ_db_MATH = 20 * np.log10(mag_YQ_MATH)
    phase_YQ_deg_MATH = np.rad2deg(phase_YQ_MATH)

    mag_YB_db_MATH = 20 * np.log10(mag_YB_MATH)
    phase_YB_deg_MATH = np.rad2deg(phase_YB_MATH)





    #######################################################################################################################
    # Get the bode plots from AC simulations
    mag_data_HQX_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QX_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HQX = mag_data_HQX_pd.to_numpy()
    phase_data_HQX_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QX_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HQX = phase_data_HQX_pd.to_numpy()
    mag_HQX = mag_data_HQX[:,1]
    phase_HQX = phase_data_HQX[:,1]
    frequencies_ac_sim_HQX = phase_data_HQX[:,0]
    ind_1_HQX = np.searchsorted(frequencies_ac_sim_HQX, freq_lLim, side='right')
    ind_2_HQX = np.searchsorted(frequencies_ac_sim_HQX, freq_uLim, side='right')

    mag_data_HQA_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QA_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HQA = mag_data_HQA_pd.to_numpy()
    phase_data_HQA_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QA_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HQA = phase_data_HQA_pd.to_numpy()
    mag_HQA = mag_data_HQA[:,1]
    phase_HQA = phase_data_HQA[:,1]
    frequencies_ac_sim_HQA = phase_data_HQA[:,0]
    ind_1_HQA = np.searchsorted(frequencies_ac_sim_HQA, freq_lLim, side='right')
    ind_2_HQA = np.searchsorted(frequencies_ac_sim_HQA, freq_uLim, side='right')

    mag_data_HQY_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QY_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HQY = mag_data_HQY_pd.to_numpy()
    phase_data_HQY_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_QY_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HQY = phase_data_HQY_pd.to_numpy()
    mag_HQY = mag_data_HQY[:,1]
    phase_HQY = phase_data_HQY[:,1]
    frequencies_ac_sim_HQY = phase_data_HQY[:,0]
    ind_1_HQY = np.searchsorted(frequencies_ac_sim_HQY, freq_lLim, side='right')
    ind_2_HQY = np.searchsorted(frequencies_ac_sim_HQY, freq_uLim, side='right')

    mag_data_HBX_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BX_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HBX = mag_data_HBX_pd.to_numpy()
    phase_data_HBX_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BX_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HBX = phase_data_HBX_pd.to_numpy()
    mag_HBX = mag_data_HBX[:,1]
    phase_HBX = phase_data_HBX[:,1]
    frequencies_ac_sim_HBX = phase_data_HBX[:,0]
    ind_1_HBX = np.searchsorted(frequencies_ac_sim_HBX, freq_lLim, side='right')
    ind_2_HBX = np.searchsorted(frequencies_ac_sim_HBX, freq_uLim, side='right')

    mag_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BA_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HBA = mag_data_HBA_pd.to_numpy()
    phase_data_HBA_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BA_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HBA = phase_data_HBA_pd.to_numpy()
    mag_HBA = mag_data_HBA[:,1]
    phase_HBA = phase_data_HBA[:,1]
    frequencies_ac_sim_HBA = phase_data_HBA[:,0]
    ind_1_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_lLim, side='right')
    ind_2_HBA = np.searchsorted(frequencies_ac_sim_HBA, freq_uLim, side='right')


    mag_data_HBY_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BY_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HBY = mag_data_HBY_pd.to_numpy()
    phase_data_HBY_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_BY_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HBY = phase_data_HBY_pd.to_numpy()
    mag_HBY = mag_data_HBY[:,1]
    phase_HBY = phase_data_HBY[:,1]
    frequencies_ac_sim_HBY = phase_data_HBY[:,0]
    ind_1_HBY = np.searchsorted(frequencies_ac_sim_HBY, freq_lLim, side='right')
    ind_2_HBY = np.searchsorted(frequencies_ac_sim_HBY, freq_uLim, side='right')

    mag_data_HYQ_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_YQ_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HYQ = mag_data_HYQ_pd.to_numpy()
    phase_data_HYQ_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_YQ_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HYQ = phase_data_HYQ_pd.to_numpy()
    mag_HYQ = mag_data_HYQ[:,1]
    phase_HYQ = phase_data_HYQ[:,1]
    frequencies_ac_sim_HYQ = phase_data_HYQ[:,0]
    ind_1_HYQ = np.searchsorted(frequencies_ac_sim_HYQ, freq_lLim, side='right')
    ind_2_HYQ = np.searchsorted(frequencies_ac_sim_HYQ, freq_uLim, side='right')

    mag_data_HYB_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_YB_mag_ac_sim.csv', skiprows=1, header=None)
    mag_data_HYB = mag_data_HYB_pd.to_numpy()
    phase_data_HYB_pd = pd.read_csv('data/op_and_freq_resp_data/LDO/H_YB_phase_ac_sim.csv', skiprows=1, header=None)
    phase_data_HYB = phase_data_HYB_pd.to_numpy()
    mag_HYB = mag_data_HYB[:,1]
    phase_HYB = phase_data_HYB[:,1]
    frequencies_ac_sim_HYB = phase_data_HYB[:,0]
    ind_1_HYB = np.searchsorted(frequencies_ac_sim_HYB, freq_lLim, side='right')
    ind_2_HYB = np.searchsorted(frequencies_ac_sim_HYB, freq_uLim, side='right')




    #######################################################################################################################
    # %% Plot the bode plots for visualization

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    ###################################################################
    # %% Mathematical vs AC Simulation Plots
    FONTSIZE = 8
    LINEWIDTH = 2
    MARKERSIZE = 5
    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_QX_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQX[ind_1_HQX:ind_2_HQX], mag_HQX[ind_1_HQX:ind_2_HQX], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{QX}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([18, 24])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_QX_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQX[ind_1_HQX:ind_2_HQX], phase_HQX[ind_1_HQX:ind_2_HQX], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-20, 20])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_QX.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_QA_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQA[ind_1_HQA:ind_2_HQA], mag_HQA[ind_1_HQA:ind_2_HQA], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{QA}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-2.5, 3.5])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_QA_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQA[ind_1_HQA:ind_2_HQA], phase_HQA[ind_1_HQA:ind_2_HQA]+360, linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([160, 200])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_QA.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_QY_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQY[ind_1_HQY:ind_2_HQY], mag_HQY[ind_1_HQY:ind_2_HQY], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{QY}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([18, 24])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_QY_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HQY[ind_1_HQY:ind_2_HQY], phase_HQY[ind_1_HQY:ind_2_HQY], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([160, 200])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_QY.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_BX_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBX[ind_1_HBX:ind_2_HBX], mag_HBX[ind_1_HBX:ind_2_HBX], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{BX}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([18, 24])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_BX_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBX[ind_1_HBX:ind_2_HBX], phase_HBX[ind_1_HBX:ind_2_HBX], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-20, 20])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_BX.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_BA_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], mag_HBA[ind_1_HBA:ind_2_HBA], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{BA}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-3, 3])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_BA_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBA[ind_1_HBA:ind_2_HBA], phase_HBA[ind_1_HBA:ind_2_HBA], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-20, 20])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_BA.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_BY_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBY[ind_1_HBY:ind_2_HBY], mag_HBY[ind_1_HBY:ind_2_HBY], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{BY}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([18, 24])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_BY_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HBY[ind_1_HBY:ind_2_HBY], phase_HBY[ind_1_HBY:ind_2_HBY], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([160, 200])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_BY.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_YQ_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HYQ[ind_1_HYQ:ind_2_HYQ], mag_HYQ[ind_1_HYQ:ind_2_HYQ], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{YQ}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([43, 49])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_YQ_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HYQ[ind_1_HYQ:ind_2_HYQ], phase_HYQ[ind_1_HYQ:ind_2_HYQ], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-60, 10])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0, h_pad=0)
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_YQ.pdf', dpi=300, bbox_inches='tight')
    plt.show()



    # %%
    plt.figure(figsize=(1.7, 1.7))
    plt.subplot(2, 1, 1)
    plt.semilogx(freqs_hz_MATH, mag_YB_db_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HYB[ind_1_HYB:ind_2_HYB], mag_HYB[ind_1_HYB:ind_2_HYB], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.title(r'$\hat{\mathcal{H}}_{YB}$', fontsize = FONTSIZE)
    plt.ylabel('Mag.', fontsize = FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([44, 50])
    plt.grid(True)

    # Phase Plot
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs_hz_MATH, phase_YB_deg_MATH, linewidth=3, linestyle='-', label="LDIM")
    plt.semilogx(frequencies_ac_sim_HYB[ind_1_HYB:ind_2_HYB], phase_HYB[ind_1_HYB:ind_2_HYB], linewidth=1.5, linestyle='--', label="AC Sim")
    plt.xlabel('Frequency (Hz)', fontsize = FONTSIZE)
    plt.ylabel('Ph.', fontsize = FONTSIZE)
    plt.xlim([freq_lLim, freq_uLim])
    plt.ylim([-60, 10])
    plt.grid(True)
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
    plt.savefig(f'data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_H_YB.pdf', dpi=300, bbox_inches='tight')
    plt.show()


    # %% Combined Difference Plot (LDIM - AC Sim)
    # Evaluate the LDIM (mathematical) TFs exactly at the AC sim's own frequency points,
    # so the LDIM-AC error can be computed pointwise. H_QA's AC sim phase needs the same
    # +360 deg correction applied to it above (to undo the AC sim tool's phase wrapping)
    # before the difference is meaningful.
    diff_specs = [
        {'tex_label': 'QX', 'H_TF': H_QX_MATH, 'ac_freq': frequencies_ac_sim_HQX, 'ind_1': ind_1_HQX, 'ind_2': ind_2_HQX, 'ac_mag': mag_HQX, 'ac_phase': phase_HQX, 'phase_offset_deg': 0},
        {'tex_label': 'QA', 'H_TF': H_QA_MATH, 'ac_freq': frequencies_ac_sim_HQA, 'ind_1': ind_1_HQA, 'ind_2': ind_2_HQA, 'ac_mag': mag_HQA, 'ac_phase': phase_HQA, 'phase_offset_deg': 360},
        {'tex_label': 'QY', 'H_TF': H_QY_MATH, 'ac_freq': frequencies_ac_sim_HQY, 'ind_1': ind_1_HQY, 'ind_2': ind_2_HQY, 'ac_mag': mag_HQY, 'ac_phase': phase_HQY, 'phase_offset_deg': 0},
        {'tex_label': 'BX', 'H_TF': H_BX_MATH, 'ac_freq': frequencies_ac_sim_HBX, 'ind_1': ind_1_HBX, 'ind_2': ind_2_HBX, 'ac_mag': mag_HBX, 'ac_phase': phase_HBX, 'phase_offset_deg': 0},
        {'tex_label': 'BA', 'H_TF': H_BA_MATH, 'ac_freq': frequencies_ac_sim_HBA, 'ind_1': ind_1_HBA, 'ind_2': ind_2_HBA, 'ac_mag': mag_HBA, 'ac_phase': phase_HBA, 'phase_offset_deg': 0},
        {'tex_label': 'BY', 'H_TF': H_BY_MATH, 'ac_freq': frequencies_ac_sim_HBY, 'ind_1': ind_1_HBY, 'ind_2': ind_2_HBY, 'ac_mag': mag_HBY, 'ac_phase': phase_HBY, 'phase_offset_deg': 0},
        {'tex_label': 'YQ', 'H_TF': H_YQ_MATH, 'ac_freq': frequencies_ac_sim_HYQ, 'ind_1': ind_1_HYQ, 'ind_2': ind_2_HYQ, 'ac_mag': mag_HYQ, 'ac_phase': phase_HYQ, 'phase_offset_deg': 0},
        {'tex_label': 'YB', 'H_TF': H_YB_MATH, 'ac_freq': frequencies_ac_sim_HYB, 'ind_1': ind_1_HYB, 'ind_2': ind_2_HYB, 'ac_mag': mag_HYB, 'ac_phase': phase_HYB, 'phase_offset_deg': 0},
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
    diff_colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    diff_linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (3, 5, 1, 5))]
    diff_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    n_markers_target = 8

    def shade_from_zero(ax, freqs, values, color):
        '''Shades the region between zero and a single curve's value at each frequency,
        so the shaded area itself reads as that curve's deviation from a perfect match.'''
        ax.fill_between(freqs, 0, values, color=color, alpha=0.15, linewidth=0, zorder=0)

    fig_diff, (axd_mag, axd_phase) = plt.subplots(2, 1, figsize=(3.4, 2.8))
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
    axd_mag.legend(fontsize=FONTSIZE, loc='upper left', ncol=3, handlelength=1)

    plt.tight_layout(pad=0.5, h_pad=1.0)
    plt.savefig('data/experimental_results/plots/TF_vs_AC/LDO_LDIM_vs_AC_Diff_Combined.pdf', dpi=300, bbox_inches='tight')
    plt.show()



# Entry point of the script
if __name__ == "__main__":
    main()  