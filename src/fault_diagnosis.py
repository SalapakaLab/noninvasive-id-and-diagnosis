import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import control as ct
from scipy import signal as sig
import warnings
import scipy.linalg
from src.tf_estimation import estimate_tf_PSD, estimate_tf_PSD_single, merge_LF_HF_frequency_resp, sample_logarithmically
from src.math_models import get_simple_math_TF_biopot_sensor_python, get_math_TF_Hardware_Circuit_python, get_math_TF_5_tran_OTA_python

def reset_biopot_circ_params():
    '''
    Resets the biopotential sensor circuit parameters to their nominal values.
    This is useful to ensure that each parameter estimation/ fault diagnosis run starts from the same baseline.
    The values here are based on the Cadence simulations and the circuit diagram of the biopotential sensor.

    Returns:
    -------
    Circ_params: dictionary containing the nominal values of the circuit parameters
    '''
    Circ_params = {}
    Circ_params['Ra'] = 1e3
    Circ_params['Ca'] = 10e-6
    Circ_params['Rc'] = 1e3
    Circ_params['Cc'] = 10e-6
    Circ_params['Ro'] = 10e3
    Circ_params['Co'] = 32e-9
    Circ_params['Aa'] = 412.49
    Circ_params['Ab'] = 33.77
    Circ_params['Ac'] = 412.49
    Circ_params['Ao'] = 552.97
    
    return Circ_params



def parse_circuit_params_biopot_sensor(Circ_params):
    '''
    Parses the biopotential sensor circuit parameters from the dictionary and returns them in the order required for the TF equations.
    
    Parameters:
    ----------
    Circ_params: dictionary containing the circuit parameters
    
    Returns:
    --------
    Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co: individual circuit parameters
    '''
    Ra = Circ_params['Ra']
    Ca = Circ_params['Ca']
    Rc = Circ_params['Rc']
    Cc = Circ_params['Cc']
    Ro = Circ_params['Ro']
    Co = Circ_params['Co']
    Aa = Circ_params['Aa']
    Ab = Circ_params['Ab']
    Ac = Circ_params['Ac']
    Ao = Circ_params['Ao']

    return Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co



def get_tf_diff(H_healthy, H_faulty, adj, f):
    '''
    Computes the average magnitude and phase difference between the healthy and faulty transfer 
    functions for the edges present in the adjacency matrix.
    
    Parameters:
    ----------
    H_healthy: nNodes X nNodes X nFreqs array containing the estimated transfer functions for the healthy case
    H_faulty: nNodes X nNodes X nFreqs array containing the estimated transfer functions for the faulty case
    adj: nNodes X nNodes adjacency matrix
    f: frequency vector

    Returns:
    --------
    H_avg_mag_diff: dictionary containing the average magnitude differences
    H_avg_phase_diff: dictionary containing the average phase differences
    '''
    nNodes = adj.shape[0]
    H_avg_mag_diff = {}
    H_avg_phase_diff = {}
    
    for i in range(nNodes):
        for j in range(nNodes):
            if adj[j,i] == 1:    
                mag_diff = np.abs(H_healthy[j,i,:]) - np.abs(H_faulty[j,i,:])
                phase_diff = np.angle(H_healthy[j,i,:]) - np.angle(H_faulty[j,i,:])
                
                H_avg_mag_diff["H_"+str(j+1)+str(i+1)] = sum(abs(mag_diff))/len(f)
                H_avg_phase_diff["H_"+str(j+1)+str(i+1)] = sum(abs(phase_diff))/len(f)
                
    return H_avg_mag_diff, H_avg_phase_diff



def find_faulty_parameter_biopot_sensor(HF_data_healthy, LF_data_Fealthy, Tag_row_num, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, nFFT, Adj, freq_lLim, freq_mLim, freq_uLim):
    '''
    Identifies the potentially faulty parameter in the biopotential sensor circuit by comparing
    the estimated transfer functions from the healthy and faulty data.
    
    Parameters:
    ----------
    HF_data_healthy: numpy array containing the healthy high-frequency data
    LF_data_Fealthy: numpy array containing the healthy low-frequency data
    Tag_row_num: integer representing the row number of the faulty data
    f_samp_HF: float representing the sampling frequency for high-frequency data
    nperseg_HF: integer representing the segment length for high-frequency data
    noverlap_HF: integer representing the overlap length for high-frequency data
    win_HF: string representing the window function for high-frequency data
    f_samp_LF: float representing the sampling frequency for low-frequency data
    nperseg_LF: integer representing the segment length for low-frequency data
    noverlap_LF: integer representing the overlap length for low-frequency data
    win_LF: string representing the window function for low-frequency data
    nFFT: integer representing the number of FFT points
    Adj: nNodes X nNodes adjacency matrix
    freq_lLim: float representing the lower limit of the frequency range
    freq_mLim: float representing the middle limit of the frequency range
    freq_uLim: float representing the upper limit of the frequency range

    Returns:
    --------
    potential_faulty_params: dictionary containing the potentially faulty parameters and their estimated values
    '''
    Circ_params = reset_biopot_circ_params()
    list_of_params_to_check = ['Ra', 'Ca', 'Ro', 'Co', 'Rc', 'Cc']# 
    data_Normal_HF = HF_data_healthy
    data_Faulty_HF = pd.read_csv(
        f'data/time_series_data/bio_pot_sensor/faulty/Biopotential_Sensor_Faulty_row{Tag_row_num}_100mHz50kHz_processed_data.txt', 
        delimiter=',', 
        dtype=np.float64,
        header=None,       
        engine='c'         
    ).to_numpy()
    data_Normal_LF = LF_data_Fealthy
    data_Faulty_LF = pd.read_csv(
        f'data/time_series_data/bio_pot_sensor/faulty/Biopotential_Sensor_Faulty_row{Tag_row_num}_10mHz5kHz_processed_data.txt', 
        delimiter=',', 
        dtype=np.float64,
        header=None,       
        engine='c'         
    ).to_numpy()
    
    H_healthy_est_HF, H_faulty_est_HF, f_HF = estimate_tf_PSD(data_Normal_HF, data_Faulty_HF, Adj, nFFT, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, freq_mLim, freq_uLim)
    H_healthy_est_LF, H_faulty_est_LF, f_LF = estimate_tf_PSD(data_Normal_LF, data_Faulty_LF, Adj, nFFT, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, freq_lLim, freq_mLim)

    H_healthy_combined_linear, f_combined_linear = merge_LF_HF_frequency_resp(H_healthy_est_HF, H_healthy_est_LF, f_HF, f_LF, freq_lLim, freq_mLim, freq_uLim)
    H_faulty_combined_linear, _ = merge_LF_HF_frequency_resp(H_faulty_est_HF, H_faulty_est_LF, f_HF, f_LF, freq_lLim, freq_mLim, freq_uLim)
    
    f_combined, H_healthy_combined = sample_logarithmically(f_combined_linear, H_healthy_combined_linear, num_log_points=100)
    _, H_faulty_combined = sample_logarithmically(f_combined_linear, H_faulty_combined_linear, num_log_points=100)

    H_avg_mag_diff, _ = get_tf_diff(H_healthy_combined, H_faulty_combined, Adj, f_combined)
    
    max_mag_key = max(H_avg_mag_diff, key=H_avg_mag_diff.get)
    omega_freq_resp = 2*np.pi*f_combined
    
    min_err_param_dict = {}
    error_dict = {}
    delta_param = 0.99
    
    if max_mag_key == "H_51":
        print("Detected fault in H_oa")
        Estimated_H_faulty = H_faulty_combined[4,0,:]
        for param in list_of_params_to_check:
            Circ_params = reset_biopot_circ_params()
            param_uLim = Circ_params[param] + Circ_params[param]*delta_param
            param_lLim = Circ_params[param] - Circ_params[param]*delta_param
            param_range = np.linspace(param_lLim,param_uLim,1000)
            error_min = np.inf
            
            
            for param_instance in param_range:
                Circ_params[param] = param_instance
                Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co = parse_circuit_params_biopot_sensor(Circ_params)
                H_oa, _, _, _ = get_simple_math_TF_biopot_sensor_python(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co)
                H_oa_freq_resp = H_oa(1j*omega_freq_resp)
                error_vector = H_oa_freq_resp - Estimated_H_faulty
                error_vector_at_considered_freqs = error_vector
                error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
                
                if error_temp < error_min:
                    error_min = error_temp
                    min_err_param_dict[param] = param_instance
                    error_dict[param] = error_min
            
            print(f"lowest error with {param} = {error_min}")
                    
    if max_mag_key == "H_53":
        print("Detected fault in H_oc")
        Estimated_H_faulty = H_faulty_combined[4,2,:]
        for param in list_of_params_to_check:
            Circ_params = reset_biopot_circ_params()
            param_uLim = Circ_params[param] + Circ_params[param]*delta_param
            param_lLim = Circ_params[param] - Circ_params[param]*delta_param
            param_range = np.linspace(param_lLim,param_uLim,1000)
            error_min = np.inf
            
            
            for param_instance in param_range:
                Circ_params[param] = param_instance
                Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co = parse_circuit_params_biopot_sensor(Circ_params)
                _, H_oc, _, _ = get_simple_math_TF_biopot_sensor_python(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co)
                H_oc_freq_resp = H_oc(1j*omega_freq_resp)
                error_vector = H_oc_freq_resp - Estimated_H_faulty
                error_vector_at_considered_freqs = error_vector
                error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
                
                if error_temp < error_min:
                    error_min = error_temp
                    min_err_param_dict[param] = param_instance
                    error_dict[param] = error_min

            print(f"lowest error with {param} = {error_min}")
                    
    if max_mag_key == "H_54":
        print("Detected fault in H_od")
        Estimated_H_faulty = H_faulty_combined[4,3,:]
        for param in list_of_params_to_check:
            Circ_params = reset_biopot_circ_params()
            param_uLim = Circ_params[param] + Circ_params[param]*delta_param
            param_lLim = Circ_params[param] - Circ_params[param]*delta_param
            param_range = np.linspace(param_lLim,param_uLim,1000)
            error_min = np.inf
            
            
            for param_instance in param_range:
                Circ_params[param] = param_instance
                Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co = parse_circuit_params_biopot_sensor(Circ_params)
                _, _, H_od, _ = get_simple_math_TF_biopot_sensor_python(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co)
                H_od_freq_resp = H_od(1j*omega_freq_resp)
                error_vector = H_od_freq_resp - Estimated_H_faulty
                error_vector_at_considered_freqs = error_vector
                error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
                
                if error_temp < error_min:
                    error_min = error_temp
                    min_err_param_dict[param] = param_instance
                    error_dict[param] = error_min
            
            print(f"lowest error with {param} = {error_min}")
                    
    if max_mag_key == "H_42":
        print("Detected fault in H_db")
        Estimated_H_faulty = H_faulty_combined[3,1,:]
        for param in list_of_params_to_check:
            Circ_params = reset_biopot_circ_params()
            param_uLim = Circ_params[param] + Circ_params[param]*delta_param
            param_lLim = Circ_params[param] - Circ_params[param]*delta_param
            param_range = np.linspace(param_lLim,param_uLim,1000)
            error_min = np.inf
            
            for param_instance in param_range:
                Circ_params[param] = param_instance
                Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co = parse_circuit_params_biopot_sensor(Circ_params)
                _, _, _, H_db = get_simple_math_TF_biopot_sensor_python(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co)
                H_db_freq_resp = H_db(1j*omega_freq_resp)
                error_vector = H_db_freq_resp - Estimated_H_faulty
                error_vector_at_considered_freqs = error_vector
                error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
                
                if error_temp < error_min:
                    error_min = error_temp
                    min_err_param_dict[param] = param_instance
                    error_dict[param] = error_min
                    
            print(f"lowest error with {param} = {error_min}")
            
    potential_faulty_params = {}

    lowest_key = min(error_dict, key=error_dict.get)
    potential_faulty_params[lowest_key] = min_err_param_dict[lowest_key]

    return potential_faulty_params



def get_simple_math_TF_biopot_sensor_freq_resp(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co, s):
    '''
    Frequency-response counterpart of get_simple_math_TF_biopot_sensor_python. Instead of building
    symbolic control.TransferFunction objects and evaluating them afterwards, this substitutes the
    Laplace variable up front and evaluates the same SIMPLIFIED expressions directly in complex
    arithmetic. Building the TF objects (in particular the three ct.minreal calls) costs ~12 ms per
    parameter value while evaluating the closed form costs microseconds, so this is the form to use
    inside a parameter sweep.

    Every argument is broadcast against every other one, so a whole sweep can be evaluated in a
    single call: pass the swept parameter as a column vector of shape (nParamValues, 1) and s as a
    row vector of shape (1, nFreqs), and the returned responses have shape (nParamValues, nFreqs).

    Parameters:
    ----------
    Aa: Open loop gain of the OpAmp in the input stage a
    Ab: Open loop gain of the OpAmp in the buffer stage b
    Ac: Open loop gain of the OpAmp in the input stage c
    Ao: Open loop gain of the OpAmp in the output stage o
    Ra: Resistance value for the input stage a.
    Ca: Capacitance value for the input stage a.
    Rc: Resistance value for the input stage c.
    Cc: Capacitance value for the input stage c.
    Ro: Resistance value for the output stage.
    Co: Capacitance value for the output stage.
    s: complex Laplace variable at which to evaluate the responses, i.e. 1j*omega

    Returns:
    --------
    A tuple containing the frequency responses of H_oa, H_oc, H_od and H_db, each broadcast to the
    common shape of the inputs.
    '''
    Za = Ra + 1/(s*Ca)
    Zc = Rc + 1/(s*Cc)
    Zo = 1/(1/Ro + s*Co)
    As = Ao/(1/Za + 1/Zc + 1/Zo)

    # Shared by H_oa, H_oc and H_od, so it is worth forming only once
    loop_term = 1 + As/Zo

    H_oa = -(Aa*As)/((1+Aa)*loop_term*Za)
    H_oc = -(Ac*As)/((1+Ac)*loop_term*Zc)
    H_od = Ao/loop_term
    # H_db is frequency independent; the zeros give it the same shape as the others
    H_db = Ab/(1+Ab) + np.zeros_like(H_od)

    return H_oa, H_oc, H_od, H_db



def estimate_combined_freq_resp_biopot_sensor(data_HF, data_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, nFFT, Adj, freq_lLim, freq_mLim, freq_uLim):
    '''
    Estimates the transfer functions of one biopotential sensor dataset and returns them on the
    logarithmically sampled frequency grid used for fault diagnosis, i.e. it runs the PSD based
    estimation on the HF and LF recordings, merges the two bands and resamples the result.

    Parameters:
    ----------
    data_HF: numpy array containing the high-frequency data
    data_LF: numpy array containing the low-frequency data
    f_samp_HF: float representing the sampling frequency for high-frequency data
    nperseg_HF: integer representing the segment length for high-frequency data
    noverlap_HF: integer representing the overlap length for high-frequency data
    win_HF: string representing the window function for high-frequency data
    f_samp_LF: float representing the sampling frequency for low-frequency data
    nperseg_LF: integer representing the segment length for low-frequency data
    noverlap_LF: integer representing the overlap length for low-frequency data
    win_LF: string representing the window function for low-frequency data
    nFFT: integer representing the number of FFT points
    Adj: nNodes X nNodes adjacency matrix
    freq_lLim: float representing the lower limit of the frequency range
    freq_mLim: float representing the middle limit of the frequency range
    freq_uLim: float representing the upper limit of the frequency range

    Returns:
    --------
    f_combined: numpy array containing the logarithmically sampled frequency grid
    H_combined: nNodes X nNodes X len(f_combined) array containing the estimated transfer functions
    '''
    H_est_HF, f_HF = estimate_tf_PSD_single(data_HF, Adj, nFFT, f_samp_HF, nperseg_HF, noverlap_HF, win_HF)
    H_est_LF, f_LF = estimate_tf_PSD_single(data_LF, Adj, nFFT, f_samp_LF, nperseg_LF, noverlap_LF, win_LF)

    H_combined_linear, f_combined_linear = merge_LF_HF_frequency_resp(H_est_HF, H_est_LF, f_HF, f_LF, freq_lLim, freq_mLim, freq_uLim)
    f_combined, H_combined = sample_logarithmically(f_combined_linear, H_combined_linear, num_log_points=100)

    return f_combined, H_combined



def find_faulty_parameter_biopot_sensor_vectorized(HF_data_healthy, LF_data_Fealthy, Tag_row_num, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, nFFT, Adj, freq_lLim, freq_mLim, freq_uLim, nGridPoints=1000, healthy_freq_resp=None):
    '''
    Vectorized fault diagnosis function. It identifies the potentially
    faulty parameter in the biopotential sensor circuit.

    Parameters:
    ----------
    HF_data_healthy: numpy array containing the healthy high-frequency data (ignored when healthy_freq_resp is given)
    LF_data_Fealthy: numpy array containing the healthy low-frequency data (ignored when healthy_freq_resp is given)
    Tag_row_num: integer representing the row number of the faulty data
    f_samp_HF: float representing the sampling frequency for high-frequency data
    nperseg_HF: integer representing the segment length for high-frequency data
    noverlap_HF: integer representing the overlap length for high-frequency data
    win_HF: string representing the window function for high-frequency data
    f_samp_LF: float representing the sampling frequency for low-frequency data
    nperseg_LF: integer representing the segment length for low-frequency data
    noverlap_LF: integer representing the overlap length for low-frequency data
    win_LF: string representing the window function for low-frequency data
    nFFT: integer representing the number of FFT points
    Adj: nNodes X nNodes adjacency matrix
    freq_lLim: float representing the lower limit of the frequency range
    freq_mLim: float representing the middle limit of the frequency range
    freq_uLim: float representing the upper limit of the frequency range
    nGridPoints: integer representing the number of grid points searched per parameter
    healthy_freq_resp: optional (f_combined, H_healthy_combined) tuple as returned by
        estimate_combined_freq_resp_biopot_sensor. When None the healthy response is estimated
        from HF_data_healthy and LF_data_Fealthy.

    Returns:
    --------
    potential_faulty_params: dictionary containing the potentially faulty parameters and their estimated values
    '''
    list_of_params_to_check = ['Ra', 'Ca', 'Ro', 'Co', 'Rc', 'Cc']
    data_Faulty_HF = pd.read_csv(
        f'data/time_series_data/bio_pot_sensor/faulty/Biopotential_Sensor_Faulty_row{Tag_row_num}_100mHz50kHz_processed_data.txt',
        delimiter=',',
        dtype=np.float64,
        header=None,
        engine='c'
    ).to_numpy()
    data_Faulty_LF = pd.read_csv(
        f'data/time_series_data/bio_pot_sensor/faulty/Biopotential_Sensor_Faulty_row{Tag_row_num}_10mHz5kHz_processed_data.txt',
        delimiter=',',
        dtype=np.float64,
        header=None,
        engine='c'
    ).to_numpy()

    # Reuse the caller's healthy estimate when it has one, since it is identical for every row
    if healthy_freq_resp is None:
        f_combined, H_healthy_combined = estimate_combined_freq_resp_biopot_sensor(HF_data_healthy, LF_data_Fealthy, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, nFFT, Adj, freq_lLim, freq_mLim, freq_uLim)
    else:
        f_combined, H_healthy_combined = healthy_freq_resp

    _, H_faulty_combined = estimate_combined_freq_resp_biopot_sensor(data_Faulty_HF, data_Faulty_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, nFFT, Adj, freq_lLim, freq_mLim, freq_uLim)

    H_avg_mag_diff, _ = get_tf_diff(H_healthy_combined, H_faulty_combined, Adj, f_combined)

    max_mag_key = max(H_avg_mag_diff, key=H_avg_mag_diff.get)
    omega_freq_resp = 2*np.pi*f_combined

    tf_selection = {
        "H_51": (4, 0, 0, "H_oa"),
        "H_53": (4, 2, 1, "H_oc"),
        "H_54": (4, 3, 2, "H_od"),
        "H_42": (3, 1, 3, "H_db"),
    }

    min_err_param_dict = {}
    error_dict = {}
    delta_param = 0.99

    if max_mag_key in tf_selection:
        out_node, in_node, tf_index, tf_label = tf_selection[max_mag_key]
        print(f"Detected fault in {tf_label}")
        Estimated_H_faulty = H_faulty_combined[out_node, in_node, :]

        # Row vector over frequency, so that it broadcasts against the column of grid values below
        s_freq_resp = 1j*omega_freq_resp[np.newaxis, :]

        for param in list_of_params_to_check:
            Circ_params = reset_biopot_circ_params()
            param_uLim = Circ_params[param] + Circ_params[param]*delta_param
            param_lLim = Circ_params[param] - Circ_params[param]*delta_param
            param_range = np.linspace(param_lLim, param_uLim, nGridPoints)

            # The swept parameter enters as a column, so the responses come out as
            # (nGridPoints, nFreqs) and the whole grid is evaluated in one shot
            Circ_params[param] = param_range[:, np.newaxis]
            Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co = parse_circuit_params_biopot_sensor(Circ_params)
            H_freq_resp = get_simple_math_TF_biopot_sensor_freq_resp(Aa, Ab, Ac, Ao, Ra, Ca, Rc, Cc, Ro, Co, s_freq_resp)[tf_index]

            error_per_grid_point = np.mean(np.abs(H_freq_resp - Estimated_H_faulty[np.newaxis, :]), axis=1)
            min_err_index = np.argmin(error_per_grid_point)
            error_min = error_per_grid_point[min_err_index]

            min_err_param_dict[param] = param_range[min_err_index]
            error_dict[param] = error_min

            print(f"lowest error with {param} = {error_min}")

    potential_faulty_params = {}

    lowest_key = min(error_dict, key=error_dict.get)
    potential_faulty_params[lowest_key] = min_err_param_dict[lowest_key]

    return potential_faulty_params



def stastical_validation_of_fault_diag_biopot_sensor(Tag_path, start_row, end_row, data_Normal_HF, data_Normal_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim):
    '''
    Performs statistical validation of the fault diagnosis algorithm for the biopotential sensor by 
    iterating through multiple faulty scenarios and comparing the identified faulty parameters with 
    the ground truth from the Tag file.

    Parameters:
    Tag_path: string representing the path to the Tag file
    start_row: integer representing the starting row number for validation
    end_row: integer representing the ending row number for validation
    data_Normal_HF: numpy array containing the healthy high-frequency data
    data_Normal_LF: numpy array containing the healthy low-frequency data
    f_samp_HF: float representing the sampling frequency for high-frequency data
    nperseg_HF: integer representing the segment length for high-frequency data
    noverlap_HF: integer representing the overlap length for high-frequency data
    win_HF: string representing the window function for high-frequency data
    f_samp_LF: float representing the sampling frequency for low-frequency data
    nperseg_LF: integer representing the segment length for low-frequency data
    noverlap_LF: integer representing the overlap length for low-frequency data
    win_LF: string representing the window function for low-frequency data
    n_FFT: integer representing the number of FFT points
    Adj: nNodes X nNodes adjacency matrix
    freq_lLim: float representing the lower limit of the frequency range
    freq_mLim: float representing the middle limit of the frequency range
    freq_uLim: float representing the upper limit of the frequency range

    Returns:
    --------
    confusion_matrices: dictionary containing confusion matrices for each scenario
    resistance_est_error: list containing estimated errors for resistance parameters
    capacitance_est_error: list containing estimated errors for capacitance parameters
    '''
    Tag_file = pd.read_csv(Tag_path)
    param_est_err_dict = {'Ra':[], 'Ca':[], 'Ro':[], 'Co':[], 'Rc':[], 'Cc':[]}
    confusion_matrices = {}
    
    for row_num in range(start_row,end_row+1):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        potential_faulty = find_faulty_parameter_biopot_sensor(data_Normal_HF, data_Normal_LF, row_num, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim)
        conponent_str_raw = Tag_file.loc[row_num-2,"Component ID"]
        faulty_component = conponent_str_raw[:-4]
        faulty_component_value = Tag_file.loc[row_num-2,"Value"]
        print(f"Row Num = {row_num}")
        print(f"Potential Faulty Params = {potential_faulty}")
        
        if faulty_component in potential_faulty.keys():
            true_positive += 1
            est_error = (potential_faulty[faulty_component] - faulty_component_value)/faulty_component_value
            param_est_err_dict[faulty_component].append(est_error)
            print(f"Faulty Component is {faulty_component} with value {faulty_component_value}")
            print(f"Estimation error = {est_error}")
            print("==========================================")

        for component in param_est_err_dict.keys():
            if component not in potential_faulty.keys():
                if component != faulty_component:
                    true_negative += 1
                if component == faulty_component:
                    false_negative += 1
            if component in potential_faulty.keys():
                if component != faulty_component:
                    false_positive += 1
            
        confusion_matrix = {}
        confusion_matrix['TP'] = true_positive
        confusion_matrix['FP'] = false_positive
        confusion_matrix['TN'] = true_negative
        confusion_matrix['FN'] = false_negative

        confusion_matrices[f'row{row_num}'] = confusion_matrix

    

    resistance_est_error = param_est_err_dict['Ra'] + param_est_err_dict['Rc'] + param_est_err_dict['Ro']
    capacitance_est_error = param_est_err_dict['Ca'] + param_est_err_dict['Cc'] + param_est_err_dict['Co']

    return confusion_matrices, resistance_est_error, capacitance_est_error



def stastical_validation_of_fault_diag_biopot_sensor_vectorized(Tag_path, start_row, end_row, data_Normal_HF, data_Normal_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim, nGridPoints=10000):
    '''
    Performs statistical validation of the fault diagnosis algorithm for the biopotential sensor by 
    iterating through multiple faulty scenarios and comparing the identified faulty parameters with 
    the ground truth from the Tag file.

    Parameters:
    Tag_path: string representing the path to the Tag file
    start_row: integer representing the starting row number for validation
    end_row: integer representing the ending row number for validation
    data_Normal_HF: numpy array containing the healthy high-frequency data
    data_Normal_LF: numpy array containing the healthy low-frequency data
    f_samp_HF: float representing the sampling frequency for high-frequency data
    nperseg_HF: integer representing the segment length for high-frequency data
    noverlap_HF: integer representing the overlap length for high-frequency data
    win_HF: string representing the window function for high-frequency data
    f_samp_LF: float representing the sampling frequency for low-frequency data
    nperseg_LF: integer representing the segment length for low-frequency data
    noverlap_LF: integer representing the overlap length for low-frequency data
    win_LF: string representing the window function for low-frequency data
    n_FFT: integer representing the number of FFT points
    Adj: nNodes X nNodes adjacency matrix
    freq_lLim: float representing the lower limit of the frequency range
    freq_mLim: float representing the middle limit of the frequency range
    freq_uLim: float representing the upper limit of the frequency range
    nGridPoints: integer representing the number of grid points searched per parameter

    Returns:
    --------
    confusion_matrices: dictionary containing confusion matrices for each scenario
    resistance_est_error: list containing estimated errors for resistance parameters
    capacitance_est_error: list containing estimated errors for capacitance parameters
    '''
    Tag_file = pd.read_csv(Tag_path)
    param_est_err_dict = {'Ra':[], 'Ca':[], 'Ro':[], 'Co':[], 'Rc':[], 'Cc':[]}
    confusion_matrices = {}

    # The healthy recordings are the same for every row, so their response is estimated once here
    # and handed to every call below instead of being re-estimated inside the loop
    healthy_freq_resp = estimate_combined_freq_resp_biopot_sensor(data_Normal_HF, data_Normal_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim)

    for row_num in range(start_row,end_row+1):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        potential_faulty = find_faulty_parameter_biopot_sensor_vectorized(data_Normal_HF, data_Normal_LF, row_num, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim, nGridPoints=nGridPoints, healthy_freq_resp=healthy_freq_resp)
        conponent_str_raw = Tag_file.loc[row_num-2,"Component ID"]
        faulty_component = conponent_str_raw[:-4]
        faulty_component_value = Tag_file.loc[row_num-2,"Value"]
        print(f"Row Num = {row_num}")
        print(f"Potential Faulty Params = {potential_faulty}")

        if faulty_component in potential_faulty.keys():
            true_positive += 1
            est_error = (potential_faulty[faulty_component] - faulty_component_value)/faulty_component_value
            param_est_err_dict[faulty_component].append(est_error)
            print(f"Faulty Component is {faulty_component} with value {faulty_component_value}")
            print(f"Estimation error = {est_error}")
            print("==========================================")

        for component in param_est_err_dict.keys():
            if component not in potential_faulty.keys():
                if component != faulty_component:
                    true_negative += 1
                if component == faulty_component:
                    false_negative += 1
            if component in potential_faulty.keys():
                if component != faulty_component:
                    false_positive += 1

        confusion_matrix = {}
        confusion_matrix['TP'] = true_positive
        confusion_matrix['FP'] = false_positive
        confusion_matrix['TN'] = true_negative
        confusion_matrix['FN'] = false_negative

        confusion_matrices[f'row{row_num}'] = confusion_matrix

    resistance_est_error = param_est_err_dict['Ra'] + param_est_err_dict['Rc'] + param_est_err_dict['Ro']
    capacitance_est_error = param_est_err_dict['Ca'] + param_est_err_dict['Cc'] + param_est_err_dict['Co']

    return confusion_matrices, resistance_est_error, capacitance_est_error



def report_diagnostic_results_biopot_sensor(confusion_matrices, save_dir=None):
    """
    Takes the dictionary of confusion matrices outputted by the validation function
    and generates professional aggregate reports and plots.

    Parameters:
    -----------
    confusion_matrices: dictionary containing confusion matrices for each scenario
    save_dir: optional string path of a directory to save the confusion matrix and
        precision/recall/F1 box plot PDFs into (created if it doesn't exist). If None, the
        plots are only shown, not saved.

    Returns:
    --------
    None
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    records = []
    
    # Variables to hold the global aggregate counts
    agg_TP = agg_FP = agg_TN = agg_FN = 0
    
    # Process each scenario
    for scenario_name, cm in confusion_matrices.items():
        tp = cm['TP']
        fp = cm['FP']
        tn = cm['TN']
        fn = cm['FN']
        
        # Add to global totals
        agg_TP += tp
        agg_FP += fp
        agg_TN += tn
        agg_FN += fn
        
        # Calculate metrics for this specific scenario
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        records.append({
            'Scenario': scenario_name,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1
        })
        
    df_metrics = pd.DataFrame(records)
    
    #########################################################################
    # Normalized Aggregate Confusion Matrix
    matrix_raw = np.array([
        [agg_TP, agg_FN],  # Row 1: Actual Faults
        [agg_FP, agg_TN]   # Row 2: Actual Normal
    ])
    
    # Normalize by row sums (True Labels) to get percentages
    row_sums = matrix_raw.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix_raw, row_sums, out=np.zeros_like(matrix_raw, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(1.7, 1.7))
    sns.heatmap(matrix_norm, annot=True, fmt='.1%', cmap='Blues', cbar=False,
                annot_kws={'size': FONTSIZE},
                xticklabels=['Faulty', 'Non-faulty'],
                yticklabels=['Faulty', 'Non-faulty'])
    plt.xlabel('Predicted Component State', fontsize=FONTSIZE)
    plt.ylabel('Actual Component State', fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.tight_layout(pad=0)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'bio_pot_sensor_fault_diag_confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    #########################################################################
    # Box Plot of Performance Metrics
    # Melt the dataframe so seaborn can plot multiple boxes easily
    df_melt = df_metrics.melt(id_vars='Scenario',
                              value_vars=['Precision', 'Recall', 'F1_Score'],
                              var_name='Metric',
                              value_name='Score')
    df_melt['Metric'] = df_melt['Metric'].replace('F1_Score', 'F1')
    
    plt.figure(figsize=(1.7, 1.7))

    # Draw the boxplot
    sns.boxplot(x='Metric', y='Score', data=df_melt, palette="Set2", showfliers=False,
                linewidth=0.8)
    # Add dots for individual scenarios to see the exact spread/outliers
    sns.stripplot(x='Metric', y='Score', data=df_melt, color='black', alpha=0.5, jitter=True,
                  size=3)

    plt.xlabel('')
    plt.ylabel('Score', fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE, rotation=30, ha='right')
    plt.yticks(fontsize=FONTSIZE)
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=0)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'bio_pot_sensor_fault_diag_prf_boxplot.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

    #########################################################################
    # Statistical Summary
    print("\n" + "="*50)
    print(" ALGORITHM PERFORMANCE SUMMARY")
    print("="*50)
    
    # Calculate Mean, Std, Min, Max
    summary_df = df_metrics[['Precision', 'Recall', 'F1_Score']].agg(['mean', 'std', 'min', 'max']).T
    # Format the table cleanly
    summary_df = summary_df.round(3)
    print(summary_df.to_string())
    print("="*50 + "\n")



def reset_hardware_circ_params():
    '''
    Resets the hardware circuit parameters to their nominal values.
    This is useful to ensure that each parameter estimation/ fault diagnosis run starts from the same baseline.
    The values here are based on the Cadence simulations and the circuit diagram of the biopotential sensor.

    Returns:
    -------
    circuit_params: dictionary containing the nominal values of the circuit parameters
    '''

    # DC Operating point data from Cadence Simulation
    # Make sure the device manes are entered properly.
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

    return circuit_params



def H31_TF_frequency_response_from_circuit_params_hardware(circuit_params, frequencies_hz, f_samp):
    '''
    Computes the frequency response of the H_31 transfer function for the hardware circuit based on the given circuit parameters.
    
    Parameters:
    -----------
    circuit_params: dictionary containing the circuit parameters
    frequencies_hz: numpy array containing the frequencies for which the response is estimated
    f_samp: float representing the sampling frequency

    Returns:
    --------
    H_31: numpy array containing the estimated frequency response
    '''
    R1 = circuit_params['R1']
    R2 = circuit_params['R2']
    R5 = circuit_params['R5']
    R6 = circuit_params['R6']
    R7 = circuit_params['R7']
    R8 = circuit_params['R8']
    R9 = circuit_params['R9']
    R10 = circuit_params['R10']
    R11 = circuit_params['R11']
    R12 = circuit_params['R12']
    C2 = circuit_params['C2']
    C3 = circuit_params['C3']
    C4 = circuit_params['C4']
    gm2 = circuit_params['gm2']
    rds2 = circuit_params['rds2']
    gm4 = circuit_params['gm4']
    rds4 = circuit_params['rds4']
    C6 = circuit_params['C6']
    R18 = circuit_params['R18']

    num_31_coeffs = [-C3*C4*C6*R8*R9*R10*R12*R18*gm2*rds2*rds4*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7), - (C4*C6*R9*R10*R12*R18*gm2*rds2*rds4 + C3*C4*R8*R9*R10*R12*gm2*rds2*(R18 + rds4 + R18*gm4*rds4))*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7) - C3*C4*C6*R8*R9*R10*R12*R18*gm2*rds2*rds4*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7), - (C4*C6*R9*R10*R12*R18*gm2*rds2*rds4 + C3*C4*R8*R9*R10*R12*gm2*rds2*(R18 + rds4 + R18*gm4*rds4))*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7) - C4*R9*R10*R12*gm2*rds2*(R18 + rds4 + R18*gm4*rds4)*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7), -C4*R9*R10*R12*gm2*rds2*(R18 + rds4 + R18*gm4*rds4)*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7), 0]
    den_31_coeffs = [(C4*R9*R10*R11 + C4*R9*R10*R12 + C4*R9*R11*R12 + C4*R10*R11*R12)*(C2*C3*C6*R5*R6**2*R7*R8*R18*rds2 + C2*C3*C6*R5*R6**2*R7*R8*R18*rds4 + C2*C3*C6*R5*R6**2*R8*R18*rds2*rds4 + C2*C3*C6*R6**2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R5*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R5*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R6*R7*R8*R18*rds2*rds4), (R9*R11 + R9*R12 + R10*R11 + R10*R12)*(C2*C3*C6*R5*R6**2*R7*R8*R18*rds2 + C2*C3*C6*R5*R6**2*R7*R8*R18*rds4 + C2*C3*C6*R5*R6**2*R8*R18*rds2*rds4 + C2*C3*C6*R6**2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R5*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R5*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R6*R7*R8*R18*rds2*rds4) + (C4*R9*R10*R11 + C4*R9*R10*R12 + C4*R9*R11*R12 + C4*R10*R11*R12)*(C2*C3*R5*R6**2*R8*rds2*rds4 + C2*C3*R6**2*R7*R8*rds2*rds4 + C2*C6*R5*R6**2*R18*rds2*rds4 + C2*C6*R6**2*R7*R18*rds2*rds4 + C2*C3*R5*R6**2*R7*R8*R18 + C2*C6*R5*R6**2*R7*R8*R18 + C2*C3*R5*R6**2*R7*R8*rds2 + C2*C3*R5*R6**2*R7*R8*rds4 + C2*C3*R5*R6**2*R8*R18*rds2 + C2*C3*R6**2*R7*R8*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds4 + C2*C6*R5*R6**2*R8*R18*rds4 + C2*C6*R6**2*R7*R8*R18*rds4 + C2*C3*R1*R2*R5*R8*rds2*rds4 + C2*C3*R1*R2*R6*R8*rds2*rds4 + C2*C3*R1*R2*R7*R8*rds2*rds4 + C2*C3*R1*R5*R6*R8*rds2*rds4 + C2*C3*R1*R6*R7*R8*rds2*rds4 + C2*C6*R1*R2*R5*R18*rds2*rds4 + C2*C6*R1*R2*R6*R18*rds2*rds4 + C2*C6*R1*R2*R7*R18*rds2*rds4 + C2*C6*R1*R5*R6*R18*rds2*rds4 + C2*C6*R1*R6*R7*R18*rds2*rds4 + C3*C6*R1*R5*R8*R18*rds2*rds4 + C3*C6*R1*R6*R8*R18*rds2*rds4 + C3*C6*R2*R5*R8*R18*rds2*rds4 + C3*C6*R1*R7*R8*R18*rds2*rds4 + C3*C6*R2*R6*R8*R18*rds2*rds4 + C3*C6*R2*R7*R8*R18*rds2*rds4 + C2*C3*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R5*R6*R7*R8*R18 + C2*C6*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R5*R6*R7*R8*R18 + C2*C3*R1*R2*R5*R7*R8*rds2 + C2*C3*R1*R2*R6*R7*R8*rds2 + C2*C3*R1*R2*R5*R7*R8*rds4 + C2*C3*R1*R2*R6*R7*R8*rds4 + C2*C3*R1*R5*R6*R7*R8*rds2 + C2*C3*R1*R5*R6*R7*R8*rds4 + C2*C3*R1*R2*R5*R8*R18*rds2 + C2*C3*R1*R2*R6*R8*R18*rds2 + C2*C3*R1*R2*R7*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds2 + C2*C6*R1*R2*R6*R7*R18*rds2 + C2*C3*R1*R5*R6*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds4 + C2*C6*R1*R2*R5*R8*R18*rds4 + C2*C6*R1*R2*R6*R7*R18*rds4 + C2*C3*R1*R6*R7*R8*R18*rds2 + C2*C6*R1*R2*R6*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds2 + C2*C6*R1*R2*R7*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds4 + C2*C6*R1*R5*R6*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds2 + C3*C6*R2*R5*R7*R8*R18*rds2 + C2*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R2*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds4 + C2*C6*R5*R6**2*R7*R8*R18*gm2*rds2 + C2*C3*R5*R6**2*R7*R8*R18*gm4*rds4 + C2*C3*R5*R6**2*R8*R18*gm4*rds2*rds4 + C2*C6*R5*R6**2*R8*R18*gm2*rds2*rds4 + C2*C3*R6**2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R6**2*R7*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R5*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R5*R7*R8*R18*gm4*rds4 + C2*C6*R1*R2*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R6*R7*R8*R18*gm4*rds4 + C2*C6*R1*R5*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R5*R6*R7*R8*R18*gm4*rds4 + C2*C3*R1*R2*R5*R8*R18*gm4*rds2*rds4 + C2*C3*R1*R2*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R5*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R6*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R7*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R5*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R5*R6*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R6*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R6*R7*R8*R18*gm2*rds2*rds4), (C4*R9*R10*R11 + C4*R9*R10*R12 + C4*R9*R11*R12 + C4*R10*R11*R12)*(C2*R5*R6**2*R7*R8 + C2*R5*R6**2*R7*R18 + C2*R5*R6**2*R8*R18 + C2*R6**2*R7*R8*R18 + C2*R5*R6**2*R7*rds2 + C2*R5*R6**2*R7*rds4 + C2*R5*R6**2*R8*rds4 + C2*R6**2*R7*R8*rds4 + C2*R5*R6**2*R18*rds2 + C2*R6**2*R7*R18*rds2 + C2*R5*R6**2*rds2*rds4 + C2*R6**2*R7*rds2*rds4 + C2*R1*R2*R5*R7*R8 + C2*R1*R2*R6*R7*R8 + C2*R1*R5*R6*R7*R8 + C2*R1*R2*R5*R7*R18 + C2*R1*R2*R5*R8*R18 + C2*R1*R2*R6*R7*R18 + C2*R1*R2*R6*R8*R18 + C2*R1*R2*R7*R8*R18 + C2*R1*R5*R6*R7*R18 + C2*R1*R5*R6*R8*R18 + C2*R1*R6*R7*R8*R18 + C3*R1*R5*R7*R8*R18 + C3*R1*R6*R7*R8*R18 + C3*R2*R5*R7*R8*R18 + C3*R2*R6*R7*R8*R18 + C6*R1*R5*R7*R8*R18 + C6*R1*R6*R7*R8*R18 + C6*R2*R5*R7*R8*R18 + C6*R2*R6*R7*R8*R18 + C2*R1*R2*R5*R7*rds2 + C2*R1*R2*R6*R7*rds2 + C2*R1*R2*R5*R7*rds4 + C2*R1*R2*R5*R8*rds4 + C2*R1*R2*R6*R7*rds4 + C2*R1*R2*R6*R8*rds4 + C2*R1*R5*R6*R7*rds2 + C2*R1*R2*R7*R8*rds4 + C2*R1*R5*R6*R7*rds4 + C2*R1*R5*R6*R8*rds4 + C3*R1*R5*R7*R8*rds2 + C3*R1*R6*R7*R8*rds2 + C3*R2*R5*R7*R8*rds2 + C2*R1*R6*R7*R8*rds4 + C3*R1*R5*R7*R8*rds4 + C3*R2*R6*R7*R8*rds2 + C3*R1*R6*R7*R8*rds4 + C3*R2*R5*R7*R8*rds4 + C2*R1*R2*R5*R18*rds2 + C3*R2*R6*R7*R8*rds4 + C2*R1*R2*R6*R18*rds2 + C2*R1*R2*R7*R18*rds2 + C2*R1*R5*R6*R18*rds2 + C2*R1*R6*R7*R18*rds2 + C3*R1*R5*R8*R18*rds2 + C3*R1*R6*R8*R18*rds2 + C3*R2*R5*R8*R18*rds2 + C3*R1*R7*R8*R18*rds2 + C3*R2*R6*R8*R18*rds2 + C6*R1*R5*R7*R18*rds2 + C3*R2*R7*R8*R18*rds2 + C6*R1*R6*R7*R18*rds2 + C6*R2*R5*R7*R18*rds2 + C6*R1*R5*R7*R18*rds4 + C6*R2*R6*R7*R18*rds2 + C6*R1*R5*R8*R18*rds4 + C6*R1*R6*R7*R18*rds4 + C6*R2*R5*R7*R18*rds4 + C6*R1*R6*R8*R18*rds4 + C6*R2*R5*R8*R18*rds4 + C6*R2*R6*R7*R18*rds4 + C6*R1*R7*R8*R18*rds4 + C6*R2*R6*R8*R18*rds4 + C6*R2*R7*R8*R18*rds4 + C2*R1*R2*R5*rds2*rds4 + C2*R1*R2*R6*rds2*rds4 + C2*R1*R2*R7*rds2*rds4 + C2*R1*R5*R6*rds2*rds4 + C2*R1*R6*R7*rds2*rds4 + C3*R1*R5*R8*rds2*rds4 + C3*R1*R6*R8*rds2*rds4 + C3*R2*R5*R8*rds2*rds4 + C3*R1*R7*R8*rds2*rds4 + C3*R2*R6*R8*rds2*rds4 + C3*R2*R7*R8*rds2*rds4 + C6*R1*R5*R18*rds2*rds4 + C6*R1*R6*R18*rds2*rds4 + C6*R2*R5*R18*rds2*rds4 + C6*R1*R7*R18*rds2*rds4 + C6*R2*R6*R18*rds2*rds4 + C6*R2*R7*R18*rds2*rds4 + C2*R5*R6**2*R7*R8*gm2*rds2 + C2*R5*R6**2*R8*R18*gm2*rds2 + C2*R6**2*R7*R8*R18*gm2*rds2 + C2*R5*R6**2*R7*R18*gm4*rds4 + C2*R5*R6**2*R8*R18*gm4*rds4 + C2*R6**2*R7*R8*R18*gm4*rds4 + C2*R5*R6**2*R8*gm2*rds2*rds4 + C2*R6**2*R7*R8*gm2*rds2*rds4 + C2*R5*R6**2*R18*gm4*rds2*rds4 + C2*R6**2*R7*R18*gm4*rds2*rds4 + C2*R1*R2*R5*R7*R8*gm2*rds2 + C2*R1*R2*R6*R7*R8*gm2*rds2 + C2*R1*R5*R6*R7*R8*gm2*rds2 + C2*R1*R2*R5*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm2*rds2 + C2*R1*R2*R7*R8*R18*gm2*rds2 + C2*R1*R2*R5*R7*R18*gm4*rds4 + C2*R1*R2*R5*R8*R18*gm4*rds4 + C2*R1*R2*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm4*rds4 + C2*R1*R2*R7*R8*R18*gm4*rds4 + C2*R1*R6*R7*R8*R18*gm2*rds2 + C2*R1*R5*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm4*rds4 + C6*R1*R5*R7*R8*R18*gm2*rds2 + C2*R1*R6*R7*R8*R18*gm4*rds4 + C3*R1*R5*R7*R8*R18*gm4*rds4 + C6*R1*R6*R7*R8*R18*gm2*rds2 + C6*R2*R5*R7*R8*R18*gm2*rds2 + C3*R1*R6*R7*R8*R18*gm4*rds4 + C3*R2*R5*R7*R8*R18*gm4*rds4 + C6*R2*R6*R7*R8*R18*gm2*rds2 + C3*R2*R6*R7*R8*R18*gm4*rds4 + C2*R1*R2*R5*R8*gm2*rds2*rds4 + C2*R1*R2*R6*R8*gm2*rds2*rds4 + C2*R1*R2*R7*R8*gm2*rds2*rds4 + C2*R1*R5*R6*R8*gm2*rds2*rds4 + C2*R1*R6*R7*R8*gm2*rds2*rds4 + C2*R1*R2*R5*R18*gm4*rds2*rds4 + C2*R1*R2*R6*R18*gm4*rds2*rds4 + C2*R1*R2*R7*R18*gm4*rds2*rds4 + C2*R1*R5*R6*R18*gm4*rds2*rds4 + C2*R1*R6*R7*R18*gm4*rds2*rds4 + C3*R1*R5*R8*R18*gm4*rds2*rds4 + C3*R1*R6*R8*R18*gm4*rds2*rds4 + C3*R2*R5*R8*R18*gm4*rds2*rds4 + C6*R1*R5*R8*R18*gm2*rds2*rds4 + C3*R1*R7*R8*R18*gm4*rds2*rds4 + C3*R2*R6*R8*R18*gm4*rds2*rds4 + C6*R1*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R5*R8*R18*gm2*rds2*rds4 + C3*R2*R7*R8*R18*gm4*rds2*rds4 + C6*R1*R7*R8*R18*gm2*rds2*rds4 + C6*R2*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R7*R8*R18*gm2*rds2*rds4 + C2*R5*R6**2*R8*R18*gm2*gm4*rds2*rds4 + C2*R6**2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R5*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R5*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R6*R7*R8*R18*gm2*gm4*rds2*rds4) + (R9*R11 + R9*R12 + R10*R11 + R10*R12)*(C2*C3*R5*R6**2*R8*rds2*rds4 + C2*C3*R6**2*R7*R8*rds2*rds4 + C2*C6*R5*R6**2*R18*rds2*rds4 + C2*C6*R6**2*R7*R18*rds2*rds4 + C2*C3*R5*R6**2*R7*R8*R18 + C2*C6*R5*R6**2*R7*R8*R18 + C2*C3*R5*R6**2*R7*R8*rds2 + C2*C3*R5*R6**2*R7*R8*rds4 + C2*C3*R5*R6**2*R8*R18*rds2 + C2*C3*R6**2*R7*R8*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds4 + C2*C6*R5*R6**2*R8*R18*rds4 + C2*C6*R6**2*R7*R8*R18*rds4 + C2*C3*R1*R2*R5*R8*rds2*rds4 + C2*C3*R1*R2*R6*R8*rds2*rds4 + C2*C3*R1*R2*R7*R8*rds2*rds4 + C2*C3*R1*R5*R6*R8*rds2*rds4 + C2*C3*R1*R6*R7*R8*rds2*rds4 + C2*C6*R1*R2*R5*R18*rds2*rds4 + C2*C6*R1*R2*R6*R18*rds2*rds4 + C2*C6*R1*R2*R7*R18*rds2*rds4 + C2*C6*R1*R5*R6*R18*rds2*rds4 + C2*C6*R1*R6*R7*R18*rds2*rds4 + C3*C6*R1*R5*R8*R18*rds2*rds4 + C3*C6*R1*R6*R8*R18*rds2*rds4 + C3*C6*R2*R5*R8*R18*rds2*rds4 + C3*C6*R1*R7*R8*R18*rds2*rds4 + C3*C6*R2*R6*R8*R18*rds2*rds4 + C3*C6*R2*R7*R8*R18*rds2*rds4 + C2*C3*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R5*R6*R7*R8*R18 + C2*C6*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R5*R6*R7*R8*R18 + C2*C3*R1*R2*R5*R7*R8*rds2 + C2*C3*R1*R2*R6*R7*R8*rds2 + C2*C3*R1*R2*R5*R7*R8*rds4 + C2*C3*R1*R2*R6*R7*R8*rds4 + C2*C3*R1*R5*R6*R7*R8*rds2 + C2*C3*R1*R5*R6*R7*R8*rds4 + C2*C3*R1*R2*R5*R8*R18*rds2 + C2*C3*R1*R2*R6*R8*R18*rds2 + C2*C3*R1*R2*R7*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds2 + C2*C6*R1*R2*R6*R7*R18*rds2 + C2*C3*R1*R5*R6*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds4 + C2*C6*R1*R2*R5*R8*R18*rds4 + C2*C6*R1*R2*R6*R7*R18*rds4 + C2*C3*R1*R6*R7*R8*R18*rds2 + C2*C6*R1*R2*R6*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds2 + C2*C6*R1*R2*R7*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds4 + C2*C6*R1*R5*R6*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds2 + C3*C6*R2*R5*R7*R8*R18*rds2 + C2*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R2*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds4 + C2*C6*R5*R6**2*R7*R8*R18*gm2*rds2 + C2*C3*R5*R6**2*R7*R8*R18*gm4*rds4 + C2*C3*R5*R6**2*R8*R18*gm4*rds2*rds4 + C2*C6*R5*R6**2*R8*R18*gm2*rds2*rds4 + C2*C3*R6**2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R6**2*R7*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R5*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R5*R7*R8*R18*gm4*rds4 + C2*C6*R1*R2*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R6*R7*R8*R18*gm4*rds4 + C2*C6*R1*R5*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R5*R6*R7*R8*R18*gm4*rds4 + C2*C3*R1*R2*R5*R8*R18*gm4*rds2*rds4 + C2*C3*R1*R2*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R5*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R6*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R7*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R5*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R5*R6*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R6*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R6*R7*R8*R18*gm2*rds2*rds4), (R9*R11 + R9*R12 + R10*R11 + R10*R12)*(C2*R5*R6**2*R7*R8 + C2*R5*R6**2*R7*R18 + C2*R5*R6**2*R8*R18 + C2*R6**2*R7*R8*R18 + C2*R5*R6**2*R7*rds2 + C2*R5*R6**2*R7*rds4 + C2*R5*R6**2*R8*rds4 + C2*R6**2*R7*R8*rds4 + C2*R5*R6**2*R18*rds2 + C2*R6**2*R7*R18*rds2 + C2*R5*R6**2*rds2*rds4 + C2*R6**2*R7*rds2*rds4 + C2*R1*R2*R5*R7*R8 + C2*R1*R2*R6*R7*R8 + C2*R1*R5*R6*R7*R8 + C2*R1*R2*R5*R7*R18 + C2*R1*R2*R5*R8*R18 + C2*R1*R2*R6*R7*R18 + C2*R1*R2*R6*R8*R18 + C2*R1*R2*R7*R8*R18 + C2*R1*R5*R6*R7*R18 + C2*R1*R5*R6*R8*R18 + C2*R1*R6*R7*R8*R18 + C3*R1*R5*R7*R8*R18 + C3*R1*R6*R7*R8*R18 + C3*R2*R5*R7*R8*R18 + C3*R2*R6*R7*R8*R18 + C6*R1*R5*R7*R8*R18 + C6*R1*R6*R7*R8*R18 + C6*R2*R5*R7*R8*R18 + C6*R2*R6*R7*R8*R18 + C2*R1*R2*R5*R7*rds2 + C2*R1*R2*R6*R7*rds2 + C2*R1*R2*R5*R7*rds4 + C2*R1*R2*R5*R8*rds4 + C2*R1*R2*R6*R7*rds4 + C2*R1*R2*R6*R8*rds4 + C2*R1*R5*R6*R7*rds2 + C2*R1*R2*R7*R8*rds4 + C2*R1*R5*R6*R7*rds4 + C2*R1*R5*R6*R8*rds4 + C3*R1*R5*R7*R8*rds2 + C3*R1*R6*R7*R8*rds2 + C3*R2*R5*R7*R8*rds2 + C2*R1*R6*R7*R8*rds4 + C3*R1*R5*R7*R8*rds4 + C3*R2*R6*R7*R8*rds2 + C3*R1*R6*R7*R8*rds4 + C3*R2*R5*R7*R8*rds4 + C2*R1*R2*R5*R18*rds2 + C3*R2*R6*R7*R8*rds4 + C2*R1*R2*R6*R18*rds2 + C2*R1*R2*R7*R18*rds2 + C2*R1*R5*R6*R18*rds2 + C2*R1*R6*R7*R18*rds2 + C3*R1*R5*R8*R18*rds2 + C3*R1*R6*R8*R18*rds2 + C3*R2*R5*R8*R18*rds2 + C3*R1*R7*R8*R18*rds2 + C3*R2*R6*R8*R18*rds2 + C6*R1*R5*R7*R18*rds2 + C3*R2*R7*R8*R18*rds2 + C6*R1*R6*R7*R18*rds2 + C6*R2*R5*R7*R18*rds2 + C6*R1*R5*R7*R18*rds4 + C6*R2*R6*R7*R18*rds2 + C6*R1*R5*R8*R18*rds4 + C6*R1*R6*R7*R18*rds4 + C6*R2*R5*R7*R18*rds4 + C6*R1*R6*R8*R18*rds4 + C6*R2*R5*R8*R18*rds4 + C6*R2*R6*R7*R18*rds4 + C6*R1*R7*R8*R18*rds4 + C6*R2*R6*R8*R18*rds4 + C6*R2*R7*R8*R18*rds4 + C2*R1*R2*R5*rds2*rds4 + C2*R1*R2*R6*rds2*rds4 + C2*R1*R2*R7*rds2*rds4 + C2*R1*R5*R6*rds2*rds4 + C2*R1*R6*R7*rds2*rds4 + C3*R1*R5*R8*rds2*rds4 + C3*R1*R6*R8*rds2*rds4 + C3*R2*R5*R8*rds2*rds4 + C3*R1*R7*R8*rds2*rds4 + C3*R2*R6*R8*rds2*rds4 + C3*R2*R7*R8*rds2*rds4 + C6*R1*R5*R18*rds2*rds4 + C6*R1*R6*R18*rds2*rds4 + C6*R2*R5*R18*rds2*rds4 + C6*R1*R7*R18*rds2*rds4 + C6*R2*R6*R18*rds2*rds4 + C6*R2*R7*R18*rds2*rds4 + C2*R5*R6**2*R7*R8*gm2*rds2 + C2*R5*R6**2*R8*R18*gm2*rds2 + C2*R6**2*R7*R8*R18*gm2*rds2 + C2*R5*R6**2*R7*R18*gm4*rds4 + C2*R5*R6**2*R8*R18*gm4*rds4 + C2*R6**2*R7*R8*R18*gm4*rds4 + C2*R5*R6**2*R8*gm2*rds2*rds4 + C2*R6**2*R7*R8*gm2*rds2*rds4 + C2*R5*R6**2*R18*gm4*rds2*rds4 + C2*R6**2*R7*R18*gm4*rds2*rds4 + C2*R1*R2*R5*R7*R8*gm2*rds2 + C2*R1*R2*R6*R7*R8*gm2*rds2 + C2*R1*R5*R6*R7*R8*gm2*rds2 + C2*R1*R2*R5*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm2*rds2 + C2*R1*R2*R7*R8*R18*gm2*rds2 + C2*R1*R2*R5*R7*R18*gm4*rds4 + C2*R1*R2*R5*R8*R18*gm4*rds4 + C2*R1*R2*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm4*rds4 + C2*R1*R2*R7*R8*R18*gm4*rds4 + C2*R1*R6*R7*R8*R18*gm2*rds2 + C2*R1*R5*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm4*rds4 + C6*R1*R5*R7*R8*R18*gm2*rds2 + C2*R1*R6*R7*R8*R18*gm4*rds4 + C3*R1*R5*R7*R8*R18*gm4*rds4 + C6*R1*R6*R7*R8*R18*gm2*rds2 + C6*R2*R5*R7*R8*R18*gm2*rds2 + C3*R1*R6*R7*R8*R18*gm4*rds4 + C3*R2*R5*R7*R8*R18*gm4*rds4 + C6*R2*R6*R7*R8*R18*gm2*rds2 + C3*R2*R6*R7*R8*R18*gm4*rds4 + C2*R1*R2*R5*R8*gm2*rds2*rds4 + C2*R1*R2*R6*R8*gm2*rds2*rds4 + C2*R1*R2*R7*R8*gm2*rds2*rds4 + C2*R1*R5*R6*R8*gm2*rds2*rds4 + C2*R1*R6*R7*R8*gm2*rds2*rds4 + C2*R1*R2*R5*R18*gm4*rds2*rds4 + C2*R1*R2*R6*R18*gm4*rds2*rds4 + C2*R1*R2*R7*R18*gm4*rds2*rds4 + C2*R1*R5*R6*R18*gm4*rds2*rds4 + C2*R1*R6*R7*R18*gm4*rds2*rds4 + C3*R1*R5*R8*R18*gm4*rds2*rds4 + C3*R1*R6*R8*R18*gm4*rds2*rds4 + C3*R2*R5*R8*R18*gm4*rds2*rds4 + C6*R1*R5*R8*R18*gm2*rds2*rds4 + C3*R1*R7*R8*R18*gm4*rds2*rds4 + C3*R2*R6*R8*R18*gm4*rds2*rds4 + C6*R1*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R5*R8*R18*gm2*rds2*rds4 + C3*R2*R7*R8*R18*gm4*rds2*rds4 + C6*R1*R7*R8*R18*gm2*rds2*rds4 + C6*R2*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R7*R8*R18*gm2*rds2*rds4 + C2*R5*R6**2*R8*R18*gm2*gm4*rds2*rds4 + C2*R6**2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R5*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R5*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R6*R7*R8*R18*gm2*gm4*rds2*rds4) + (C4*R9*R10*R11 + C4*R9*R10*R12 + C4*R9*R11*R12 + C4*R10*R11*R12)*(R1*R5*R7*R8 + R1*R6*R7*R8 + R2*R5*R7*R8 + R2*R6*R7*R8 + R1*R5*R7*R18 + R1*R5*R8*R18 + R1*R6*R7*R18 + R2*R5*R7*R18 + R1*R6*R8*R18 + R2*R5*R8*R18 + R2*R6*R7*R18 + R1*R7*R8*R18 + R2*R6*R8*R18 + R2*R7*R8*R18 + R1*R5*R7*rds2 + R1*R6*R7*rds2 + R2*R5*R7*rds2 + R1*R5*R7*rds4 + R2*R6*R7*rds2 + R1*R5*R8*rds4 + R1*R6*R7*rds4 + R2*R5*R7*rds4 + R1*R6*R8*rds4 + R2*R5*R8*rds4 + R2*R6*R7*rds4 + R1*R7*R8*rds4 + R2*R6*R8*rds4 + R2*R7*R8*rds4 + R1*R5*R18*rds2 + R1*R6*R18*rds2 + R2*R5*R18*rds2 + R1*R7*R18*rds2 + R2*R6*R18*rds2 + R2*R7*R18*rds2 + R1*R5*rds2*rds4 + R1*R6*rds2*rds4 + R2*R5*rds2*rds4 + R1*R7*rds2*rds4 + R2*R6*rds2*rds4 + R2*R7*rds2*rds4 + R1*R5*R7*R8*gm2*rds2 + R1*R6*R7*R8*gm2*rds2 + R2*R5*R7*R8*gm2*rds2 + R2*R6*R7*R8*gm2*rds2 + R1*R5*R8*R18*gm2*rds2 + R1*R6*R8*R18*gm2*rds2 + R2*R5*R8*R18*gm2*rds2 + R1*R7*R8*R18*gm2*rds2 + R2*R6*R8*R18*gm2*rds2 + R1*R5*R7*R18*gm4*rds4 + R2*R7*R8*R18*gm2*rds2 + R1*R5*R8*R18*gm4*rds4 + R1*R6*R7*R18*gm4*rds4 + R2*R5*R7*R18*gm4*rds4 + R1*R6*R8*R18*gm4*rds4 + R2*R5*R8*R18*gm4*rds4 + R2*R6*R7*R18*gm4*rds4 + R1*R7*R8*R18*gm4*rds4 + R2*R6*R8*R18*gm4*rds4 + R2*R7*R8*R18*gm4*rds4 + R1*R5*R8*gm2*rds2*rds4 + R1*R6*R8*gm2*rds2*rds4 + R2*R5*R8*gm2*rds2*rds4 + R1*R7*R8*gm2*rds2*rds4 + R2*R6*R8*gm2*rds2*rds4 + R2*R7*R8*gm2*rds2*rds4 + R1*R5*R18*gm4*rds2*rds4 + R1*R6*R18*gm4*rds2*rds4 + R2*R5*R18*gm4*rds2*rds4 + R1*R7*R18*gm4*rds2*rds4 + R2*R6*R18*gm4*rds2*rds4 + R2*R7*R18*gm4*rds2*rds4 + R1*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R7*R8*R18*gm2*gm4*rds2*rds4 + R2*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R7*R8*R18*gm2*gm4*rds2*rds4), (R9*R11 + R9*R12 + R10*R11 + R10*R12)*(R1*R5*R7*R8 + R1*R6*R7*R8 + R2*R5*R7*R8 + R2*R6*R7*R8 + R1*R5*R7*R18 + R1*R5*R8*R18 + R1*R6*R7*R18 + R2*R5*R7*R18 + R1*R6*R8*R18 + R2*R5*R8*R18 + R2*R6*R7*R18 + R1*R7*R8*R18 + R2*R6*R8*R18 + R2*R7*R8*R18 + R1*R5*R7*rds2 + R1*R6*R7*rds2 + R2*R5*R7*rds2 + R1*R5*R7*rds4 + R2*R6*R7*rds2 + R1*R5*R8*rds4 + R1*R6*R7*rds4 + R2*R5*R7*rds4 + R1*R6*R8*rds4 + R2*R5*R8*rds4 + R2*R6*R7*rds4 + R1*R7*R8*rds4 + R2*R6*R8*rds4 + R2*R7*R8*rds4 + R1*R5*R18*rds2 + R1*R6*R18*rds2 + R2*R5*R18*rds2 + R1*R7*R18*rds2 + R2*R6*R18*rds2 + R2*R7*R18*rds2 + R1*R5*rds2*rds4 + R1*R6*rds2*rds4 + R2*R5*rds2*rds4 + R1*R7*rds2*rds4 + R2*R6*rds2*rds4 + R2*R7*rds2*rds4 + R1*R5*R7*R8*gm2*rds2 + R1*R6*R7*R8*gm2*rds2 + R2*R5*R7*R8*gm2*rds2 + R2*R6*R7*R8*gm2*rds2 + R1*R5*R8*R18*gm2*rds2 + R1*R6*R8*R18*gm2*rds2 + R2*R5*R8*R18*gm2*rds2 + R1*R7*R8*R18*gm2*rds2 + R2*R6*R8*R18*gm2*rds2 + R1*R5*R7*R18*gm4*rds4 + R2*R7*R8*R18*gm2*rds2 + R1*R5*R8*R18*gm4*rds4 + R1*R6*R7*R18*gm4*rds4 + R2*R5*R7*R18*gm4*rds4 + R1*R6*R8*R18*gm4*rds4 + R2*R5*R8*R18*gm4*rds4 + R2*R6*R7*R18*gm4*rds4 + R1*R7*R8*R18*gm4*rds4 + R2*R6*R8*R18*gm4*rds4 + R2*R7*R8*R18*gm4*rds4 + R1*R5*R8*gm2*rds2*rds4 + R1*R6*R8*gm2*rds2*rds4 + R2*R5*R8*gm2*rds2*rds4 + R1*R7*R8*gm2*rds2*rds4 + R2*R6*R8*gm2*rds2*rds4 + R2*R7*R8*gm2*rds2*rds4 + R1*R5*R18*gm4*rds2*rds4 + R1*R6*R18*gm4*rds2*rds4 + R2*R5*R18*gm4*rds2*rds4 + R1*R7*R18*gm4*rds2*rds4 + R2*R6*R18*gm4*rds2*rds4 + R2*R7*R18*gm4*rds2*rds4 + R1*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R7*R8*R18*gm2*gm4*rds2*rds4 + R2*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R7*R8*R18*gm2*gm4*rds2*rds4)]
 

    H_31_s = sig.TransferFunction(num_31_coeffs, den_31_coeffs)
    dt = 1/f_samp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.linalg.LinAlgWarning)
        H_31_z = H_31_s.to_discrete(dt=dt, method='bilinear')

    num = H_31_z.num
    den = H_31_z.den

    if len(num) < len(den):
        # Pad with leading zeros
        num = np.pad(num, (len(den) - len(num), 0), 'constant')

    _, H_resp = sig.freqz(num, den, worN=frequencies_hz, fs=f_samp)
    
    return H_resp



def H32_TF_frequency_response_from_circuit_params_hardware(circuit_params, frequencies_hz, f_samp):
    '''
    Compute the frequency response of H32 from the circuit parameters for the hardware implementation.
    
    Parameters:
    -----------
    circuit_params: dict containing the circuit parameters
    frequencies_hz: numpy array containing the frequencies at which to compute the frequency response
    f_samp: float containing the sampling frequency

    Returns:
    --------
    H_32: numpy array containing the frequency response of H32 at the specified frequencies
    '''
    list_of_params_to_check = ['R1', 'R2', 'R5', 'R6', 'R7', 'R17', 'R18', 'R19', 'R20', 'R21', 'C2', 'C6', 'C7']

    R1 = circuit_params['R1']
    R2 = circuit_params['R2']
    R5 = circuit_params['R5']
    R6 = circuit_params['R6']
    R7 = circuit_params['R7']
    R8 = circuit_params['R8']
    R17 = circuit_params['R17']
    R18 = circuit_params['R18']
    R19 = circuit_params['R19']
    R20 = circuit_params['R20']
    R21 = circuit_params['R21']
    C2 = circuit_params['C2']
    C6 = circuit_params['C6']
    C7 = circuit_params['C7']
    gm2 = circuit_params['gm2']
    rds2 = circuit_params['rds2']
    gm4 = circuit_params['gm4']
    rds4 = circuit_params['rds4']
    C3 = circuit_params['C3']
    R18 = circuit_params['R18']

    num_32_coeffs = [-C3*C6*C7*R8*R17*R18*R19*R21*gm4*rds2*rds4*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7), - (C3*C7*R8*R17*R19*R21*gm4*rds2*rds4 + C6*C7*R17*R18*R19*R21*gm4*rds4*(R8 + rds2 + R8*gm2*rds2))*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7) - C3*C6*C7*R8*R17*R18*R19*R21*gm4*rds2*rds4*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7), - (C3*C7*R8*R17*R19*R21*gm4*rds2*rds4 + C6*C7*R17*R18*R19*R21*gm4*rds4*(R8 + rds2 + R8*gm2*rds2))*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7) - C7*R17*R19*R21*gm4*rds4*(R8 + rds2 + R8*gm2*rds2)*(C2*R5*R6**2*R7 + C2*R1*R2*R5*R7 + C2*R1*R2*R6*R7 + C2*R1*R5*R6*R7), -C7*R17*R19*R21*gm4*rds4*(R8 + rds2 + R8*gm2*rds2)*(R1*R5*R7 + R1*R6*R7 + R2*R5*R7 + R2*R6*R7), 0]
    den_32_coeffs = [(C7*R17*R19*R20 + C7*R17*R19*R21 + C7*R17*R20*R21 + C7*R19*R20*R21)*(C2*C3*C6*R5*R6**2*R7*R8*R18*rds2 + C2*C3*C6*R5*R6**2*R7*R8*R18*rds4 + C2*C3*C6*R5*R6**2*R8*R18*rds2*rds4 + C2*C3*C6*R6**2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R5*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R5*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R6*R7*R8*R18*rds2*rds4), (R17*R20 + R17*R21 + R19*R20 + R19*R21)*(C2*C3*C6*R5*R6**2*R7*R8*R18*rds2 + C2*C3*C6*R5*R6**2*R7*R8*R18*rds4 + C2*C3*C6*R5*R6**2*R8*R18*rds2*rds4 + C2*C3*C6*R6**2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R2*R5*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds2 + C2*C3*C6*R1*R5*R6*R7*R8*R18*rds4 + C2*C3*C6*R1*R2*R5*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R2*R7*R8*R18*rds2*rds4 + C2*C3*C6*R1*R5*R6*R8*R18*rds2*rds4 + C2*C3*C6*R1*R6*R7*R8*R18*rds2*rds4) + (C7*R17*R19*R20 + C7*R17*R19*R21 + C7*R17*R20*R21 + C7*R19*R20*R21)*(C2*C3*R5*R6**2*R8*rds2*rds4 + C2*C3*R6**2*R7*R8*rds2*rds4 + C2*C6*R5*R6**2*R18*rds2*rds4 + C2*C6*R6**2*R7*R18*rds2*rds4 + C2*C3*R5*R6**2*R7*R8*R18 + C2*C6*R5*R6**2*R7*R8*R18 + C2*C3*R5*R6**2*R7*R8*rds2 + C2*C3*R5*R6**2*R7*R8*rds4 + C2*C3*R5*R6**2*R8*R18*rds2 + C2*C3*R6**2*R7*R8*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds4 + C2*C6*R5*R6**2*R8*R18*rds4 + C2*C6*R6**2*R7*R8*R18*rds4 + C2*C3*R1*R2*R5*R8*rds2*rds4 + C2*C3*R1*R2*R6*R8*rds2*rds4 + C2*C3*R1*R2*R7*R8*rds2*rds4 + C2*C3*R1*R5*R6*R8*rds2*rds4 + C2*C3*R1*R6*R7*R8*rds2*rds4 + C2*C6*R1*R2*R5*R18*rds2*rds4 + C2*C6*R1*R2*R6*R18*rds2*rds4 + C2*C6*R1*R2*R7*R18*rds2*rds4 + C2*C6*R1*R5*R6*R18*rds2*rds4 + C2*C6*R1*R6*R7*R18*rds2*rds4 + C3*C6*R1*R5*R8*R18*rds2*rds4 + C3*C6*R1*R6*R8*R18*rds2*rds4 + C3*C6*R2*R5*R8*R18*rds2*rds4 + C3*C6*R1*R7*R8*R18*rds2*rds4 + C3*C6*R2*R6*R8*R18*rds2*rds4 + C3*C6*R2*R7*R8*R18*rds2*rds4 + C2*C3*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R5*R6*R7*R8*R18 + C2*C6*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R5*R6*R7*R8*R18 + C2*C3*R1*R2*R5*R7*R8*rds2 + C2*C3*R1*R2*R6*R7*R8*rds2 + C2*C3*R1*R2*R5*R7*R8*rds4 + C2*C3*R1*R2*R6*R7*R8*rds4 + C2*C3*R1*R5*R6*R7*R8*rds2 + C2*C3*R1*R5*R6*R7*R8*rds4 + C2*C3*R1*R2*R5*R8*R18*rds2 + C2*C3*R1*R2*R6*R8*R18*rds2 + C2*C3*R1*R2*R7*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds2 + C2*C6*R1*R2*R6*R7*R18*rds2 + C2*C3*R1*R5*R6*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds4 + C2*C6*R1*R2*R5*R8*R18*rds4 + C2*C6*R1*R2*R6*R7*R18*rds4 + C2*C3*R1*R6*R7*R8*R18*rds2 + C2*C6*R1*R2*R6*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds2 + C2*C6*R1*R2*R7*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds4 + C2*C6*R1*R5*R6*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds2 + C3*C6*R2*R5*R7*R8*R18*rds2 + C2*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R2*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds4 + C2*C6*R5*R6**2*R7*R8*R18*gm2*rds2 + C2*C3*R5*R6**2*R7*R8*R18*gm4*rds4 + C2*C3*R5*R6**2*R8*R18*gm4*rds2*rds4 + C2*C6*R5*R6**2*R8*R18*gm2*rds2*rds4 + C2*C3*R6**2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R6**2*R7*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R5*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R5*R7*R8*R18*gm4*rds4 + C2*C6*R1*R2*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R6*R7*R8*R18*gm4*rds4 + C2*C6*R1*R5*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R5*R6*R7*R8*R18*gm4*rds4 + C2*C3*R1*R2*R5*R8*R18*gm4*rds2*rds4 + C2*C3*R1*R2*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R5*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R6*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R7*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R5*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R5*R6*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R6*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R6*R7*R8*R18*gm2*rds2*rds4), (C7*R17*R19*R20 + C7*R17*R19*R21 + C7*R17*R20*R21 + C7*R19*R20*R21)*(C2*R5*R6**2*R7*R8 + C2*R5*R6**2*R7*R18 + C2*R5*R6**2*R8*R18 + C2*R6**2*R7*R8*R18 + C2*R5*R6**2*R7*rds2 + C2*R5*R6**2*R7*rds4 + C2*R5*R6**2*R8*rds4 + C2*R6**2*R7*R8*rds4 + C2*R5*R6**2*R18*rds2 + C2*R6**2*R7*R18*rds2 + C2*R5*R6**2*rds2*rds4 + C2*R6**2*R7*rds2*rds4 + C2*R1*R2*R5*R7*R8 + C2*R1*R2*R6*R7*R8 + C2*R1*R5*R6*R7*R8 + C2*R1*R2*R5*R7*R18 + C2*R1*R2*R5*R8*R18 + C2*R1*R2*R6*R7*R18 + C2*R1*R2*R6*R8*R18 + C2*R1*R2*R7*R8*R18 + C2*R1*R5*R6*R7*R18 + C2*R1*R5*R6*R8*R18 + C2*R1*R6*R7*R8*R18 + C3*R1*R5*R7*R8*R18 + C3*R1*R6*R7*R8*R18 + C3*R2*R5*R7*R8*R18 + C3*R2*R6*R7*R8*R18 + C6*R1*R5*R7*R8*R18 + C6*R1*R6*R7*R8*R18 + C6*R2*R5*R7*R8*R18 + C6*R2*R6*R7*R8*R18 + C2*R1*R2*R5*R7*rds2 + C2*R1*R2*R6*R7*rds2 + C2*R1*R2*R5*R7*rds4 + C2*R1*R2*R5*R8*rds4 + C2*R1*R2*R6*R7*rds4 + C2*R1*R2*R6*R8*rds4 + C2*R1*R5*R6*R7*rds2 + C2*R1*R2*R7*R8*rds4 + C2*R1*R5*R6*R7*rds4 + C2*R1*R5*R6*R8*rds4 + C3*R1*R5*R7*R8*rds2 + C3*R1*R6*R7*R8*rds2 + C3*R2*R5*R7*R8*rds2 + C2*R1*R6*R7*R8*rds4 + C3*R1*R5*R7*R8*rds4 + C3*R2*R6*R7*R8*rds2 + C3*R1*R6*R7*R8*rds4 + C3*R2*R5*R7*R8*rds4 + C2*R1*R2*R5*R18*rds2 + C3*R2*R6*R7*R8*rds4 + C2*R1*R2*R6*R18*rds2 + C2*R1*R2*R7*R18*rds2 + C2*R1*R5*R6*R18*rds2 + C2*R1*R6*R7*R18*rds2 + C3*R1*R5*R8*R18*rds2 + C3*R1*R6*R8*R18*rds2 + C3*R2*R5*R8*R18*rds2 + C3*R1*R7*R8*R18*rds2 + C3*R2*R6*R8*R18*rds2 + C6*R1*R5*R7*R18*rds2 + C3*R2*R7*R8*R18*rds2 + C6*R1*R6*R7*R18*rds2 + C6*R2*R5*R7*R18*rds2 + C6*R1*R5*R7*R18*rds4 + C6*R2*R6*R7*R18*rds2 + C6*R1*R5*R8*R18*rds4 + C6*R1*R6*R7*R18*rds4 + C6*R2*R5*R7*R18*rds4 + C6*R1*R6*R8*R18*rds4 + C6*R2*R5*R8*R18*rds4 + C6*R2*R6*R7*R18*rds4 + C6*R1*R7*R8*R18*rds4 + C6*R2*R6*R8*R18*rds4 + C6*R2*R7*R8*R18*rds4 + C2*R1*R2*R5*rds2*rds4 + C2*R1*R2*R6*rds2*rds4 + C2*R1*R2*R7*rds2*rds4 + C2*R1*R5*R6*rds2*rds4 + C2*R1*R6*R7*rds2*rds4 + C3*R1*R5*R8*rds2*rds4 + C3*R1*R6*R8*rds2*rds4 + C3*R2*R5*R8*rds2*rds4 + C3*R1*R7*R8*rds2*rds4 + C3*R2*R6*R8*rds2*rds4 + C3*R2*R7*R8*rds2*rds4 + C6*R1*R5*R18*rds2*rds4 + C6*R1*R6*R18*rds2*rds4 + C6*R2*R5*R18*rds2*rds4 + C6*R1*R7*R18*rds2*rds4 + C6*R2*R6*R18*rds2*rds4 + C6*R2*R7*R18*rds2*rds4 + C2*R5*R6**2*R7*R8*gm2*rds2 + C2*R5*R6**2*R8*R18*gm2*rds2 + C2*R6**2*R7*R8*R18*gm2*rds2 + C2*R5*R6**2*R7*R18*gm4*rds4 + C2*R5*R6**2*R8*R18*gm4*rds4 + C2*R6**2*R7*R8*R18*gm4*rds4 + C2*R5*R6**2*R8*gm2*rds2*rds4 + C2*R6**2*R7*R8*gm2*rds2*rds4 + C2*R5*R6**2*R18*gm4*rds2*rds4 + C2*R6**2*R7*R18*gm4*rds2*rds4 + C2*R1*R2*R5*R7*R8*gm2*rds2 + C2*R1*R2*R6*R7*R8*gm2*rds2 + C2*R1*R5*R6*R7*R8*gm2*rds2 + C2*R1*R2*R5*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm2*rds2 + C2*R1*R2*R7*R8*R18*gm2*rds2 + C2*R1*R2*R5*R7*R18*gm4*rds4 + C2*R1*R2*R5*R8*R18*gm4*rds4 + C2*R1*R2*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm4*rds4 + C2*R1*R2*R7*R8*R18*gm4*rds4 + C2*R1*R6*R7*R8*R18*gm2*rds2 + C2*R1*R5*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm4*rds4 + C6*R1*R5*R7*R8*R18*gm2*rds2 + C2*R1*R6*R7*R8*R18*gm4*rds4 + C3*R1*R5*R7*R8*R18*gm4*rds4 + C6*R1*R6*R7*R8*R18*gm2*rds2 + C6*R2*R5*R7*R8*R18*gm2*rds2 + C3*R1*R6*R7*R8*R18*gm4*rds4 + C3*R2*R5*R7*R8*R18*gm4*rds4 + C6*R2*R6*R7*R8*R18*gm2*rds2 + C3*R2*R6*R7*R8*R18*gm4*rds4 + C2*R1*R2*R5*R8*gm2*rds2*rds4 + C2*R1*R2*R6*R8*gm2*rds2*rds4 + C2*R1*R2*R7*R8*gm2*rds2*rds4 + C2*R1*R5*R6*R8*gm2*rds2*rds4 + C2*R1*R6*R7*R8*gm2*rds2*rds4 + C2*R1*R2*R5*R18*gm4*rds2*rds4 + C2*R1*R2*R6*R18*gm4*rds2*rds4 + C2*R1*R2*R7*R18*gm4*rds2*rds4 + C2*R1*R5*R6*R18*gm4*rds2*rds4 + C2*R1*R6*R7*R18*gm4*rds2*rds4 + C3*R1*R5*R8*R18*gm4*rds2*rds4 + C3*R1*R6*R8*R18*gm4*rds2*rds4 + C3*R2*R5*R8*R18*gm4*rds2*rds4 + C6*R1*R5*R8*R18*gm2*rds2*rds4 + C3*R1*R7*R8*R18*gm4*rds2*rds4 + C3*R2*R6*R8*R18*gm4*rds2*rds4 + C6*R1*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R5*R8*R18*gm2*rds2*rds4 + C3*R2*R7*R8*R18*gm4*rds2*rds4 + C6*R1*R7*R8*R18*gm2*rds2*rds4 + C6*R2*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R7*R8*R18*gm2*rds2*rds4 + C2*R5*R6**2*R8*R18*gm2*gm4*rds2*rds4 + C2*R6**2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R5*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R5*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R6*R7*R8*R18*gm2*gm4*rds2*rds4) + (R17*R20 + R17*R21 + R19*R20 + R19*R21)*(C2*C3*R5*R6**2*R8*rds2*rds4 + C2*C3*R6**2*R7*R8*rds2*rds4 + C2*C6*R5*R6**2*R18*rds2*rds4 + C2*C6*R6**2*R7*R18*rds2*rds4 + C2*C3*R5*R6**2*R7*R8*R18 + C2*C6*R5*R6**2*R7*R8*R18 + C2*C3*R5*R6**2*R7*R8*rds2 + C2*C3*R5*R6**2*R7*R8*rds4 + C2*C3*R5*R6**2*R8*R18*rds2 + C2*C3*R6**2*R7*R8*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds2 + C2*C6*R5*R6**2*R7*R18*rds4 + C2*C6*R5*R6**2*R8*R18*rds4 + C2*C6*R6**2*R7*R8*R18*rds4 + C2*C3*R1*R2*R5*R8*rds2*rds4 + C2*C3*R1*R2*R6*R8*rds2*rds4 + C2*C3*R1*R2*R7*R8*rds2*rds4 + C2*C3*R1*R5*R6*R8*rds2*rds4 + C2*C3*R1*R6*R7*R8*rds2*rds4 + C2*C6*R1*R2*R5*R18*rds2*rds4 + C2*C6*R1*R2*R6*R18*rds2*rds4 + C2*C6*R1*R2*R7*R18*rds2*rds4 + C2*C6*R1*R5*R6*R18*rds2*rds4 + C2*C6*R1*R6*R7*R18*rds2*rds4 + C3*C6*R1*R5*R8*R18*rds2*rds4 + C3*C6*R1*R6*R8*R18*rds2*rds4 + C3*C6*R2*R5*R8*R18*rds2*rds4 + C3*C6*R1*R7*R8*R18*rds2*rds4 + C3*C6*R2*R6*R8*R18*rds2*rds4 + C3*C6*R2*R7*R8*R18*rds2*rds4 + C2*C3*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R2*R5*R7*R8*R18 + C2*C3*R1*R5*R6*R7*R8*R18 + C2*C6*R1*R2*R6*R7*R8*R18 + C2*C6*R1*R5*R6*R7*R8*R18 + C2*C3*R1*R2*R5*R7*R8*rds2 + C2*C3*R1*R2*R6*R7*R8*rds2 + C2*C3*R1*R2*R5*R7*R8*rds4 + C2*C3*R1*R2*R6*R7*R8*rds4 + C2*C3*R1*R5*R6*R7*R8*rds2 + C2*C3*R1*R5*R6*R7*R8*rds4 + C2*C3*R1*R2*R5*R8*R18*rds2 + C2*C3*R1*R2*R6*R8*R18*rds2 + C2*C3*R1*R2*R7*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds2 + C2*C6*R1*R2*R6*R7*R18*rds2 + C2*C3*R1*R5*R6*R8*R18*rds2 + C2*C6*R1*R2*R5*R7*R18*rds4 + C2*C6*R1*R2*R5*R8*R18*rds4 + C2*C6*R1*R2*R6*R7*R18*rds4 + C2*C3*R1*R6*R7*R8*R18*rds2 + C2*C6*R1*R2*R6*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds2 + C2*C6*R1*R2*R7*R8*R18*rds4 + C2*C6*R1*R5*R6*R7*R18*rds4 + C2*C6*R1*R5*R6*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds2 + C3*C6*R2*R5*R7*R8*R18*rds2 + C2*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R1*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds2 + C3*C6*R1*R6*R7*R8*R18*rds4 + C3*C6*R2*R5*R7*R8*R18*rds4 + C3*C6*R2*R6*R7*R8*R18*rds4 + C2*C6*R5*R6**2*R7*R8*R18*gm2*rds2 + C2*C3*R5*R6**2*R7*R8*R18*gm4*rds4 + C2*C3*R5*R6**2*R8*R18*gm4*rds2*rds4 + C2*C6*R5*R6**2*R8*R18*gm2*rds2*rds4 + C2*C3*R6**2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R6**2*R7*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R5*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R5*R7*R8*R18*gm4*rds4 + C2*C6*R1*R2*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R2*R6*R7*R8*R18*gm4*rds4 + C2*C6*R1*R5*R6*R7*R8*R18*gm2*rds2 + C2*C3*R1*R5*R6*R7*R8*R18*gm4*rds4 + C2*C3*R1*R2*R5*R8*R18*gm4*rds2*rds4 + C2*C3*R1*R2*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R5*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R2*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R2*R6*R8*R18*gm2*rds2*rds4 + C2*C6*R1*R2*R7*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R5*R6*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R5*R6*R8*R18*gm2*rds2*rds4 + C2*C3*R1*R6*R7*R8*R18*gm4*rds2*rds4 + C2*C6*R1*R6*R7*R8*R18*gm2*rds2*rds4), (R17*R20 + R17*R21 + R19*R20 + R19*R21)*(C2*R5*R6**2*R7*R8 + C2*R5*R6**2*R7*R18 + C2*R5*R6**2*R8*R18 + C2*R6**2*R7*R8*R18 + C2*R5*R6**2*R7*rds2 + C2*R5*R6**2*R7*rds4 + C2*R5*R6**2*R8*rds4 + C2*R6**2*R7*R8*rds4 + C2*R5*R6**2*R18*rds2 + C2*R6**2*R7*R18*rds2 + C2*R5*R6**2*rds2*rds4 + C2*R6**2*R7*rds2*rds4 + C2*R1*R2*R5*R7*R8 + C2*R1*R2*R6*R7*R8 + C2*R1*R5*R6*R7*R8 + C2*R1*R2*R5*R7*R18 + C2*R1*R2*R5*R8*R18 + C2*R1*R2*R6*R7*R18 + C2*R1*R2*R6*R8*R18 + C2*R1*R2*R7*R8*R18 + C2*R1*R5*R6*R7*R18 + C2*R1*R5*R6*R8*R18 + C2*R1*R6*R7*R8*R18 + C3*R1*R5*R7*R8*R18 + C3*R1*R6*R7*R8*R18 + C3*R2*R5*R7*R8*R18 + C3*R2*R6*R7*R8*R18 + C6*R1*R5*R7*R8*R18 + C6*R1*R6*R7*R8*R18 + C6*R2*R5*R7*R8*R18 + C6*R2*R6*R7*R8*R18 + C2*R1*R2*R5*R7*rds2 + C2*R1*R2*R6*R7*rds2 + C2*R1*R2*R5*R7*rds4 + C2*R1*R2*R5*R8*rds4 + C2*R1*R2*R6*R7*rds4 + C2*R1*R2*R6*R8*rds4 + C2*R1*R5*R6*R7*rds2 + C2*R1*R2*R7*R8*rds4 + C2*R1*R5*R6*R7*rds4 + C2*R1*R5*R6*R8*rds4 + C3*R1*R5*R7*R8*rds2 + C3*R1*R6*R7*R8*rds2 + C3*R2*R5*R7*R8*rds2 + C2*R1*R6*R7*R8*rds4 + C3*R1*R5*R7*R8*rds4 + C3*R2*R6*R7*R8*rds2 + C3*R1*R6*R7*R8*rds4 + C3*R2*R5*R7*R8*rds4 + C2*R1*R2*R5*R18*rds2 + C3*R2*R6*R7*R8*rds4 + C2*R1*R2*R6*R18*rds2 + C2*R1*R2*R7*R18*rds2 + C2*R1*R5*R6*R18*rds2 + C2*R1*R6*R7*R18*rds2 + C3*R1*R5*R8*R18*rds2 + C3*R1*R6*R8*R18*rds2 + C3*R2*R5*R8*R18*rds2 + C3*R1*R7*R8*R18*rds2 + C3*R2*R6*R8*R18*rds2 + C6*R1*R5*R7*R18*rds2 + C3*R2*R7*R8*R18*rds2 + C6*R1*R6*R7*R18*rds2 + C6*R2*R5*R7*R18*rds2 + C6*R1*R5*R7*R18*rds4 + C6*R2*R6*R7*R18*rds2 + C6*R1*R5*R8*R18*rds4 + C6*R1*R6*R7*R18*rds4 + C6*R2*R5*R7*R18*rds4 + C6*R1*R6*R8*R18*rds4 + C6*R2*R5*R8*R18*rds4 + C6*R2*R6*R7*R18*rds4 + C6*R1*R7*R8*R18*rds4 + C6*R2*R6*R8*R18*rds4 + C6*R2*R7*R8*R18*rds4 + C2*R1*R2*R5*rds2*rds4 + C2*R1*R2*R6*rds2*rds4 + C2*R1*R2*R7*rds2*rds4 + C2*R1*R5*R6*rds2*rds4 + C2*R1*R6*R7*rds2*rds4 + C3*R1*R5*R8*rds2*rds4 + C3*R1*R6*R8*rds2*rds4 + C3*R2*R5*R8*rds2*rds4 + C3*R1*R7*R8*rds2*rds4 + C3*R2*R6*R8*rds2*rds4 + C3*R2*R7*R8*rds2*rds4 + C6*R1*R5*R18*rds2*rds4 + C6*R1*R6*R18*rds2*rds4 + C6*R2*R5*R18*rds2*rds4 + C6*R1*R7*R18*rds2*rds4 + C6*R2*R6*R18*rds2*rds4 + C6*R2*R7*R18*rds2*rds4 + C2*R5*R6**2*R7*R8*gm2*rds2 + C2*R5*R6**2*R8*R18*gm2*rds2 + C2*R6**2*R7*R8*R18*gm2*rds2 + C2*R5*R6**2*R7*R18*gm4*rds4 + C2*R5*R6**2*R8*R18*gm4*rds4 + C2*R6**2*R7*R8*R18*gm4*rds4 + C2*R5*R6**2*R8*gm2*rds2*rds4 + C2*R6**2*R7*R8*gm2*rds2*rds4 + C2*R5*R6**2*R18*gm4*rds2*rds4 + C2*R6**2*R7*R18*gm4*rds2*rds4 + C2*R1*R2*R5*R7*R8*gm2*rds2 + C2*R1*R2*R6*R7*R8*gm2*rds2 + C2*R1*R5*R6*R7*R8*gm2*rds2 + C2*R1*R2*R5*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm2*rds2 + C2*R1*R2*R7*R8*R18*gm2*rds2 + C2*R1*R2*R5*R7*R18*gm4*rds4 + C2*R1*R2*R5*R8*R18*gm4*rds4 + C2*R1*R2*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm2*rds2 + C2*R1*R2*R6*R8*R18*gm4*rds4 + C2*R1*R2*R7*R8*R18*gm4*rds4 + C2*R1*R6*R7*R8*R18*gm2*rds2 + C2*R1*R5*R6*R7*R18*gm4*rds4 + C2*R1*R5*R6*R8*R18*gm4*rds4 + C6*R1*R5*R7*R8*R18*gm2*rds2 + C2*R1*R6*R7*R8*R18*gm4*rds4 + C3*R1*R5*R7*R8*R18*gm4*rds4 + C6*R1*R6*R7*R8*R18*gm2*rds2 + C6*R2*R5*R7*R8*R18*gm2*rds2 + C3*R1*R6*R7*R8*R18*gm4*rds4 + C3*R2*R5*R7*R8*R18*gm4*rds4 + C6*R2*R6*R7*R8*R18*gm2*rds2 + C3*R2*R6*R7*R8*R18*gm4*rds4 + C2*R1*R2*R5*R8*gm2*rds2*rds4 + C2*R1*R2*R6*R8*gm2*rds2*rds4 + C2*R1*R2*R7*R8*gm2*rds2*rds4 + C2*R1*R5*R6*R8*gm2*rds2*rds4 + C2*R1*R6*R7*R8*gm2*rds2*rds4 + C2*R1*R2*R5*R18*gm4*rds2*rds4 + C2*R1*R2*R6*R18*gm4*rds2*rds4 + C2*R1*R2*R7*R18*gm4*rds2*rds4 + C2*R1*R5*R6*R18*gm4*rds2*rds4 + C2*R1*R6*R7*R18*gm4*rds2*rds4 + C3*R1*R5*R8*R18*gm4*rds2*rds4 + C3*R1*R6*R8*R18*gm4*rds2*rds4 + C3*R2*R5*R8*R18*gm4*rds2*rds4 + C6*R1*R5*R8*R18*gm2*rds2*rds4 + C3*R1*R7*R8*R18*gm4*rds2*rds4 + C3*R2*R6*R8*R18*gm4*rds2*rds4 + C6*R1*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R5*R8*R18*gm2*rds2*rds4 + C3*R2*R7*R8*R18*gm4*rds2*rds4 + C6*R1*R7*R8*R18*gm2*rds2*rds4 + C6*R2*R6*R8*R18*gm2*rds2*rds4 + C6*R2*R7*R8*R18*gm2*rds2*rds4 + C2*R5*R6**2*R8*R18*gm2*gm4*rds2*rds4 + C2*R6**2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R5*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R2*R7*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R5*R6*R8*R18*gm2*gm4*rds2*rds4 + C2*R1*R6*R7*R8*R18*gm2*gm4*rds2*rds4) + (C7*R17*R19*R20 + C7*R17*R19*R21 + C7*R17*R20*R21 + C7*R19*R20*R21)*(R1*R5*R7*R8 + R1*R6*R7*R8 + R2*R5*R7*R8 + R2*R6*R7*R8 + R1*R5*R7*R18 + R1*R5*R8*R18 + R1*R6*R7*R18 + R2*R5*R7*R18 + R1*R6*R8*R18 + R2*R5*R8*R18 + R2*R6*R7*R18 + R1*R7*R8*R18 + R2*R6*R8*R18 + R2*R7*R8*R18 + R1*R5*R7*rds2 + R1*R6*R7*rds2 + R2*R5*R7*rds2 + R1*R5*R7*rds4 + R2*R6*R7*rds2 + R1*R5*R8*rds4 + R1*R6*R7*rds4 + R2*R5*R7*rds4 + R1*R6*R8*rds4 + R2*R5*R8*rds4 + R2*R6*R7*rds4 + R1*R7*R8*rds4 + R2*R6*R8*rds4 + R2*R7*R8*rds4 + R1*R5*R18*rds2 + R1*R6*R18*rds2 + R2*R5*R18*rds2 + R1*R7*R18*rds2 + R2*R6*R18*rds2 + R2*R7*R18*rds2 + R1*R5*rds2*rds4 + R1*R6*rds2*rds4 + R2*R5*rds2*rds4 + R1*R7*rds2*rds4 + R2*R6*rds2*rds4 + R2*R7*rds2*rds4 + R1*R5*R7*R8*gm2*rds2 + R1*R6*R7*R8*gm2*rds2 + R2*R5*R7*R8*gm2*rds2 + R2*R6*R7*R8*gm2*rds2 + R1*R5*R8*R18*gm2*rds2 + R1*R6*R8*R18*gm2*rds2 + R2*R5*R8*R18*gm2*rds2 + R1*R7*R8*R18*gm2*rds2 + R2*R6*R8*R18*gm2*rds2 + R1*R5*R7*R18*gm4*rds4 + R2*R7*R8*R18*gm2*rds2 + R1*R5*R8*R18*gm4*rds4 + R1*R6*R7*R18*gm4*rds4 + R2*R5*R7*R18*gm4*rds4 + R1*R6*R8*R18*gm4*rds4 + R2*R5*R8*R18*gm4*rds4 + R2*R6*R7*R18*gm4*rds4 + R1*R7*R8*R18*gm4*rds4 + R2*R6*R8*R18*gm4*rds4 + R2*R7*R8*R18*gm4*rds4 + R1*R5*R8*gm2*rds2*rds4 + R1*R6*R8*gm2*rds2*rds4 + R2*R5*R8*gm2*rds2*rds4 + R1*R7*R8*gm2*rds2*rds4 + R2*R6*R8*gm2*rds2*rds4 + R2*R7*R8*gm2*rds2*rds4 + R1*R5*R18*gm4*rds2*rds4 + R1*R6*R18*gm4*rds2*rds4 + R2*R5*R18*gm4*rds2*rds4 + R1*R7*R18*gm4*rds2*rds4 + R2*R6*R18*gm4*rds2*rds4 + R2*R7*R18*gm4*rds2*rds4 + R1*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R7*R8*R18*gm2*gm4*rds2*rds4 + R2*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R7*R8*R18*gm2*gm4*rds2*rds4), (R17*R20 + R17*R21 + R19*R20 + R19*R21)*(R1*R5*R7*R8 + R1*R6*R7*R8 + R2*R5*R7*R8 + R2*R6*R7*R8 + R1*R5*R7*R18 + R1*R5*R8*R18 + R1*R6*R7*R18 + R2*R5*R7*R18 + R1*R6*R8*R18 + R2*R5*R8*R18 + R2*R6*R7*R18 + R1*R7*R8*R18 + R2*R6*R8*R18 + R2*R7*R8*R18 + R1*R5*R7*rds2 + R1*R6*R7*rds2 + R2*R5*R7*rds2 + R1*R5*R7*rds4 + R2*R6*R7*rds2 + R1*R5*R8*rds4 + R1*R6*R7*rds4 + R2*R5*R7*rds4 + R1*R6*R8*rds4 + R2*R5*R8*rds4 + R2*R6*R7*rds4 + R1*R7*R8*rds4 + R2*R6*R8*rds4 + R2*R7*R8*rds4 + R1*R5*R18*rds2 + R1*R6*R18*rds2 + R2*R5*R18*rds2 + R1*R7*R18*rds2 + R2*R6*R18*rds2 + R2*R7*R18*rds2 + R1*R5*rds2*rds4 + R1*R6*rds2*rds4 + R2*R5*rds2*rds4 + R1*R7*rds2*rds4 + R2*R6*rds2*rds4 + R2*R7*rds2*rds4 + R1*R5*R7*R8*gm2*rds2 + R1*R6*R7*R8*gm2*rds2 + R2*R5*R7*R8*gm2*rds2 + R2*R6*R7*R8*gm2*rds2 + R1*R5*R8*R18*gm2*rds2 + R1*R6*R8*R18*gm2*rds2 + R2*R5*R8*R18*gm2*rds2 + R1*R7*R8*R18*gm2*rds2 + R2*R6*R8*R18*gm2*rds2 + R1*R5*R7*R18*gm4*rds4 + R2*R7*R8*R18*gm2*rds2 + R1*R5*R8*R18*gm4*rds4 + R1*R6*R7*R18*gm4*rds4 + R2*R5*R7*R18*gm4*rds4 + R1*R6*R8*R18*gm4*rds4 + R2*R5*R8*R18*gm4*rds4 + R2*R6*R7*R18*gm4*rds4 + R1*R7*R8*R18*gm4*rds4 + R2*R6*R8*R18*gm4*rds4 + R2*R7*R8*R18*gm4*rds4 + R1*R5*R8*gm2*rds2*rds4 + R1*R6*R8*gm2*rds2*rds4 + R2*R5*R8*gm2*rds2*rds4 + R1*R7*R8*gm2*rds2*rds4 + R2*R6*R8*gm2*rds2*rds4 + R2*R7*R8*gm2*rds2*rds4 + R1*R5*R18*gm4*rds2*rds4 + R1*R6*R18*gm4*rds2*rds4 + R2*R5*R18*gm4*rds2*rds4 + R1*R7*R18*gm4*rds2*rds4 + R2*R6*R18*gm4*rds2*rds4 + R2*R7*R18*gm4*rds2*rds4 + R1*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R5*R8*R18*gm2*gm4*rds2*rds4 + R1*R7*R8*R18*gm2*gm4*rds2*rds4 + R2*R6*R8*R18*gm2*gm4*rds2*rds4 + R2*R7*R8*R18*gm2*gm4*rds2*rds4)]
 
    H_32_s = sig.TransferFunction(num_32_coeffs, den_32_coeffs)
    dt = 1/f_samp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.linalg.LinAlgWarning)
        H_32_z = H_32_s.to_discrete(dt=dt, method='bilinear')

    num = H_32_z.num
    den = H_32_z.den

    if len(num) < len(den):
        # Pad with leading zeros
        num = np.pad(num, (len(den) - len(num), 0), 'constant')

    _, H_resp = sig.freqz(num, den, worN=frequencies_hz, fs=f_samp)

    return H_resp



def H43_TF_frequency_response_from_circuit_params_hardware(circuit_params, frequencies_hz, f_samp):
    '''
    Calculate the frequency response of H_43 based on the circuit parameters.

    Parameters:
    -----------
    circuit_params (dict): A dictionary containing the circuit parameters.
    frequencies_hz (array-like): An array of frequencies (in Hz) at which to evaluate the frequency response.
    f_samp (float): The sampling frequency (in Hz) for the discrete-time representation.

    Returns:
    --------
    H_resp (array): The frequency response of H_43 at the specified frequencies.
    '''
    R1 = circuit_params['R1']
    R2 = circuit_params['R2']
    R3 = circuit_params['R3']
    R4 = circuit_params['R4']
    R5 = circuit_params['R5']
    R6 = circuit_params['R6']
    C1 = circuit_params['C1']
    C2 = circuit_params['C2']
    gm5 = circuit_params['gm5']
    rds5 = circuit_params['rds5']
    
    num_43_coeffs = [-C1*C2*R1*R2*R3*R4*R6*gm5*rds5, -C2*R1*R2*R3*R6*gm5*rds5, 0]
    den_43_coeffs = [(C1*R3*R4 + C1*R4*rds5)*(C2*R1*R2*R5 + C2*R1*R2*R6 + C2*R1*R5*R6 + C2*R2*R5*R6), (R3 + R4 + rds5 + R4*gm5*rds5)*(C2*R1*R2*R5 + C2*R1*R2*R6 + C2*R1*R5*R6 + C2*R2*R5*R6) + (C1*R3*R4 + C1*R4*rds5)*(R1*R5 + R1*R6 + R2*R5 + R2*R6), (R3 + R4 + rds5 + R4*gm5*rds5)*(R1*R5 + R1*R6 + R2*R5 + R2*R6)]
    
    H_43_s = sig.TransferFunction(num_43_coeffs, den_43_coeffs)
    dt = 1/f_samp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=scipy.linalg.LinAlgWarning)
        H_43_z = H_43_s.to_discrete(dt=dt, method='bilinear')

    num = H_43_z.num
    den = H_43_z.den

    if len(num) < len(den):
        # Pad with leading zeros
        num = np.pad(num, (len(den) - len(num), 0), 'constant')

    _, H_resp = sig.freqz(num, den, worN=frequencies_hz, fs=f_samp)

    return H_resp



def find_faulty_parameter_H_31_hardware(list_of_params_to_check, frequencies_hz, Estimated_H_31, f_samp):
    '''
    Find the faulty parameter in the H_31 of circuit by comparing the estimated frequency response with the theoretical frequency response calculated from the circuit parameters.
    
    Parameters:
    -----------
    list_of_params_to_check (list): A list of parameter names to check.
    frequencies_hz (array-like): An array of frequencies (in Hz) at which to evaluate the frequency response.
    Estimated_H_31 (array): The estimated frequency response of H_31.
    f_samp (float): The sampling frequency (in Hz) for the discrete-time representation.

    Returns:
    --------
    faulty_param (list): A list containing the name of the faulty parameter and its value.
    '''
    min_err_param_dict = {}
    error_dict = {}
    for param in list_of_params_to_check:
        Circ_params = reset_hardware_circ_params()
        
        param_uLim = Circ_params[param] + Circ_params[param]*0.99
        param_lLim = Circ_params[param] - Circ_params[param]*0.99
        param_range = np.linspace(param_lLim,param_uLim,10000)
        error_min = np.inf
        
        for param_instance in param_range:
            Circ_params[param] = param_instance
            h = H31_TF_frequency_response_from_circuit_params_hardware(Circ_params, frequencies_hz, f_samp)
            error_vector = h - Estimated_H_31
            error_vector_at_considered_freqs = error_vector
            error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
            
            if error_temp < error_min:
                error_min = error_temp
                min_err_param_dict[param] = param_instance
                error_dict[param] = error_min
                     
        print(f"lowest error with {param} = {error_min}")

    faulty_param = {}
    
    lowest_key = min(error_dict, key=error_dict.get)
    faulty_param[lowest_key] = min_err_param_dict[lowest_key]
                
    return faulty_param
    


def find_faulty_parameter_H_32_hardware(list_of_params_to_check, frequencies_hz, Estimated_H_32, f_samp):
    '''
    Find the faulty parameter in the H_32 of circuit by comparing the estimated frequency response with the theoretical frequency response calculated from the circuit parameters.
    
    Parameters:
    -----------
    list_of_params_to_check (list): A list of parameter names to check.
    frequencies_hz (array-like): An array of frequencies (in Hz) at which to evaluate the frequency response.
    Estimated_H_32 (array): The estimated frequency response of H_32.
    f_samp (float): The sampling frequency (in Hz) for the discrete-time representation.

    Returns:
    --------
    faulty_param (list): A list containing the name of the faulty parameter and its value.
    '''
    min_err_param_dict = {}
    error_dict = {}
    for param in list_of_params_to_check:
        Circ_params = reset_hardware_circ_params()
        
        param_uLim = Circ_params[param] + Circ_params[param]*0.99
        param_lLim = Circ_params[param] - Circ_params[param]*0.99
        param_range = np.linspace(param_lLim,param_uLim,10000)
        error_min = np.inf
        
        for param_instance in param_range:
            Circ_params[param] = param_instance
            h = H32_TF_frequency_response_from_circuit_params_hardware(Circ_params, frequencies_hz, f_samp)
            error_vector = h - Estimated_H_32
            error_vector_at_considered_freqs = error_vector
            error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
            
            if error_temp < error_min:
                error_min = error_temp
                min_err_param_dict[param] = param_instance
                error_dict[param] = error_min
                     
        print(f"lowest error with {param} = {error_min}")

    faulty_param = {}
    
    lowest_key = min(error_dict, key=error_dict.get)
    faulty_param[lowest_key] = min_err_param_dict[lowest_key]
                
    return faulty_param



def find_faulty_parameter_H_43_hardware(list_of_params_to_check, frequencies_hz, Estimated_H_43, f_samp):
    '''
    Find the faulty parameter in the H_43 of circuit by comparing the estimated frequency response with the theoretical frequency response calculated from the circuit parameters.

    Parameters:
    -----------
    list_of_params_to_check (list): A list of parameter names to check.
    frequencies_hz (array-like): An array of frequencies (in Hz) at which to evaluate the frequency response.
    Estimated_H_43 (array): The estimated frequency response of H_43.
    f_samp (float): The sampling frequency (in Hz) for the discrete-time representation.

    Returns:
    --------
    faulty_param (list): A list containing the name of the faulty parameter and its value.
    '''
    min_err_param_dict = {}
    error_dict = {}
    for param in list_of_params_to_check:
        Circ_params = reset_hardware_circ_params()
        
        param_uLim = Circ_params[param] + Circ_params[param]*0.99
        param_lLim = Circ_params[param] - Circ_params[param]*0.99
        param_range = np.linspace(param_lLim,param_uLim,10000)
        error_min = np.inf
        
        for param_instance in param_range:
            Circ_params[param] = param_instance
            h = H43_TF_frequency_response_from_circuit_params_hardware(Circ_params, frequencies_hz, f_samp)
            error_vector = h - Estimated_H_43
            error_vector_at_considered_freqs = error_vector
            error_temp = np.sum(np.abs(error_vector_at_considered_freqs))/len(error_vector_at_considered_freqs)
            
            if error_temp < error_min:
                error_min = error_temp
                min_err_param_dict[param] = param_instance
                error_dict[param] = error_min

        print(f"lowest error with {param} = {error_min}")

    faulty_param = {}

    lowest_key = min(error_dict, key=error_dict.get)
    faulty_param[lowest_key] = min_err_param_dict[lowest_key]
                
    return faulty_param



def find_faulty_parameter_hardware(row_num, data_Normal, Adj, n_FFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim):
    '''
    This function identifies the faulty parameter in the hardware circuit by comparing the estimated frequency response of the healthy and faulty circuits with the theoretical frequency response calculated from the circuit parameters.
    
    Parameters:
    -----------
    row_num (int): The row number in the Excel file corresponding to the faulty data.
    data_Normal (array): The time series data of the healthy circuit.
    Adj (array): The adjacency matrix of the circuit.
    n_FFT (int): The number of points to use in the FFT calculation.
    f_samp (float): The sampling frequency (in Hz) for the discrete-time representation.
    nperseg (int): The length of each segment for the Welch method.
    noverlap (int): The number of points to overlap between segments for the Welch method.
    win (str or tuple or array_like): Desired window to use. See scipy.signal.get_window for more details.
    freq_lLim (float): The lower frequency limit (in Hz) for considering the frequency response in the fault diagnosis.
    freq_uLim (float): The upper frequency limit (in Hz) for considering the frequency response in the fault diagnosis.
    
    Returns:
    --------
    faulty_param (list): A list containing the name of the faulty parameter and its value.
    '''

    time_series_path = f"data/time_series_data/hardware_circuit/faulty"
    data_Faulty = pd.read_excel(f'{time_series_path}/V_row{row_num}.xlsx').iloc[:, 1::2].values
    # Scaling factors/gains used in data acquisition system. We need to compensate 
    # for scaling of the instrumentation amplifiers and other filters in the measurement circuit.
    # The filters have unity gain between 15Hz and 100kHz, therefore we need to compensate for the
    # instrumentation amplifier's gain. The gains of the instrumentation amplifiers listed on data sets
    # may vary within tolerance because of resistance tolerance of the R_G. We need to calibrate the gain 
    # and fine tune it in order to get proper scaling. Therefore, we choose the multiplier in the following
    # format (total datasheet gain)*(calibration factor due to component tolerance)
    data_scale_fac_1 = (991*1.99)*(1.15)
    data_scale_fac_2 = (991*1.99)*(1.15)
    data_scale_fac_3 = (991*1.99)*(1)
    data_scale_fac_4 = (496*1.99)*(0.9)
    data_Faulty[:,0] = data_Faulty[:,0]*(1/data_scale_fac_1)
    data_Faulty[:,1] = data_Faulty[:,1]*(1/data_scale_fac_2)
    data_Faulty[:,2] = data_Faulty[:,2]*(1/data_scale_fac_3)
    data_Faulty[:,3] = data_Faulty[:,3]*(1/data_scale_fac_4)

    H_healthy_est, H_faulty_est, f = estimate_tf_PSD(data_Normal, data_Faulty, Adj, n_FFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim)
    H_avg_mag_diff, _ = get_tf_diff(H_healthy_est, H_faulty_est, Adj, f)

    max_mag_key = max(H_avg_mag_diff, key=H_avg_mag_diff.get)

    index_uLim = np.searchsorted(f,freq_uLim, side='right')
    index_lLim = np.searchsorted(f,freq_lLim, side='right')
    frequencies_hz = f[index_lLim:index_uLim]
    
    if max_mag_key == "H_31":
        print("Detected fault in H_31")
        list_of_params_to_check = ['R1', 'R2', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'C2', 'C3', 'C4']
        H_31_faulty_est = H_faulty_est[2,0,:]
        faulty_param = find_faulty_parameter_H_31_hardware(list_of_params_to_check, frequencies_hz, H_31_faulty_est[index_lLim:index_uLim], f_samp)

    elif max_mag_key == "H_32":
        print("Detected fault in H_32")
        list_of_params_to_check = ['R1', 'R2', 'R5', 'R6', 'R7', 'R17', 'R18', 'R19', 'R20', 'R21', 'C2', 'C6', 'C7']
        H_32_faulty_est = H_faulty_est[2,1,:]
        faulty_param = find_faulty_parameter_H_32_hardware(list_of_params_to_check, frequencies_hz, H_32_faulty_est[index_lLim:index_uLim], f_samp)
    
    elif max_mag_key == "H_43":
        print("Detected fault in H_43")
        list_of_params_to_check = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'C1', 'C2']
        H_43_faulty_est = H_faulty_est[3,2,:]
        faulty_param = find_faulty_parameter_H_43_hardware(list_of_params_to_check, frequencies_hz, H_43_faulty_est[index_lLim:index_uLim], f_samp)

    return faulty_param