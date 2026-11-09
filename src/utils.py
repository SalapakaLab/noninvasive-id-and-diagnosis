import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



def data_FFT(time_domain_data,nSamples,n_FFT):
    '''
    Converts time domain data to frequency domain using FFT
    
    Parameters:
    ----------
    time_domain_data: numpy array of shape (nSamples*n_FFT, nNodes) representing the time domain data
    nSamples: integer representing the number of samples
    n_FFT: integer representing the number of FFT points
    
    Returns:
    -------
    y: numpy array of shape (nSamples, nNodes, n_FFT) representing the frequency domain data
    '''

    nNodes = time_domain_data.shape[1]
    y = np.zeros((nSamples,nNodes,n_FFT),dtype=complex)
    temp_ind = 0
    for ind_Traj in range(nSamples):
        x = time_domain_data[temp_ind:temp_ind+n_FFT:1,:]
        temp_ind = temp_ind + n_FFT
        X = np.fft.fft(x,axis=0)
        y[ind_Traj,:,:] = np.transpose(X)
    return y



def manual_unwrap(p):
    '''
    Manually unwraps phase values to handle discontinuities in a specific range.
    Adjusts phase values to ensure they stay within the range [-2.5π, -π/6],
    preventing discontinuous jumps that can occur in phase measurements.
    
    Parameters:
    ----------
    p: numpy array of phase values (in radians)
    
    Returns:
    -------
    unwrapped_p: numpy array of unwrapped phase values (in radians)
    '''
    
    unwrapped_p = np.copy(p)
    for i in range(1, len(p)):
        while unwrapped_p[i] > -np.pi/6:
            unwrapped_p[i] -= 2*np.pi
        while unwrapped_p[i] < -2.5*np.pi:
            unwrapped_p[i] += 2*np.pi
    return unwrapped_p



def manual_unwrap_eight_stag_TF(p):
    '''
    Manually unwraps phase values to handle discontinuities in a specific range.
    Adjusts phase values to ensure they stay within the range [-2.5π, -π/6],
    preventing discontinuous jumps that can occur in phase measurements.
    
    Parameters:
    ----------
    p: numpy array of phase values (in radians)
    
    Returns:
    -------
    unwrapped_p: numpy array of unwrapped phase values (in radians)
    '''
    
    unwrapped_p = np.copy(p)
    for i in range(1, len(p)):
        while unwrapped_p[i] > 0:
            unwrapped_p[i] -= 2*np.pi
        while unwrapped_p[i] < -2.05*np.pi:
            unwrapped_p[i] += 2*np.pi
    return unwrapped_p



def normalize_data(data):
    '''
    Normalizes the data by centering it to have zero mean and scaling it to have unit variance.

    Parameters:
    ----------
    data: numpy array of shape (nSamples, nNodes) representing the data to be normalized
    
    Returns:
    -------
    normalizedData: numpy array of shape (nSamples, nNodes) representing the normalized data
    '''

    col_means = np.mean(data,axis=0)
    col_stds = np.std(data,axis=0)
    normalizedData = (data - col_means)/col_stds
    return normalizedData



def add_correlated_noise(data, snr_db, rho, rng=None):
    '''
    Adds equicorrelated Gaussian noise to each column of data at a specified per-channel SNR.

    Noise across channels is drawn with pairwise correlation coefficient rho (via the equicorrelation
    matrix R = (1-rho)*I + rho*ones, requires -1/(nNodes-1) < rho < 1), then each channel is scaled
    independently so its noise power matches snr_db relative to that channel's own signal variance.

    Parameters:
    ----------
    data: nSamples X nNodes array of clean time-series data
    snr_db: desired signal-to-noise ratio in dB, computed per channel from that channel's variance
    rho: target pairwise correlation coefficient between noise channels
    rng: numpy.random.Generator used to draw the noise (default: a fresh np.random.default_rng())

    Returns:
    -------
    noisy_data: nSamples X nNodes array, data plus correlated noise
    '''

    if rng is None:
        rng = np.random.default_rng()
    nSamples, nNodes = data.shape
    R = (1 - rho) * np.eye(nNodes) + rho * np.ones((nNodes, nNodes))
    L = np.linalg.cholesky(R)
    white_noise = rng.standard_normal((nSamples, nNodes))
    corr_noise = white_noise @ L.T
    signal_power = np.var(data, axis=0)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy_data = data + corr_noise * np.sqrt(noise_power)
    return noisy_data



def get_confusion_matrix(true_graph,estimated_graph):
    '''
    Computes the confusion matrix (TP, FP, TN, FN) for the estimated graph compared to the true graph.

    Parameters:
    ----------
    true_graph: numpy array of shape (nNodes, nNodes) representing the true graph
    estimated_graph: numpy array of shape (nNodes, nNodes) representing the estimated graph

    Returns:
    -------
    TP: integer representing the number of true positives
    FP: integer representing the number of false positives
    TN: integer representing the number of true negatives
    FN: integer representing the number of false negatives
    '''

    num_nodes = true_graph.shape[0]
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i!=j:
                if true_graph[i,j] == 1 and estimated_graph[i,j] == 1:
                    TP += 1
                if true_graph[i,j] == 1 and estimated_graph[i,j] == 0:
                    FN += 1
                if true_graph[i,j] == 0 and estimated_graph[i,j] == 1:
                    FP += 1
                if true_graph[i,j] == 0 and estimated_graph[i,j] == 0:
                    TN += 1

    return TP, FP, TN, FN



def find_moral_graph_and_skeleton_from_adjacency_matrix(Adj):
    '''
    Computes the moral graph and skeleton from the given adjacency matrix of a directed graph.
    
    Parameters:
    ----------
    Adj: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the directed graph

    Returns:
    -------
    MG: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the estimated moral graph
    Skeleton: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the estimated skeleton
    '''

    nNodes=Adj.shape[0]
    Skeleton=(Adj+Adj.T)
    MG=np.zeros([nNodes,nNodes],dtype=bool)
    MG=np.copy(Skeleton)
    for ind1 in range(nNodes):
        for ind2 in range(nNodes):
            for ind3 in range(nNodes):
                if (Adj[ind1,ind2]==1) and ((Adj[ind1,ind3]==1)) and ind2!=ind3:
                    MG[ind2,ind3]=1
                    MG[ind3,ind2]=1
    return MG,Skeleton



def save_pag_recons_metrics(save_path, nSamplesList, skeleton_metrics, edge_metrics):
    '''
    Saves PAG reconstruction accuracy metrics (skeleton + one-vs-all edge-mark metrics) to a JSON file.

    Parameters:
    ----------
    save_path: string path of the .json file to write (parent directories are created if needed)
    nSamplesList: list of sample sizes used across the sweep
    skeleton_metrics: dict mapping metric name ('f1', 'tpr', 'fpr', 'precision', 'recall') to a list of values, one per entry in nSamplesList
    edge_metrics: dict mapping edge-mark label (e.g. 'no_edge', 'o-o', '-->') to a dict of metric name -> list of values, one per entry in nSamplesList
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "nSamplesList": list(nSamplesList),
        "skeleton": skeleton_metrics,
        "edge_marks": edge_metrics,
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)



def load_pag_recons_metrics(save_path):
    '''
    Loads PAG reconstruction accuracy metrics previously saved by save_pag_recons_metrics.

    Parameters:
    ----------
    save_path: string path of the .json file to read

    Returns:
    -------
    data: dict with keys "nSamplesList", "skeleton", "edge_marks" (see save_pag_recons_metrics)
    '''

    with open(save_path, 'r') as f:
        return json.load(f)



def save_threshold_sensitivity_data(save_path, thresh_vec, nSamplesList, f1_results, auto_thresh_results):
    '''
    Saves W-PC threshold-sweep F1 curves and their auto-selected thresholds to a JSON file, so
    plots can be restyled later without re-running the reconstruction sweep.

    Parameters:
    ----------
    save_path: string path of the .json file to write (parent directories are created if needed)
    thresh_vec: 1D array/list of threshold values swept
    nSamplesList: list of sample sizes used across the sweep
    f1_results: dict mapping nSamples -> list/array of F1 scores (one per entry in thresh_vec)
    auto_thresh_results: dict mapping nSamples -> auto-selected threshold value (or None)
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "thresh_vec": np.asarray(thresh_vec).tolist(),
        "nSamplesList": list(nSamplesList),
        "f1_results": {str(n): np.asarray(f1_results[n]).tolist() for n in nSamplesList},
        "auto_thresh_results": {str(n): auto_thresh_results[n] for n in nSamplesList},
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)



def load_threshold_sensitivity_data(save_path):
    '''
    Loads threshold-sweep F1 data previously saved by save_threshold_sensitivity_data.

    Parameters:
    ----------
    save_path: string path of the .json file to read

    Returns:
    -------
    thresh_vec: numpy array of threshold values swept
    nSamplesList: list of sample sizes used across the sweep
    f1_results: dict mapping nSamples (int) -> numpy array of F1 scores
    auto_thresh_results: dict mapping nSamples (int) -> auto-selected threshold value (or None)
    '''

    with open(save_path, 'r') as f:
        data = json.load(f)
    thresh_vec = np.array(data["thresh_vec"])
    nSamplesList = data["nSamplesList"]
    f1_results = {n: np.array(data["f1_results"][str(n)]) for n in nSamplesList}
    auto_thresh_results = {n: data["auto_thresh_results"][str(n)] for n in nSamplesList}
    return thresh_vec, nSamplesList, f1_results, auto_thresh_results



def save_noise_robustness_data(save_path, SNR_dB_list, snr_metrics, rho_list, SNR_dB_list_for_rho_sweep, rho_metrics_by_snr):
    '''
    Saves noise-robustness sweep results (F1/TPR/FPR vs. SNR, and vs. correlation coefficient at
    several fixed SNRs) to a JSON file, so plots can be restyled later without re-running the
    reconstruction sweeps.

    Parameters:
    ----------
    save_path: string path of the .json file to write (parent directories are created if needed)
    SNR_dB_list: list of SNR values (dB) swept at fixed noise correlation
    snr_metrics: dict mapping metric name ('tpr','fpr','precision','recall','f1') to a list of
        values, one per entry in SNR_dB_list
    rho_list: list of noise correlation coefficients swept at each fixed SNR
    SNR_dB_list_for_rho_sweep: list of SNR values (dB) at which the correlation-coefficient sweep was run
    rho_metrics_by_snr: dict mapping SNR value -> metrics dict (same shape as snr_metrics), one per
        entry in rho_list
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "SNR_dB_list": list(SNR_dB_list),
        "snr_metrics": {key: list(vals) for key, vals in snr_metrics.items()},
        "rho_list": list(rho_list),
        "SNR_dB_list_for_rho_sweep": list(SNR_dB_list_for_rho_sweep),
        "rho_metrics_by_snr": {
            str(snr): {key: list(vals) for key, vals in metrics.items()}
            for snr, metrics in rho_metrics_by_snr.items()
        },
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)



def load_noise_robustness_data(save_path):
    '''
    Loads noise-robustness sweep results previously saved by save_noise_robustness_data.

    Parameters:
    ----------
    save_path: string path of the .json file to read

    Returns:
    -------
    SNR_dB_list: list of SNR values (dB) swept at fixed noise correlation
    snr_metrics: dict mapping metric name -> list of values, one per entry in SNR_dB_list
    rho_list: list of noise correlation coefficients swept at each fixed SNR
    SNR_dB_list_for_rho_sweep: list of SNR values (dB) at which the correlation-coefficient sweep was run
    rho_metrics_by_snr: dict mapping SNR value -> metrics dict, one per entry in rho_list
    '''

    with open(save_path, 'r') as f:
        data = json.load(f)
    SNR_dB_list = data["SNR_dB_list"]
    snr_metrics = data["snr_metrics"]
    rho_list = data["rho_list"]
    SNR_dB_list_for_rho_sweep = data["SNR_dB_list_for_rho_sweep"]
    rho_metrics_by_snr = {snr: data["rho_metrics_by_snr"][str(snr)] for snr in SNR_dB_list_for_rho_sweep}
    return SNR_dB_list, snr_metrics, rho_list, SNR_dB_list_for_rho_sweep, rho_metrics_by_snr



def save_freq_error_sweep_data(save_path, x_values, freq_by_tf, mag_diff_by_tf, phase_diff_by_tf):
    '''
    Saves frequency-resolved transfer-function estimation error across a swept parameter (e.g.
    SNR or sample size), one curve per sweep value per transfer function, to a JSON file so plots
    can be restyled later without re-running the (expensive) estimation sweep.

    Parameters:
    ----------
    save_path: string path of the .json file to write (parent directories are created if needed)
    x_values: list of swept parameter values (e.g. SNR in dB, or sample size)
    freq_by_tf: dict mapping transfer-function label -> 1D array of AC-sim frequency points (the
        same frequency grid is used for every value in x_values, since it only depends on the TF)
    mag_diff_by_tf: dict mapping transfer-function label -> list of 1D magnitude-error arrays (dB),
        one per entry in x_values, each the same length as freq_by_tf[label]
    phase_diff_by_tf: dict mapping transfer-function label -> list of 1D phase-error arrays (deg),
        same shape as mag_diff_by_tf
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "x_values": list(x_values),
        "freq_by_tf": {label: np.asarray(freq).tolist() for label, freq in freq_by_tf.items()},
        "mag_diff_db_by_tf": {
            label: [np.asarray(arr).tolist() for arr in arrs] for label, arrs in mag_diff_by_tf.items()
        },
        "phase_diff_deg_by_tf": {
            label: [np.asarray(arr).tolist() for arr in arrs] for label, arrs in phase_diff_by_tf.items()
        },
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)



def load_freq_error_sweep_data(save_path):
    '''
    Loads a frequency-resolved error sweep previously saved by save_freq_error_sweep_data.

    Parameters:
    ----------
    save_path: string path of the .json file to read

    Returns:
    -------
    x_values: list of swept parameter values
    freq_by_tf: dict mapping transfer-function label -> numpy array of frequency points
    mag_diff_by_tf: dict mapping transfer-function label -> list of numpy magnitude-error arrays,
        one per entry in x_values
    phase_diff_by_tf: dict mapping transfer-function label -> list of numpy phase-error arrays,
        one per entry in x_values
    '''

    with open(save_path, 'r') as f:
        data = json.load(f)
    x_values = data["x_values"]
    freq_by_tf = {label: np.array(freq) for label, freq in data["freq_by_tf"].items()}
    mag_diff_by_tf = {label: [np.array(arr) for arr in arrs] for label, arrs in data["mag_diff_db_by_tf"].items()}
    phase_diff_by_tf = {label: [np.array(arr) for arr in arrs] for label, arrs in data["phase_diff_deg_by_tf"].items()}
    return x_values, freq_by_tf, mag_diff_by_tf, phase_diff_by_tf



def find_true_Essential_graph(Skeleton, Adj):
    '''
    Computes the true essential graph from the given skeleton and adjacency matrix of a directed graph.
    
    Parameters:
    ----------
    Skeleton: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the skeleton
    Adj: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the directed graph

    Returns:
    -------
    true_ess_graph: numpy array of shape (nNodes, nNodes) representing the adjacency matrix of the true essential graph
    '''
    
    num_nodes = Skeleton.shape[0]
    collider_graph = np.copy(Skeleton)
    collider_nodes = []
    for ind1 in range(num_nodes): # ind1 is the collider node index
        for ind2 in range(num_nodes):    
        #ind2 and ind3 are the indices for the parent of colliders
            for ind3 in range(num_nodes):
                if (Adj[ind1,ind2] == 1) and ((Adj[ind1,ind3] == 1)) and Skeleton[ind2,ind3] !=1 and ind2 != ind3 and ind1 != ind3 and ind1 != ind2:
                    # print( ind1," is a collider")
                    collider_graph[ind2,ind1]=0
                    collider_graph[ind3,ind1]=0
                    collider_nodes.append(ind1)                       
    # % Orienting other edges if possible
    true_ess_graph = np.copy(collider_graph)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i>j and collider_graph[i,j]==1 and collider_graph[j,i]==1:
                if i in collider_nodes:
                    true_ess_graph[i,j] = 0
                if j in collider_nodes:
                    true_ess_graph[j,i] = 0

    return true_ess_graph



def save_fault_diag_stat_validation_data(save_path, confusion_matrices, resistance_est_error, capacitance_est_error):
    '''
    Saves the biopotential sensor fault-diagnosis statistical validation results (per-scenario
    confusion matrices and parameter estimation errors) to a JSON file, so the confusion matrix
    and box plots can be restyled later without re-running the (expensive) validation sweep.

    Parameters:
    ----------
    save_path: string path of the .json file to write (parent directories are created if needed)
    confusion_matrices: dictionary mapping scenario name (e.g. 'row5') to a dict with keys
        'TP', 'FP', 'TN', 'FN'
    resistance_est_error: list of relative estimation errors for resistance parameters
    capacitance_est_error: list of relative estimation errors for capacitance parameters
    '''

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "confusion_matrices": confusion_matrices,
        "resistance_est_error": list(resistance_est_error),
        "capacitance_est_error": list(capacitance_est_error),
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)



def load_fault_diag_stat_validation_data(save_path):
    '''
    Loads biopotential sensor fault-diagnosis statistical validation results previously saved by
    save_fault_diag_stat_validation_data.

    Parameters:
    ----------
    save_path: string path of the .json file to read

    Returns:
    -------
    confusion_matrices: dictionary mapping scenario name to a dict with keys 'TP', 'FP', 'TN', 'FN'
    resistance_est_error: list of relative estimation errors for resistance parameters
    capacitance_est_error: list of relative estimation errors for capacitance parameters
    '''

    with open(save_path, 'r') as f:
        data = json.load(f)
    return data["confusion_matrices"], data["resistance_est_error"], data["capacitance_est_error"]
                    
    return true_ess_graph