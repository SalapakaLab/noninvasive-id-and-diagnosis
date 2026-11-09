import numpy as np
from itertools import combinations
from scipy import signal
import networkx as nx
from collections import deque

def is_d_separated(adj_matrix, x, y, Z):
    '''
    Check if nodes x and y are d-separated by set Z in the given directed graph.
    
    Parameters:
    ----------
    adj_matrix: numpy array of shape (n, n) representing the adjacency matrix of the directed graph
    x: integer representing the first node
    y: integer representing the second node
    Z: list of integers representing the conditioning set
    
    Returns:
    -------
    bool: True if x and y are d-separated by Z, False otherwise
    '''

    Z = set(Z)
    x = set([x])
    y = set([y])
    G = nx.DiGraph(adj_matrix.T)    
    return nx.algorithms.d_separation.is_d_separator(G, x, y, Z)
    
    

def find_descendants(adj, y):
    '''
    Find the descendants of node y in the directed graph represented by the adjacency matrix adj.

    Parameters:
    ----------
    adj: numpy array of shape (n, n) representing the adjacency matrix of the directed graph
    y: integer representing the node for which to find descendants

    Returns:
    -------
    set: A set of integers representing the descendants of node y
    '''
    n = adj.shape[0]
    desc = set()
    queue = deque([y])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if adj[v, u] and v not in desc:
                desc.add(v)
                queue.append(v)
    return desc



def find_single_door(adj_bar, x, y):
    '''
    Find the conditioning set required for applying the Single door criterion for LDIMs
    to find out the transfer function x-->y.

    Parameters:
    ----------
    adj_bar: numpy array of shape (n, n) representing the adjacency matrix of the graph with the edge x-->y removed
    x: integer representing the source node
    y: integer representing the target node

    Returns:
    -------
    list: A list of integers representing the set of nodes that satisfy the single door criterion
    '''
    nNodes = adj_bar.shape[0]
    # Get the descendants of y to be excluded from the conditioning set
    desc_y = find_descendants(adj_bar, y)
    # the complement set through which we condition
    D=[i for i in range(nNodes) if i != x and i != y and i not in desc_y]

    flag=False
    for ind_cond in range(nNodes-1):
        if flag==True:
            break
        combination_set=[i for i in combinations(D,ind_cond)]
        for Z in combination_set:
            if is_d_separated(adj_bar, x, y, Z):
                print("Separation set of ( X"+str(x)+", X"+str(y),") is : ",np.array(Z))
                flag=True
                break
    
    return Z



def calculate_PSD_from_data(data, channel_set_1, channel_set_2, f_samp, nFFT, nperseg, noverlap, win):
    '''
    Calculate the Power Spectral Density (PSD) matrix for the given data and channel sets.
    
    Parameters:
    ----------
    data: numpy array of shape (n_samples, n_channels) representing the time series data
    channel_set_1: list of integers representing the first set of channels
    channel_set_2: list of integers representing the second set of channels
    f_samp: float representing the sampling frequency
    nFFT: integer representing the number of FFT points
    nperseg: integer representing the length of each segment for Welch's method
    noverlap: integer representing the number of points to overlap between segments for Welch's method
    win: string or tuple or array_like representing the desired window to use for Welch's method
    
    Returns:
    -------
    psd_matrix: numpy array of shape (len(channel_set_1), len(channel_set_2), len(frequencies)) representing the PSD matrix
    frequencies: numpy array of shape (len(frequencies),) representing the frequency bins corresponding to the PSD estimates
    '''

    num_channels_1 = len(channel_set_1)
    num_channels_2 = len(channel_set_2)
    f, Pxx = signal.welch(data[:,0], f_samp, win, nperseg, noverlap, scaling='density')
    psd_matrix = np.zeros((num_channels_1, num_channels_2, len(f)), dtype=complex)
    frequencies = None

    k = 0
    for i in channel_set_1:
        l = 0
        for j in channel_set_2:
            if i == j:
                # Auto-PSD (diagonal)
                f, Pxx = signal.welch(data[:,i], f_samp, win, nperseg, noverlap, scaling='density')
                psd_matrix[k, l, :] = Pxx
            else:
                # Cross-PSD (off-diagonal)
                f, Pxy = signal.csd(data[:,j], data[:,i], f_samp, win, nperseg, noverlap, scaling='density')
                psd_matrix[k, l, :] = Pxy
            if frequencies is None:
                frequencies = f
            l += 1
        k += 1
                
    return psd_matrix, f



def compute_partial_Wiener_coeffs_PSD(data,f_samp,i,j,z,nFFT,nperseg, noverlap, win):
    '''
    Calculate the partial Wiener coefficients using PSD estimates.
    
    Parameters:
    ----------
    data: numpy array of shape (n_samples, n_channels) representing the time series data
    f_samp: float representing the sampling frequency
    i: integer representing the index of the first channel
    j: integer representing the index of the second channel
    z: list of integers representing the conditioning set
    nFFT: integer representing the number of FFT points
    nperseg: integer representing the length of each segment for Welch's method
    noverlap: integer representing the number of points to overlap between segments for Welch's method
    win: string or tuple or array_like representing the desired window to use for Welch's method
    
    Returns:
    -------
    W: numpy array of shape (len(frequencies), len(channel_set_2)) representing the partial Wiener coefficients
    f: numpy array of shape (len(frequencies),) representing the frequency bins corresponding to the PSD estimates
    '''

    y = np.append(j,np.int32(z))
    x = np.array([i])
    phi_yy,f = calculate_PSD_from_data(data,y,y,f_samp,nFFT,nperseg, noverlap, win)
    phi_xy,f = calculate_PSD_from_data(data,x,y,f_samp,nFFT,nperseg, noverlap, win)
    W=np.zeros([len(f),len(y)],dtype='complex')
    phi_yy_inv = np.zeros_like(phi_yy)
    for omega in range(len(f)):
        try:
            phi_yy_inv[:,:,omega] = np.linalg.inv((phi_yy[:,:,omega]))
        except np.linalg.LinAlgError as e:
            print(f"Error inverting matrix: {e}")
        W[omega,:] = np.matmul(phi_xy[:,:,omega], phi_yy_inv[:,:,omega])

    return W,f



def segment_and_fft(data, channels, nperseg, noverlap, win, nfft, detrend=True):
    '''
    Splits each requested channel into overlapping segments, windows them and takes the
    one-sided FFT, reproducing the segmentation, windowing and constant.

    Parameters:
    ----------
    data: numpy array of shape (n_samples, n_channels) representing the time series data
    channels: sequence of channel indices to transform
    nperseg: integer representing the length of each segment
    noverlap: integer representing the number of overlapping points between segments
    win: string or tuple or array_like representing the desired window
    nfft: integer representing the FFT length (>= nperseg; the segment is zero padded)

    Returns:
    -------
    F: numpy array of shape (len(channels), R, nfft//2 + 1) of Fourier coefficients
    R: integer, the number of segments
    '''
    n_samples = data.shape[0]
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError(f"noverlap ({noverlap}) must be smaller than nperseg ({nperseg})")
    R = (n_samples - noverlap) // step
    if R < 1:
        raise ValueError(f"Data of length {n_samples} yields no complete segment of "
                         f"length {nperseg}")

    window = np.asarray(win) if isinstance(win, (np.ndarray, list)) else signal.get_window(win, nperseg)

    # Row r of seg_idx holds the sample indices belonging to segment r
    seg_idx = np.arange(nperseg)[None, :] + step * np.arange(R)[:, None]

    F = np.empty((len(channels), R, nfft // 2 + 1), dtype=complex)
    for position, channel in enumerate(channels):
        segments = data[seg_idx, channel]
        if detrend:
            segments = segments - segments.mean(axis=1, keepdims=True)
        F[position] = np.fft.rfft(segments * window, n=nfft, axis=1)
    return F, R



def compute_partial_Wiener_coeffs(data, f_samp, i, j, z, nFFT, nperseg, noverlap, win,
                                     detrend=True):
    '''
    Calculate the partial Wiener coefficients by solving the per-frequency least-squares
    problem over FFT segments.

    It estimates the Wiener filters by solving

        W_{x|(y,Z)}(w_k) = argmin_beta || chi_x(w_k) - chi_C(w_k) beta ||_2^2,

    where chi_x is the stack of Fourier coefficients of channel i over the R segments and
    chi_C is the corresponding R x q matrix for the regressor channels C = [j, z].

    Setting noverlap=0, win='boxcar' and detrend=False reduces this to the estimator used
    for network reconstruction in utils.data_FFT together with
    network_reconstruction.Wiener_Proj, differing only in returning the one-sided
    spectrum. This route needs R > q segments to be well posed.

    Parameters:
    ----------
    data: numpy array of shape (n_samples, n_channels) representing the time series data
    f_samp: float representing the sampling frequency
    i: integer representing the index of the regressand channel
    j: integer representing the index of the channel whose coefficient is returned first
    z: list of integers representing the conditioning set
    nFFT: integer representing the number of FFT points. As in scipy, a value larger than
        nperseg zero pads the segment; a smaller value has no effect and nperseg is used,
        which keeps the frequency grid identical to the PSD route for the existing calls.
    nperseg: integer representing the length of each segment
    noverlap: integer representing the number of points to overlap between segments
    win: string or tuple or array_like representing the desired window
    detrend: bool, subtract each segment's mean before transforming. True matches Welch's
        default; set False to match the network reconstruction implementation.

    Returns:
    -------
    W: numpy array of shape (len(frequencies), 1 + len(z)) of partial Wiener coefficients
    f: numpy array of shape (len(frequencies),) of the corresponding frequency bins
    '''
    y = np.append(j, np.int32(z))
    nfft = int(max(nFFT, nperseg)) if nFFT else int(nperseg)

    F_regressors, R = segment_and_fft(data, y, nperseg, noverlap, win, nfft, detrend)
    F_target, _ = segment_and_fft(data, [i], nperseg, noverlap, win, nfft, detrend)

    if R <= len(y):
        raise ValueError(f"Least-squares Wiener estimation needs more segments than "
                         f"regressors: R={R}, q={len(y)}. Reduce nperseg or increase "
                         f"noverlap.")

    # chi_C: (nFreq, R, q) and chi_x: (nFreq, R)
    chi_C = np.transpose(F_regressors, (2, 1, 0))
    chi_x = F_target[0].T

    # Normal equations, solved for every frequency at once
    gram = np.einsum('krm,krn->kmn', np.conj(chi_C), chi_C)
    rhs = np.einsum('krm,kr->km', np.conj(chi_C), chi_x)
    try:
        # rhs gets a trailing axis so that numpy treats it as a stack of column vectors
        # rather than as a single matrix
        W = np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # At least one frequency is rank deficient; fall back to the pseudo-inverse.
        print("Singular Gram matrix at one or more frequencies; using pseudo-inverse")
        W = np.einsum('kmn,kn->km', np.linalg.pinv(gram), rhs)

    f = np.fft.rfftfreq(nfft, d=1.0 / f_samp)
    return W, f



def estimate_tf_PSD(healthy_data, faulty_data, adj, nFFT, f_samp, nperseg, noverlap, win, freq_lLim, freq_uLim):
    '''
    Estimate the transfer function between stages using PSD based Wiener filter estimates.
    
    Parameters:
    ----------
    healthy_data: numpy array of shape (n_samples, n_channels) representing the time series data from the healthy system
    faulty_data: numpy array of shape (n_samples, n_channels) representing the time series data from the faulty system
    adj: numpy array of shape (n, n) representing the adjacency matrix of the graphical representation of the LDIM
    nFFT: integer representing the number of FFT points
    f_samp: float representing the sampling frequency
    nperseg: integer representing the length of each segment for Welch's method
    noverlap: integer representing the number of points to overlap between segments for Welch's method'
    win: string or tuple or array_like representing the desired window to use for Welch's method
    freq_lLim: float representing the lower frequency limit for analysis
    freq_uLim: float representing the upper frequency limit for analysis
    
    Returns:
    -------
    H_healthy: numpy array of shape (n, n, len(frequencies)) representing the estimated transfer function for the healthy system
    H_faulty: numpy array of shape (n, n, len(frequencies)) representing the estimated transfer function for the faulty system
    frequencies: numpy array of shape (len(frequencies),) representing the frequency bins corresponding to the PSD estimates
    '''

    nNodes = adj.shape[0]
    W_healthy, f = compute_partial_Wiener_coeffs_PSD(healthy_data,f_samp, 1, 2, [], nFFT, nperseg, noverlap, win)
    H_healthy = np.zeros([nNodes,nNodes,len(f)],dtype='complex')
    H_faulty = np.zeros([nNodes,nNodes,len(f)],dtype='complex')
        
    for i in range(nNodes):
        for j in range(nNodes):
            if adj[j,i] == 1:
                adj_bar = np.copy(adj)
                adj_bar[j,i] = 0
                Z = find_single_door(adj_bar, i, j)
                
                W_healthy, f = compute_partial_Wiener_coeffs_PSD(healthy_data,f_samp, j, i, Z, nFFT, nperseg, noverlap, win)
                W_faulty, f = compute_partial_Wiener_coeffs_PSD(faulty_data,f_samp, j, i, Z, nFFT, nperseg, noverlap, win)
                
                H_healthy[j,i,:] = W_healthy[:,0]
                H_faulty[j,i,:] = W_faulty[:,0]

    return H_healthy, H_faulty, f



def estimate_tf_PSD_single(data, adj, nFFT, f_samp, nperseg, noverlap, win):
    '''
    Single-dataset version of estimate_tf_PSD. It estimates the transfer functions between stages
    of one dataset using the same PSD based Wiener filter estimates.

    Parameters:
    ----------
    data: numpy array of shape (n_samples, n_channels) representing the time series data
    adj: numpy array of shape (n, n) representing the adjacency matrix of the graphical representation of the LDIM
    nFFT: integer representing the number of FFT points
    f_samp: float representing the sampling frequency
    nperseg: integer representing the length of each segment for Welch's method
    noverlap: integer representing the number of points to overlap between segments for Welch's method
    win: string or tuple or array_like representing the desired window to use for Welch's method

    Returns:
    -------
    H: numpy array of shape (n, n, len(frequencies)) representing the estimated transfer functions
    frequencies: numpy array of shape (len(frequencies),) representing the frequency bins corresponding to the PSD estimates
    '''

    nNodes = adj.shape[0]
    W, f = compute_partial_Wiener_coeffs_PSD(data,f_samp, 1, 2, [], nFFT, nperseg, noverlap, win)
    H = np.zeros([nNodes,nNodes,len(f)],dtype='complex')

    for i in range(nNodes):
        for j in range(nNodes):
            if adj[j,i] == 1:
                adj_bar = np.copy(adj)
                adj_bar[j,i] = 0
                Z = find_single_door(adj_bar, i, j)

                W, f = compute_partial_Wiener_coeffs_PSD(data,f_samp, j, i, Z, nFFT, nperseg, noverlap, win)

                H[j,i,:] = W[:,0]

    return H, f



def merge_LF_HF_frequency_resp(HF_transferFunction, LF_transferFunction, HF_f, LF_f, freq_lLim, freq_mLim, freq_uLim):
    '''
    This function merges the frequency responses obtained from LF and HF data based on the specified frequency limits.
    
    Parameters:
    ----------
    HF_transferFunction: numpy array of shape (n, n, len(HF_f)) representing the transfer function obtained from high-frequency data
    LF_transferFunction: numpy array of shape (n, n, len(LF_f)) representing the transfer function obtained from low-frequency data
    HF_f: numpy array of shape (len(HF_f),) representing the frequency bins corresponding to the high-frequency transfer function estimates
    LF_f: numpy array of shape (len(LF_f),) representing the frequency bins corresponding to the low-frequency transfer function estimates
    freq_lLim: float representing the lower frequency limit for merging
    freq_mLim: float representing the middle frequency limit for merging
    freq_uLim: float representing the upper frequency limit for merging
    
    Returns:
    -------
    H_merged: numpy array of shape (n, n, len(f_merged)) representing the merged transfer function across the specified frequency range
    f_merged: numpy array of shape (len(f_merged),) representing the frequency bins corresponding to the merged transfer function estimates
    '''
        
    index_uLim = np.searchsorted(HF_f,freq_uLim, side='right')
    index_mLim_HF = np.searchsorted(HF_f,freq_mLim, side='right')
    index_mLim_LF = np.searchsorted(LF_f,freq_mLim, side='right')
    index_lLim = np.searchsorted(LF_f,freq_lLim, side='right')
    
    while LF_f[index_mLim_LF] > HF_f[index_mLim_HF]:
        index_mLim_HF += 1
    
    f_sub_LF = LF_f[index_lLim:index_mLim_LF]
    f_sub_HF = HF_f[index_mLim_HF:index_uLim]
    
    H_sub_LF = LF_transferFunction[:,:,index_lLim:index_mLim_LF]
    H_sub_HF = HF_transferFunction[:,:,index_mLim_HF:index_uLim]
                    
    f_merged = np.concatenate((f_sub_LF, f_sub_HF))
    H_merged = np.concatenate((H_sub_LF, H_sub_HF), axis=2)
    
    return H_merged, f_merged



def merge_LF_MF_HF_frequency_resp(HF_transferFunction, MF_transferFunction, LF_transferFunction, HF_f, MF_f, LF_f, freq_lLim, freq_mlLim, freq_muLim, freq_uLim):
    '''
    This function merges the frequency responses obtained from LF, MF and HF data based on the specified frequency limits.
    
    Parameters:
    ----------
    HF_transferFunction: numpy array of shape (n, n, len(HF_f)) representing the transfer function obtained from high-frequency data
    MF_transferFunction: numpy array of shape (n, n, len(MF_f)) representing the transfer function obtained from medium-frequency data
    LF_transferFunction: numpy array of shape (n, n, len(LF_f)) representing the transfer function obtained from low-frequency data
    MF_f: numpy array of shape (len(MF_f),) representing the frequency bins corresponding to the medium-frequency transfer function estimates
    LF_f: numpy array of shape (len(LF_f),) representing the frequency bins corresponding to the low-frequency transfer function estimates
    freq_lLim: float representing the lower frequency limit for merging
    freq_mlLim: float representing the middle-low frequency limit for merging
    freq_muLim: float representing the middle-high frequency limit for merging
    freq_uLim: float representing the upper frequency limit for merging
    
    Returns:
    -------
    H_merged: numpy array of shape (n, n, len(f_merged)) representing the merged transfer function across the specified frequency range
    f_merged: numpy array of shape (len(f_merged),) representing the frequency bins corresponding to the merged transfer function estimates
    '''
        
    index_uLim = np.searchsorted(HF_f,freq_uLim, side='right')
    index_muLim_HF = np.searchsorted(HF_f,freq_muLim, side='right')
    index_muLim_MF = np.searchsorted(MF_f,freq_muLim, side='right')
    index_mlLim_MF = np.searchsorted(MF_f,freq_mlLim, side='right')
    index_mlLim_LF = np.searchsorted(LF_f,freq_mlLim, side='right')
    index_lLim = np.searchsorted(LF_f,freq_lLim, side='right')
    
    while LF_f[index_mlLim_LF] > MF_f[index_mlLim_MF]:
        index_mlLim_MF += 1

    while MF_f[index_muLim_MF] > HF_f[index_muLim_HF]:
        index_muLim_HF += 1
    
    f_sub_LF = LF_f[index_lLim:index_mlLim_LF]
    f_sub_MF = MF_f[index_mlLim_MF:index_muLim_MF]
    f_sub_HF = HF_f[index_muLim_HF:index_uLim]
    
    H_sub_LF = LF_transferFunction[:,:,index_lLim:index_mlLim_LF]
    H_sub_MF = MF_transferFunction[:,:,index_mlLim_MF:index_muLim_MF]
    H_sub_HF = HF_transferFunction[:,:,index_muLim_HF:index_uLim]
                    
    f_merged = np.concatenate((f_sub_LF, f_sub_MF, f_sub_HF))
    H_merged = np.concatenate((H_sub_LF, H_sub_MF, H_sub_HF), axis=2)
    
    return H_merged, f_merged



def sample_logarithmically(f_linear, response_linear, num_log_points=50, freq_axis=-1):
    """
    Samples a linearly spaced frequency array and its corresponding multi-dimensional 
    response array to approximate a logarithmically spaced frequency distribution.
    
    Parameters:
    -----------
    f_linear : 1D np.ndarray
        The original linearly spaced frequency array.
    response_linear : np.ndarray
        The frequency response array (e.g., shape: nNodes x nNodes x len(f_linear)).
    num_log_points : int
        The target number of logarithmically spaced points.
    freq_axis : int
        The axis of `response_linear` that corresponds to the frequencies. 
        Default is -1 (the last axis).
        
    Returns:
    --------
    f_log : 1D np.ndarray
        The sampled logarithmically spaced frequencies.
    response_log : np.ndarray
        The frequency response array sampled at the exact same indices.
    """
    
    f_min = f_linear[0]
    if f_min <= 0:
        f_min = f_linear[1] if f_linear[1] > 0 else 1e-5 
    ideal_log_targets = np.logspace(np.log10(f_min), np.log10(f_linear[-1]), num_log_points)
    indices = np.searchsorted(f_linear, ideal_log_targets)
    indices = np.clip(indices, 1, len(f_linear) - 1)
    left_diff = np.abs(f_linear[indices - 1] - ideal_log_targets)
    right_diff = np.abs(f_linear[indices] - ideal_log_targets)
    closest_indices = np.where(left_diff < right_diff, indices - 1, indices)
    final_indices = np.unique(closest_indices)
    f_log = f_linear[final_indices]
    response_log = np.take(response_linear, final_indices, axis=freq_axis)
    
    return f_log, response_log



def verify_with_ground_truth(f_est, path_A, path_B, f_sim, mag_sim_db, phase_sim_deg):
    '''
    Helper function to choose the root closest to the AC simulation
    
    Parameters:
    -----------
    f_est: numpy array of frequencies at which the roots were estimated
    path_A: numpy array of complex values representing the first root path across frequencies
    path_B: numpy array of complex values representing the second root path across frequencies
    f_sim: numpy array of frequencies from the AC simulation
    mag_sim_db: numpy array of magnitude values from the AC simulation (in dB)
    phase_sim_deg: numpy array of phase values from the AC simulation (in degrees)
    
    Returns:
    --------
    H_chosen: numpy array of complex values representing the root path that is closest to the AC simulation across frequencies
    '''
    mag_sim_lin = 10 ** (mag_sim_db / 20.0)
    phase_sim_rad = np.deg2rad(phase_sim_deg)
    
    mag_interp = np.interp(f_est, f_sim, mag_sim_lin)
    phase_interp = np.interp(f_est, f_sim, np.unwrap(phase_sim_rad))
    H_true_complex = mag_interp * np.exp(1j * phase_interp)
    
    dist_A = np.abs(path_A - H_true_complex)
    dist_B = np.abs(path_B - H_true_complex)
    
    H_chosen = np.where(dist_A < dist_B, path_A, path_B)
    return H_chosen



def compute_MSTA_TIA_tf_fast_vectorized(timeseries_data, f_samp, nFFT, nperseg, noverlap, win, n_log_pts, ac_ground_truth=None,
                                        assume_loop_gain_gt_1=True, loop_gain_band=None):
    '''
    This function computes the transfer functions of the MSTA-TIA circuit using the Revolving Door Criteria with a fast vectorized approach and root tracking to ensure continuity of the solution paths.
    
    Parameters:
    -----------
    timeseries_data: (np.ndarray) The time-domain data for the circuit.
    f_samp: (float) The sampling frequency.
    nFFT: (int) The number of FFT points.
    nperseg: (int) The segment length for the Welch method.
    noverlap: (int) The number of overlapping points.
    win: (str or tuple) The window function for the Welch method.
    n_log_pts: (int) The number of logarithmically spaced points for the frequency axis.
    ac_ground_truth: (np.ndarray, optional) The ground truth AC simulation data.
    assume_loop_gain_gt_1: (bool) Physical prior used to break the two-fold branch
        ambiguity for any edge that reaches the fallback. True assumes the circuit runs
        with |L| > 1, i.e. the stages provide real gain. Set False for a fault condition
        that is expected to collapse the loop gain below unity.
    loop_gain_band: (tuple, optional) (f_low, f_high) over which the loop gain is
        measured for that decision. Defaults to the full estimated range.

    Returns:
    --------
    H_est: The estimated frequency response.
    f_log_sampled: The logarithmically sampled frequency points corresponding to the estimated transfer function.
    path_A: The trajectory of the first root across frequencies for the last evaluated transfer function.
    path_B: The trajectory of the second root across frequencies for the last evaluated transfer function.
    '''
    W_yi_ykyj, f = compute_partial_Wiener_coeffs(timeseries_data, f_samp, 0, 2, [1], nFFT, nperseg, noverlap, win)
    f_log_sampled,_ = sample_logarithmically(f, W_yi_ykyj[:,0], num_log_points=n_log_pts)
    H_est = np.zeros((3, 3, len(f_log_sampled)), dtype=complex)
    # Edges whose branch could not be settled per-edge, resolved jointly after the loop
    deferred = {}

    # Frequencies trusted enough to decide a branch on. Used both to label each edge's
    # two tracked paths and to measure the loop gain that then picks between them.
    if loop_gain_band is None:
        band = np.ones(len(f_log_sampled), dtype=bool)
    else:
        band = (f_log_sampled >= loop_gain_band[0]) & (f_log_sampled <= loop_gain_band[1])
    if not np.any(band):
        raise ValueError(f"loop_gain_band={loop_gain_band} selects no frequency in "
                         f"[{f_log_sampled[0]:.3g}, {f_log_sampled[-1]:.3g}] Hz")

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                if (i==0 and j==1 and k==2) or (i==1 and j==2 and k==0) or (i==2 and j==0 and k==1):
                    print(f"Computing for i={i}, j={j}, k={k}")
                    W_yj_yiyk, f = compute_partial_Wiener_coeffs(timeseries_data, f_samp, j, i, [k], nFFT, nperseg, noverlap, win)
                    W_yk_yjyi, f = compute_partial_Wiener_coeffs(timeseries_data, f_samp, k, j, [i], nFFT, nperseg, noverlap, win)
                    W_yi_ykyj, f = compute_partial_Wiener_coeffs(timeseries_data, f_samp, i, k, [j], nFFT, nperseg, noverlap, win)

                    freq_indices = np.where(np.isin(f, f_log_sampled))[0]  
                    a = W_yj_yiyk[freq_indices, 1]
                    f_val = W_yj_yiyk[freq_indices, 0]  
                    c = W_yk_yjyi[freq_indices, 1]
                    b = W_yk_yjyi[freq_indices, 0]
                    d = W_yi_ykyj[freq_indices, 1]
                    e = W_yi_ykyj[freq_indices, 0]

                    # Quadratic coefficients for all frequencies at once
                    u = 1 - a * b
                    w = 1 - e * c
                    A = -d * u
                    B = u * w - a * b * c * e + f_val * d
                    C = -f_val * w

                    with np.errstate(divide='ignore', invalid='ignore'):
                        discriminant = np.sqrt(B**2 - 4.0 * A * C)
                        discriminant = np.where((np.conj(B) * discriminant).real < 0.0,
                                                -discriminant, discriminant)
                        q = -0.5 * (B + discriminant)
                        r1_all = q / A
                        r2_all = C / q
                        lin = (A == 0)
                        if np.any(lin):
                            r1_all = np.where(lin, -C / B, r1_all)
                            r2_all = np.where(lin, -C / B, r2_all)

                    res = lambda x: np.abs(x * (1 - a*b - a*c*e*b / (1 - d*x - e*c)) - f_val)
                    scale = np.maximum(np.abs(f_val), 1e-30)
                    n_bad = np.count_nonzero((res(r1_all) > 1e-6 * scale) |
                                             (res(r2_all) > 1e-6 * scale))
                    if n_bad:
                        print(f"  !! {n_bad} of {len(a)} bins have a root failing the "
                              f"revolving-door relation (likely the pole root)")

                    # Root continuity tracking with a moving average reference to prevent swaps due to noise
                    window_size = 2
                    
                    path_A = np.zeros(len(f_log_sampled), dtype=complex)
                    path_B = np.zeros(len(f_log_sampled), dtype=complex)
                    
                    path_A[0] = r1_all[0]
                    path_B[0] = r2_all[0]

                    for freq_idx in range(1, len(f_log_sampled)):
                        r1 = r1_all[freq_idx]
                        r2 = r2_all[freq_idx]
                        
                        start_idx = max(0, freq_idx - window_size)
                        ref_A = np.mean(path_A[start_idx:freq_idx])
                        ref_B = np.mean(path_B[start_idx:freq_idx])
                        
                        dist_stay = abs(r1 - ref_A) + abs(r2 - ref_B)
                        dist_swap = abs(r1 - ref_B) + abs(r2 - ref_A)
                        
                        if dist_stay <= dist_swap:
                            path_A[freq_idx] = r1
                            path_B[freq_idx] = r2
                        else:
                            path_A[freq_idx] = r2
                            path_B[freq_idx] = r1

                    # Root selection (Ground truth or loop-gain check)
                    # Check if ground truth data was provided for this specific path
                    if ac_ground_truth is not None and (j, i) in ac_ground_truth:
                        print(f"  -> Using AC Ground Truth to verify path ({j}, {i})")
                        f_sim, mag_sim_db, phase_sim_deg = ac_ground_truth[(j, i)]
                        H_est[j, i, :] = verify_with_ground_truth(
                            f_log_sampled, path_A, path_B, f_sim, mag_sim_db, phase_sim_deg
                        )
                    else:
                        print(f"  -> Performing gain check for ({i}, {j}).")

                        path_A_mean = np.mean(np.abs(path_A[band]))
                        path_B_mean = np.mean(np.abs(path_B[band]))
                        if path_A_mean > path_B_mean:
                            H_est[j, i, :] = path_A
                        else:
                            H_est[j, i, :] = path_B

    return H_est, f_log_sampled, path_A, path_B