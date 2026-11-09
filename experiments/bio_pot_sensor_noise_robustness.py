'''This script evaluates robustness of network reconstruction for the Differential EMG Sensor
(Biopotential Sensor) to correlated measurement noise. Correlated Gaussian noise is added to the
Cadence simulated time series at a range of SNRs (fixed noise correlation) and, separately, at a
range of noise correlation coefficients (fixed SNR). At each noise setting the W-PC threshold is
re-tuned automatically and the network is reconstructed, and reconstruction error metrics are
plotted against noise strength.'''

import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.network_reconstruction import Run_PC_Wiener_test, estimate_essential_graph, tune_WPC_threshold
from src.utils import (add_correlated_noise, get_confusion_matrix, data_FFT,
                       find_moral_graph_and_skeleton_from_adjacency_matrix, find_true_Essential_graph,
                       save_noise_robustness_data, load_noise_robustness_data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_PATH = 'data/experimental_results/algo_outputs/bio_pot_sensor_noise_robustness.json'
# Okabe-Ito colorblind-safe palette (blue, orange, bluish green, vermillion, sky blue)
OKABE_ITO_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9']


def reconstruct_and_score(noisy_data, nSamples, nFFT, q, thresh_range, freq_choice,
                           avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph):
    '''
    Runs the auto-thresholded W-PC reconstruction on a (possibly noisy) time series
    and scores the resulting essential graph against the ground truth.

    Parameters:
    ----------
    noisy_data: nSamples_total X nNodes array of time-series data (clean or noise-corrupted)
    nSamples: number of FFT blocks (of length nFFT) to use for reconstruction/threshold-tuning
    nFFT: number of FFT points per block
    q: maximum size of the separation set to consider
    thresh_range: [start, stop, step] range of thresholds to search when auto-tuning
    freq_choice: list of frequency indices to consider for the projection coefficient
    avg_deg_lim, max_deg_lim, min_deg_lim: degree constraints used for threshold tuning
    trueEssGraph: nNodes X nNodes ground-truth essential graph adjacency matrix

    Returns:
    -------
    metrics: dict with keys 'tpr', 'fpr', 'precision', 'recall', 'f1'
    '''

    nperseg = noisy_data[:nSamples, :].shape[0] // nFFT
    fft_data = data_FFT(noisy_data[:nSamples, :], nperseg, nFFT)
    threshd = tune_WPC_threshold(noisy_data, [nFFT], nperseg, thresh_range, freq_choice,
                                  avg_deg_lim, max_deg_lim, min_deg_lim, nSamples, q)
    C = Run_PC_Wiener_test(fft_data, threshd, freq_choice, q)

    nNodes = noisy_data.shape[1]
    Top_est_half = np.zeros([nNodes, nNodes])
    for c in C:
        if C[c] is None:
            Top_est_half[c] = 1
    Top_est = Top_est_half + Top_est_half.T
    est_essential_graph = estimate_essential_graph(Top_est, C)

    TP, FP, TN, FN = get_confusion_matrix(trueEssGraph, est_essential_graph)
    tpr = TP / (TP + FN) if (TP + FN) != 0 else 0
    fpr = FP / (TN + FP) if (TN + FP) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {'tpr': tpr, 'fpr': fpr, 'precision': precision, 'recall': recall, 'f1': f1}


def run_sweep(time_series_data, noise_param_list, make_noisy_data, nSamples, nFFT, q,
              thresh_range, freq_choice, avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph, seed):
    '''
    Sweeps a noise parameter (SNR or correlation coefficient), reconstructing and scoring at each
    setting with a freshly-seeded noise draw for reproducibility.

    Parameters:
    ----------
    time_series_data: nSamples_total X nNodes clean time-series data
    noise_param_list: list of values for the swept noise parameter
    make_noisy_data: function(clean_data, param, rng) -> noisy_data
    nSamples, nFFT, q, thresh_range, freq_choice, avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph:
        passed through to reconstruct_and_score
    seed: base RNG seed; each sweep point gets its own default_rng(seed) so runs are reproducible

    Returns:
    -------
    metrics_arr: dict mapping metric name ('tpr','fpr','precision','recall','f1') to a list of
        values, one per entry in noise_param_list
    '''

    metrics_arr = {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1': []}
    for param in noise_param_list:
        rng = np.random.default_rng(seed)
        noisy_data = make_noisy_data(time_series_data, param, rng)
        metrics = reconstruct_and_score(noisy_data, nSamples, nFFT, q, thresh_range, freq_choice,
                                         avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph)
        for key in metrics_arr:
            metrics_arr[key].append(metrics[key])
        print(f"param={param}: F1={metrics['f1']:.3f}, TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")

    return metrics_arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                         help=f'Skip the reconstruction sweeps and replot from previously saved data at {DATA_PATH}')
    args = parser.parse_args()

    if args.load:
        (SNR_dB_list, snr_metrics, rho_list,
         SNR_dB_list_for_rho_sweep, rho_metrics_by_snr) = load_noise_robustness_data(DATA_PATH)
    else:
        #######################################################################################################################
        # Load time series data
        time_series_path = "data/time_series_data/bio_pot_sensor/healthy"
        list_of_nodes = ['a', 'b', 'c', 'd', 'o']

        time_series_data = np.column_stack([
            pd.read_csv(f'{time_series_path}/V{node}_10mHz5kHz.csv').iloc[:, 1:].values.flatten()
            for node in list_of_nodes
        ])
        nNodes = time_series_data.shape[1]
        nFFT = 64
        q = nNodes - 2
        freq_choice = range(nFFT)
        nSamples = 1000000
        Adj = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0]])
        _, Skeleton = find_moral_graph_and_skeleton_from_adjacency_matrix(Adj)
        trueEssGraph = find_true_Essential_graph(Skeleton, Adj)
        avg_deg_lim = [1.4, 2]
        thresh_range = [0.01, 0.2, 0.01]
        max_deg_lim = 3
        min_deg_lim = 1
        seed = 42

        #######################################################################################################################
        # Sweep 1: SNR, at fixed (zero) noise correlation
        SNR_dB_list = [30, 25, 20, 15, 5, 0, -5]
        rho_fixed = 0.0

        print("----------------------------------------______")
        print("Sweeping SNR (correlation coefficient fixed at", rho_fixed, ")")
        snr_metrics = run_sweep(
            time_series_data, SNR_dB_list,
            lambda data, snr_db, rng: add_correlated_noise(data, snr_db, rho_fixed, rng),
            nSamples, nFFT, q, thresh_range, freq_choice, avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph, seed)

        #######################################################################################################################
        # Sweep 2: noise correlation coefficient, at a few fixed SNR levels
        rho_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
        SNR_dB_list_for_rho_sweep = [30, 25, 15]

        rho_metrics_by_snr = {}
        for SNR_dB_fixed in SNR_dB_list_for_rho_sweep:
            print("----------------------------------------______")
            print("Sweeping correlation coefficient (SNR fixed at", SNR_dB_fixed, "dB)")
            rho_metrics_by_snr[SNR_dB_fixed] = run_sweep(
                time_series_data, rho_list,
                lambda data, rho, rng, snr_db=SNR_dB_fixed: add_correlated_noise(data, snr_db, rho, rng),
                nSamples, nFFT, q, thresh_range, freq_choice, avg_deg_lim, max_deg_lim, min_deg_lim, trueEssGraph, seed)

        save_noise_robustness_data(DATA_PATH, SNR_dB_list, snr_metrics, rho_list,
                                    SNR_dB_list_for_rho_sweep, rho_metrics_by_snr)

    #######################################################################################################################
    # Plot results
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    save_dir = 'data/experimental_results/plots/Noise_Robustness'
    os.makedirs(save_dir, exist_ok=True)

    # Metric -> (linestyle, marker); color separately encodes SNR in the rho-sweep plot below.
    metric_styles = {'f1': ('-', 'o'), 'tpr': ('--', 'd'), 'fpr': (':', '*')}
    metric_labels = {'f1': 'F-1', 'tpr': 'TPR', 'fpr': 'FPR'}

    plt.figure(figsize=(3.5, 2.5))  # IEEE single-column width
    for idx, key in enumerate(['f1', 'tpr', 'fpr']):
        linestyle, marker = metric_styles[key]
        plt.plot(SNR_dB_list, snr_metrics[key], linestyle=linestyle, marker=marker,
                 color=OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)], markersize=4,
                 label=metric_labels[key])
    plt.gca().invert_xaxis()
    plt.xlabel('SNR (dB)', fontsize=FONTSIZE)
    plt.ylabel('Performance Metrics', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)
    plt.savefig(f'{save_dir}/bio_pot_sensor_noise_robustness_vs_SNR.pdf', dpi=300, bbox_inches='tight')

    _, ax = plt.subplots(figsize=(1.7, 1.3))
    color_handles = []
    for idx, SNR_dB_fixed in enumerate(SNR_dB_list_for_rho_sweep):
        color = OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)]
        metrics = rho_metrics_by_snr[SNR_dB_fixed]
        for key, (linestyle, marker) in metric_styles.items():
            ax.plot(rho_list, metrics[key], linestyle=linestyle, marker=marker,
                    color=color, markersize=4)
        color_handles.append(mlines.Line2D([], [], color=color, label=f'SNR={SNR_dB_fixed} dB'))

    style_handles = [
        mlines.Line2D([], [], color='black', linestyle=linestyle, marker=marker, label=metric_labels[key])
        for key, (linestyle, marker) in metric_styles.items()
    ]
    metric_header = mlines.Line2D([], [], linestyle='None', marker='None', label='Metric:')
    snr_header = mlines.Line2D([], [], linestyle='None', marker='None', label='SNR:')
    ax.legend(handles=[metric_header] + style_handles + [snr_header] + color_handles,
              fontsize=FONTSIZE - 1, loc='center left', bbox_to_anchor=(1.02, 0.5),
              handlelength=1.5, labelspacing=0.3, frameon=False)

    ax.set_xlabel('Noise correlation coefficient', fontsize=FONTSIZE)
    ax.set_ylabel('Performance Metrics', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True)
    plt.savefig(f'{save_dir}/bio_pot_sensor_noise_robustness_vs_rho.pdf', dpi=300, bbox_inches='tight')
    # plt.show()


# Entry point of the script
if __name__ == "__main__":
    main()
