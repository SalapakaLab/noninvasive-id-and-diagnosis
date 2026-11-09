'''This script examines the sensitivity of network reconstruction for the Differential EMG Sensor
(Biopotential Sensor) to the choice of W-PC threshold. F1 score is swept over a fine grid of
thresholds at several sample sizes, showing how the plateau of near-optimal thresholds widens as
more data becomes available, and marking where the sparsity-based auto-selection rule (the same
average/min/max-degree criterion used by tune_WPC_threshold) lands on each curve.'''

import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.network_reconstruction import Run_PC_Wiener_test, estimate_essential_graph, compute_undir_graph_degrees
from src.utils import (get_confusion_matrix, data_FFT, find_moral_graph_and_skeleton_from_adjacency_matrix,
                       find_true_Essential_graph, save_threshold_sensitivity_data, load_threshold_sensitivity_data)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

DATA_PATH = 'data/experimental_results/algo_outputs/bio_pot_sensor_threshold_sensitivity.json'


def sweep_threshold_f1(fft_data, thresh_vec, freq_choice, q, trueEssGraph, avg_deg_lim, max_deg_lim, min_deg_lim):
    '''
    Sweeps a grid of W-PC thresholds, reconstructing the essential graph at each and scoring F1
    against ground truth. Also locates the sparsity-based auto-selected threshold: the first
    (lowest) threshold in the grid whose estimated skeleton has average/min/max degree within the
    given bounds -- this mirrors the selection rule used by tune_WPC_threshold, computed here
    directly from the already-reconstructed skeletons instead of re-running the search.

    Parameters:
    ----------
    fft_data: nSamples X nNodes X nFFT frequency-domain data for the sample size of interest
    thresh_vec: 1D array of threshold values to sweep, in increasing order
    freq_choice: list of frequency indices to consider for the projection coefficient
    q: maximum size of the separation set to consider
    trueEssGraph: nNodes X nNodes ground-truth essential graph adjacency matrix
    avg_deg_lim, max_deg_lim, min_deg_lim: degree constraints defining the auto-selection rule

    Returns:
    -------
    f1_arr: array of F1 scores, one per entry in thresh_vec
    auto_thresh: threshold value picked by the sparsity rule (None if no threshold in the grid qualifies)
    '''

    nNodes = fft_data.shape[1]
    f1_arr = np.zeros(len(thresh_vec))
    auto_thresh = None
    for idx, t in enumerate(thresh_vec):
        C = Run_PC_Wiener_test(fft_data, t, freq_choice, q)
        Top_est_half = np.zeros([nNodes, nNodes])
        for c in C:
            if C[c] is None:
                Top_est_half[c] = 1
        Top_est = Top_est_half + Top_est_half.T
        est_essential_graph = estimate_essential_graph(Top_est, C)

        TP, FP, TN, FN = get_confusion_matrix(trueEssGraph, est_essential_graph)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_arr[idx] = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        if auto_thresh is None:
            min_deg, max_deg, avg_deg = compute_undir_graph_degrees(Top_est)
            if avg_deg_lim[0] <= avg_deg <= avg_deg_lim[1] and max_deg <= max_deg_lim and min_deg >= min_deg_lim:
                auto_thresh = t

    return f1_arr, auto_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                         help=f'Skip the reconstruction sweep and replot from previously saved data at {DATA_PATH}')
    args = parser.parse_args()

    if args.load:
        thresh_vec, nSamplesList, f1_results, auto_thresh_results = load_threshold_sensitivity_data(DATA_PATH)
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
        nSamplesList = [640000, 800000, 1000000]
        Adj = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [1, 0, 1, 1, 0]])
        _, Skeleton = find_moral_graph_and_skeleton_from_adjacency_matrix(Adj)
        trueEssGraph = find_true_Essential_graph(Skeleton, Adj)
        avg_deg_lim = [1.4, 2]
        max_deg_lim = 3
        min_deg_lim = 1
        thresh_vec = np.arange(0.01, 1 + 1e-9, 0.01)

        #######################################################################################################################
        # Sweep threshold at each sample size
        f1_results = {}
        auto_thresh_results = {}
        for nSamples in nSamplesList:
            nperseg = time_series_data[:nSamples, :].shape[0] // nFFT
            fft_data = data_FFT(time_series_data[:nSamples, :], nperseg, nFFT)
            f1_arr, auto_thresh = sweep_threshold_f1(fft_data, thresh_vec, freq_choice, q, trueEssGraph,
                                                      avg_deg_lim, max_deg_lim, min_deg_lim)
            f1_results[nSamples] = f1_arr
            auto_thresh_results[nSamples] = auto_thresh

            print("----------------------------------------______")
            print('Sample Size =', nSamples)
            print('Auto-selected threshold =', auto_thresh)

        save_threshold_sensitivity_data(DATA_PATH, thresh_vec, nSamplesList, f1_results, auto_thresh_results)

    #######################################################################################################################
    # Plot results
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    # Okabe-Ito colorblind-safe palette (blue, orange, bluish green); vermillion/red is
    # deliberately reserved for the auto-selected marker so it stands out from the curves.
    okabe_ito_colors = ['#0072B2', '#E69F00', '#009E73']
    line_styles = ['-', '--', '-.']
    marker_styles = ['o', 's', '^']
    marker_step = max(1, len(thresh_vec) // 5)
    # Stagger each curve's marker phase so markers from different curves land at different
    # thresholds instead of stacking on top of each other where the curves nearly coincide.
    marker_offsets = [round(idx * marker_step / len(nSamplesList)) for idx in range(len(nSamplesList))]

    plt.figure(figsize=(3.5, 2.5))
    for idx, nSamples in enumerate(nSamplesList):
        color = okabe_ito_colors[idx % len(okabe_ito_colors)]
        plt.plot(thresh_vec, f1_results[nSamples],
                 linestyle=line_styles[idx % len(line_styles)],
                 marker=marker_styles[idx % len(marker_styles)],
                 markevery=(marker_offsets[idx], marker_step), markersize=4, color=color,
                 label=f'N={nSamples}')
        auto_thresh = auto_thresh_results[nSamples]
        if auto_thresh is not None:
            auto_idx = np.argmin(np.abs(thresh_vec - auto_thresh))
            plt.plot(auto_thresh, f1_results[nSamples][auto_idx], marker='*', markersize=9,
                     color='red', markeredgecolor='black', markeredgewidth=0.5,
                     linestyle='None', zorder=5)

    auto_marker_proxy = mlines.Line2D([], [], marker='*', markersize=9, color='none',
                                       markerfacecolor='red', markeredgecolor='black',
                                       markeredgewidth=0.5, linestyle='None', label='Auto-selected')
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles + [auto_marker_proxy], fontsize=FONTSIZE, loc='lower right')

    plt.xlabel('Threshold', fontsize=FONTSIZE)
    plt.ylabel('F-1 Score', fontsize=FONTSIZE)
    plt.ylim(0.3, 1.05)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.grid(True)

    save_dir = 'data/experimental_results/plots/Threshold_Sensitivity'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/bio_pot_sensor_threshold_sensitivity.pdf', dpi=300, bbox_inches='tight')
    # plt.show()


# Entry point of the script
if __name__ == "__main__":
    main()
