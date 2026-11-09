'''This script implements the experiment related to network reconstruction for three STIA amplifier'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.network_reconstruction import Run_PC_Wiener_test, estimate_essential_graph, tune_WPC_threshold
from src.utils import normalize_data, get_confusion_matrix, data_FFT, find_moral_graph_and_skeleton_from_adjacency_matrix, find_true_Essential_graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    #######################################################################################################################
    # Load time series data
    time_series_path = time_series_path = "data/time_series_data/eight_stagger_tuned_RF_amp/Healthy"
    
    # Load all data
    time_series_data = pd.read_csv(f'{time_series_path}/V_20M10T.csv').iloc[:, 1::2].values
    means = np.mean(time_series_data, axis=0)
    max_abs = np.max(np.abs(time_series_data), axis=0)
    # Avoid division by zero
    max_abs = np.where(max_abs == 0, 1, max_abs)
    time_series_data = (time_series_data - means) / max_abs
    print("Shape of the time series data: ", time_series_data.shape)
    nNodes = time_series_data.shape[1]
    nFFT = 4096
    q = 3
    # threshd = 0.06
    thresh_range = [0.01, 0.1, 0.01]
    avg_deg_lim = [1.3, 2]
    max_deg_lim = 3
    min_deg_lim = 1
    freq_choice = range(nFFT)
    # nSamplesList = [200000, 500000, 800000, time_series_data.shape[0]]
    nSamplesList = [25600, 102400, 320000, 500000, time_series_data.shape[0]]
    Adj = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0]])
    _,Skeleton = find_moral_graph_and_skeleton_from_adjacency_matrix(Adj)
    trueEssGraph = find_true_Essential_graph(Skeleton, Adj)

    TPR_arr = []
    FPR_arr = []
    precision_arr = []
    recall_arr = []
    f1_arr = []
    for nSamples in nSamplesList:
        nperseg = (time_series_data[:nSamples,:].shape[0])// nFFT
        fft_data = data_FFT(time_series_data[:nSamples,:],nperseg,nFFT)
        threshd = tune_WPC_threshold(time_series_data,[nFFT],nperseg,thresh_range,freq_choice,avg_deg_lim,max_deg_lim,min_deg_lim,nSamples,q)
        print(f"Tuned threshold for nSamples={nSamples}: {threshd}")
        C = Run_PC_Wiener_test(fft_data,threshd,freq_choice,q)
        print("Estimated Conditional Independence Sets: ", C)
        Top_est_half = np.zeros([nNodes,nNodes])
        for c in C:
            if C[c] == None: Top_est_half[c]=1
        Top_est = (Top_est_half + Top_est_half.T)
        est_essential_graph = estimate_essential_graph(Top_est,C)

        TP, FP, TN, FN = get_confusion_matrix(trueEssGraph, est_essential_graph)
        TPR_arr.append(TP/(TP+FN) if (TP + FN) != 0 else 0)
        FPR_arr.append(FP/(TN+FP) if (TN + FP) != 0 else 0)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1        = (2 * precision * recall) /(precision + recall) if (precision + recall) != 0 else 0
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)

        print("----------------------------------------______")
        print('Sample Size=', nSamples)
        print(f"TPR = {TPR_arr[-1]}, FPR = {FPR_arr[-1]}")
        print('F1 Score = ',f1)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    plt.figure(figsize=(1.7, 1.3))
    plt.semilogx(nSamplesList, f1_arr, '-o', label="F-1")
    # plt.semilogx(nSamplesList, precision_arr, '-s', label="Precision")
    # plt.semilogx(nSamplesList, recall_arr, '-^', label="Recall")
    plt.semilogx(nSamplesList, TPR_arr, '-d', label="TPR")
    plt.semilogx(nSamplesList, FPR_arr, '-*', label="FPR")
    # plt.title('Sample Size vs. Performance Metrics')
    plt.ylabel('Performance Metrics', fontsize = FONTSIZE)
    plt.xlabel('Sample Size', fontsize = FONTSIZE)
    plt.legend(fontsize = FONTSIZE)
    # plt.ylim([-35,30])
    plt.xticks([25600, 102400, 500000], ['25.6K', '102.4K', '500K'])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig('data/experimental_results/plots/reconstruction_results/eight_stag_amp_recons_error.pdf', dpi=300, bbox_inches='tight')
    plt.grid(True)
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()  