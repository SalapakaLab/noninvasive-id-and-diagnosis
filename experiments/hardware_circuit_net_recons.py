'''This script implements the experiment related to network reconstruction for the hardware circuit'''

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

def main():
    #######################################################################################################################
    # Load time series data
    time_series_path = "data/time_series_data/hardware_circuit/healthy"

    # Load all data
    time_series_data = pd.read_csv(f'{time_series_path}/Healthy_Hardware_Data_1MS_s.csv').iloc[:, 1::2].values
    nNodes = time_series_data.shape[1]
    nFFT = 64
    q = nNodes - 2
    # threshd = 0.1
    freq_choice = range(nFFT)
    nSamplesList = [400, 800, 1600, 3200, 6400, 12800]
    Adj = np.array([[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 1, 0]])
    _,Skeleton = find_moral_graph_and_skeleton_from_adjacency_matrix(Adj)
    trueEssGraph = find_true_Essential_graph(Skeleton, Adj)
    avg_deg_lim = [1.1, 1.9]
    thresh_range = [0.01, 0.2, 0.01]
    max_deg_lim = 3
    min_deg_lim = 1

    TPR_arr = []
    FPR_arr = []
    precision_arr = []
    recall_arr = []
    f1_arr = []
    for nSamples in nSamplesList:
        nperseg = (time_series_data[:nSamples,:].shape[0])// nFFT
        fft_data = data_FFT(time_series_data[:nSamples,:],nperseg,nFFT)
        threshd = tune_WPC_threshold(time_series_data,[nFFT],nperseg,thresh_range,freq_choice,avg_deg_lim,max_deg_lim,min_deg_lim,nSamples,q)
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
    # plt.semilogx(nSamplesList, recall_arr, '-^', label="Recall",)
    plt.semilogx(nSamplesList, TPR_arr, '-d', label="TPR")
    plt.semilogx(nSamplesList, FPR_arr, '-*', label="FPR")
    # plt.title('Sample Size vs. Performance Metrics', fontsize = FONTSIZE)
    plt.ylabel('Performance Metrics', fontsize = FONTSIZE)
    plt.xlabel('Sample Size', fontsize = FONTSIZE)
    plt.legend(loc='center right', fontsize = FONTSIZE)
    # plt.xlim([10000,600000])
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.savefig('data/experimental_results/plots/reconstruction_results/hardware_circuit_recons_error.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()  