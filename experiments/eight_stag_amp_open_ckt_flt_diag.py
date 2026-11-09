'''This script implements the experiment related to network reconstruction for eight stage amplifier when there is a 
open circuit fault between stage 7 and stage 8'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.network_reconstruction import Run_PC_Wiener_test, estimate_essential_graph
from src.utils import normalize_data, get_confusion_matrix, data_FFT, find_moral_graph_and_skeleton_from_adjacency_matrix, find_true_Essential_graph
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

def main():
    f_min = 1e10
    f_max = 2e11

    #######################################################################################################################
    # Load time series data
    time_series_path = time_series_path = "data/time_series_data/eight_stagger_tuned_RF_amp/Faulty"
    list_of_nodes = ['1','2','3','4','5','6','7','8']
    
    # Load all data
    time_series_data = (np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_20M10T_open_ckt_flt.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ]))
    means = np.mean(time_series_data, axis=0)
    max_abs = np.max(np.abs(time_series_data), axis=0)
    # Avoid division by zero
    max_abs = np.where(max_abs == 0, 1, max_abs)
    time_series_data = (time_series_data - means) / max_abs
    print("Shape of the time series data: ", time_series_data.shape)
    nNodes = time_series_data.shape[1]
    nFFT = 4096
    f_samp = 1/50e-15
    frequencies = np.fft.rfftfreq(nFFT, d=1/f_samp)
    q = 3
    threshd = 0.15
    freq_choice = np.where((frequencies >= f_min) & (frequencies <= f_max))[0]
    nSamplesList = [time_series_data.shape[0]]
    normalAdj = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0]])
    _,Skeleton = find_moral_graph_and_skeleton_from_adjacency_matrix(normalAdj)
    normalEssGraph = find_true_Essential_graph(Skeleton, normalAdj)

    nSamples = nSamplesList[0]
    nperseg = (time_series_data[:nSamples,:].shape[0])// nFFT
    fft_data = data_FFT(time_series_data[:nSamples,:],nperseg,nFFT)
    C = Run_PC_Wiener_test(fft_data,threshd,freq_choice,q)
    Top_est_half = np.zeros([nNodes,nNodes])
    for c in C:
        if C[c] == None: Top_est_half[c]=1
    Top_est = (Top_est_half + Top_est_half.T)
    est_essential_graph = estimate_essential_graph(Top_est,C)
    print("Estimated Essential Graph: \n", est_essential_graph)

    EssGraphDiff = est_essential_graph - normalEssGraph
    print("Difference between estimated and true essential graph: \n", EssGraphDiff)

    for i in range(nNodes):
        for j in range(nNodes):
            if EssGraphDiff[i,j] != 0:
                print(f"Difference at position ({i}, {j}): Estimated = {est_essential_graph[i,j]}, True = {normalEssGraph[i,j]}")



# Entry point of the script
if __name__ == "__main__":
    main()  