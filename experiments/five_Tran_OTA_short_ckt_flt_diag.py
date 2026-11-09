'''This script implements the experiment related to network reconstruction for three STIA amplifier when there is a 
short circuit fault between node 1 and stage 2'''

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
    #######################################################################################################################
    # Load time series data
    time_series_path = "data/time_series_data/five_tran_ota/faulty"
    
    # Load all data
    time_series_data = normalize_data(pd.read_csv(f'{time_series_path}/V_100Hz100MHz_short_ckt_fault.csv').iloc[:, [1, 3, 7]].values)
    nNodes = time_series_data.shape[1]
    f_samp = 1/5e-9
    freq_uLim = 1e7
    freq_lLim = 1e3
    nFFT = 4096
    q = nNodes - 2
    threshd = 1e-6
    freq_choice = range(nFFT)
    nSamplesList = [time_series_data.shape[0]]
    normalAdj = np.array([[0, 1, 0],
                        [1, 0, 0],
                        [1, 1, 0]])
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