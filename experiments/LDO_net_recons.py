'''This script implements the experiment related to network reconstruction for three STIA amplifier'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.utils import normalize_data, data_FFT, save_pag_recons_metrics
from src.cyclic_network_reconstruction import algCCD, find_tau_CCD_avg_deg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import matplotlib.ticker as ticker



def evaluate_pag(psi_adj_true, edge_points_true, psi_adj_pred, edge_points_pred):
    num_nodes = psi_adj_true.shape[0]
    
    y_true_binary = []
    y_pred_binary = []
    
    y_true_multi = []
    y_pred_multi = []
    

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            
            true_adj = psi_adj_true[i, j]
            pred_adj = psi_adj_pred[i, j]
            
            y_true_binary.append(true_adj)
            y_pred_binary.append(pred_adj)
            
            # Default to 'no_edge' if there is no adjacency
            true_edge = edge_points_true.get((i, j), 'no_edge') if true_adj else 'no_edge'
            pred_edge = edge_points_pred.get((i, j), 'no_edge') if pred_adj else 'no_edge'
            
            y_true_multi.append(true_edge)
            y_pred_multi.append(pred_edge)

    return y_true_binary, y_pred_binary, y_true_multi, y_pred_multi

def main():
    #######################################################################################################################
    # Load time series data
    time_series_path = "data/time_series_data/LDO"
    
    # Load all data
    time_series_data = normalize_data(pd.read_csv(f'{time_series_path}/V_100Hz10MHz_fsamp_50ns.csv').iloc[:, 1::2].values)
    nNodes = time_series_data.shape[1]
    nFFT = 64
    # threshd = 0.019
    tauSearchLow = 0.01
    tauSearchHigh = 0.025
    tauSearchRes = 0.001
    avgDegLow = 3.2
    avgDegHigh = 3.6
    min_deg_lim = 1
    max_deg_lim = 4
    freq_choice = range(nFFT)
    nSamplesList = [25600, 51200, 100000, 150000, 200000]
    
    psi_adj = np.array([[0, 0, 1, 1, 1],
                        [0, 0, 1, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 0]])

    true_edge_end_points = {(0,2): '-->', (0,3): '-->', (0,4): '-->', (1,2): '-->', (1,3): '-->', (1,4): '-->', (2,3): 'o-o', (2,4): 'o-o', (3,4): 'o-o'}

    true_underlines = {}
    TPR_skel_arr = []
    FPR_skel_arr = []
    precision_skel_arr = []
    recall_skel_arr = []
    f1_skel_arr = []

    # One-vs-all edge-mark classes: 'no_edge' plus every mark type present in the true PAG
    edge_labels = ['no_edge'] + sorted(set(true_edge_end_points.values()))
    TPR_edgpt_dict = {label: [] for label in edge_labels}
    FPR_edgpt_dict = {label: [] for label in edge_labels}
    precision_edgpt_dict = {label: [] for label in edge_labels}
    recall_edgpt_dict = {label: [] for label in edge_labels}
    f1_edgpt_dict = {label: [] for label in edge_labels}

    for nSamples in nSamplesList:
        nperseg = (time_series_data[:nSamples,:].shape[0])// nFFT
        fft_data = data_FFT(time_series_data[:nSamples,:],nperseg,nFFT)
        threshd = find_tau_CCD_avg_deg(fft_data,freq_choice,tauSearchLow,tauSearchHigh,tauSearchRes,avgDegLow,avgDegHigh,min_deg_lim,max_deg_lim,nSamples)
        print(f"Tuned threshold for nSamples={nSamples}: {threshd}")
        psi,EdgeEndPoints,Underlines = algCCD(nNodes,fft_data,freq_choice,threshd)
        y_true_bin, y_pred_bin, y_true_mul, y_pred_mul = evaluate_pag(psi_adj, true_edge_end_points, psi, EdgeEndPoints)
        print(f"\n\nResults for nSamples={nSamples}:")
        print("Discovered PAG using CCD algorithm:")
        print("psi=")
        print(psi)
        print("\n Edge End Points=")
        print(EdgeEndPoints)
        print("\n Underlines=")
        print(Underlines)
        print("\nSkeleton classification output:")
        print(f"y true bin = {y_true_bin}")
        print(f"y pred bin = {y_pred_bin}")
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        TPR_skel_arr.append(tp/(tp+fn) if (tp + fn) != 0 else 0)
        FPR_skel_arr.append(fp/(fp+tn) if (fp + tn) != 0 else 0)
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1        = (2 * precision * recall) /(precision + recall) if (precision + recall) != 0 else 0
        precision_skel_arr.append(precision)
        recall_skel_arr.append(recall)
        f1_skel_arr.append(f1)
        print(classification_report(y_true_bin, y_pred_bin))

        print("\nClassification Report for EdgeClassification:")
        print(f"y true mul = {y_true_mul}")
        print(f"y pred mul = {y_pred_mul}")
        matrix = confusion_matrix(y_true_mul, y_pred_mul, labels=edge_labels)
        print("Confusion Matrix for Multi-class Classification:")
        print(matrix)

        # One-vs-all TPR/FPR/precision/recall/F1 for every edge-mark class
        mcm = multilabel_confusion_matrix(y_true_mul, y_pred_mul, labels=edge_labels)
        for label, (tn, fp, fn, tp) in zip(edge_labels, mcm.reshape(len(edge_labels), 4)):
            tpr       = tp / (tp + fn) if (tp + fn) != 0 else 0
            fpr       = fp / (fp + tn) if (fp + tn) != 0 else 0
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall    = tpr
            f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            TPR_edgpt_dict[label].append(tpr)
            FPR_edgpt_dict[label].append(fpr)
            precision_edgpt_dict[label].append(precision)
            recall_edgpt_dict[label].append(recall)
            f1_edgpt_dict[label].append(f1)

        print(classification_report(y_true_mul, y_pred_mul, labels=edge_labels))
        print("--------------------------------------")

    skeleton_metrics = {
        'f1': f1_skel_arr, 'tpr': TPR_skel_arr, 'fpr': FPR_skel_arr,
        'precision': precision_skel_arr, 'recall': recall_skel_arr,
    }
    edge_metrics = {
        label: {
            'f1': f1_edgpt_dict[label], 'tpr': TPR_edgpt_dict[label], 'fpr': FPR_edgpt_dict[label],
            'precision': precision_edgpt_dict[label], 'recall': recall_edgpt_dict[label],
        }
        for label in edge_labels
    }
    save_pag_recons_metrics(
        'data/experimental_results/algo_outputs/LDO_metrics.json',
        nSamplesList, skeleton_metrics, edge_metrics,
    )

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8
    plt.figure(figsize=(1.7, 1.3))
    plt.semilogx(nSamplesList, f1_skel_arr, '-o', label="F-1")
    # plt.semilogx(nSamplesList, precision_arr, '-s', label="Precision")
    # plt.semilogx(nSamplesList, recall_arr, '-^', label="Recall")
    plt.semilogx(nSamplesList, TPR_skel_arr, '-d', label="TPR")
    plt.semilogx(nSamplesList, FPR_skel_arr, '-*', label="FPR")
    # plt.title('Sample Size vs. Performance Metrics')
    plt.ylabel('Performance Metrics', fontsize = FONTSIZE)
    plt.xlabel('Sample Size', fontsize = FONTSIZE)
    plt.legend(fontsize = FONTSIZE)
    # plt.ylim([-35,30])
    plt.xticks([25600, 51200, 102400, 200000], ['25.6K', '51.2K', '102.4K', '200K'])
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
    plt.savefig('data/experimental_results/plots/reconstruction_results/LDO_skel_recons_error.pdf', dpi=300, bbox_inches='tight')
    plt.grid(True)
    plt.show()


    edge_label_tags = {
        'no_edge': 'no_edge',
        'o-o': 'circle_circle',
        '-->': 'arrow_fwd',
        '<--': 'arrow_bwd',
        '---': 'tail_tail',
        'o--': 'circle_tail',
        '--o': 'tail_circle',
    }
    for label in edge_labels:
        label_tag = edge_label_tags.get(label, ''.join(c if c.isalnum() else '_' for c in label))

        plt.figure(figsize=(1.7, 1.3))
        plt.semilogx(nSamplesList, f1_edgpt_dict[label], '-o', label="F-1")
        # plt.semilogx(nSamplesList, precision_edgpt_dict[label], '-s', label="Precision")
        # plt.semilogx(nSamplesList, recall_edgpt_dict[label], '-^', label="Recall")
        plt.semilogx(nSamplesList, TPR_edgpt_dict[label], '-d', label="TPR")
        plt.semilogx(nSamplesList, FPR_edgpt_dict[label], '-*', label="FPR")
        # plt.title(f'Sample Size vs. Performance Metrics ({label})')
        plt.ylabel('Performance Metrics', fontsize = FONTSIZE)
        plt.xlabel('Sample Size', fontsize = FONTSIZE)
        plt.legend(fontsize = FONTSIZE)
        # plt.ylim([-35,30])
        # plt.xticks([200000, 500000, time_series_data.shape[0]], ['200K', '800K', '1M'])
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        plt.gca().xaxis.set_minor_locator(ticker.NullLocator())
        plt.savefig(f'data/experimental_results/plots/reconstruction_results/LDO_edgpt_recons_error_{label_tag}.pdf', dpi=300, bbox_inches='tight')
        plt.grid(True)
        plt.show()

# Entry point of the script
if __name__ == "__main__":
    main()  