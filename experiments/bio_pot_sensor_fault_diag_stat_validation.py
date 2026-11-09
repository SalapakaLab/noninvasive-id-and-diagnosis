
import sys
import os
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.fault_diagnosis import stastical_validation_of_fault_diag_biopot_sensor_vectorized, report_diagnostic_results_biopot_sensor
from src.utils import save_fault_diag_stat_validation_data, load_fault_diag_stat_validation_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = 'data/experimental_results/algo_outputs/bio_pot_sensor_fault_diag_stat_validation.json'
PLOT_DIR = 'data/experimental_results/plots/Fault_Diag_Stat_Validation'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true',
                         help=f'Skip the validation sweep and replot from previously saved data at {DATA_PATH}')
    args = parser.parse_args()

    if args.load:
        conf_matrices, resistance_est_error, capacitance_est_error = load_fault_diag_stat_validation_data(DATA_PATH)
    else:
        conf_matrices, resistance_est_error, capacitance_est_error = run_validation()
        save_fault_diag_stat_validation_data(DATA_PATH, conf_matrices, resistance_est_error, capacitance_est_error)

    report_diagnostic_results_biopot_sensor(conf_matrices, save_dir=PLOT_DIR)

    plot_estimation_error_boxplot(resistance_est_error, capacitance_est_error)

    print(f"Number of cases with error less than 25% for resistance estimation: {np.sum(np.abs(resistance_est_error) < 0.25)}")
    print(f"Number of cases with error less than 25% for capacitance estimation: {np.sum(np.abs(capacitance_est_error) < 0.25)}")

    print(f"Number of cases with error less than 20% for resistance estimation: {np.sum(np.abs(resistance_est_error) < 0.20)}")
    print(f"Number of cases with error less than 20% for capacitance estimation: {np.sum(np.abs(capacitance_est_error) < 0.20)}")

    print(f"Number of cases with error less than 15% for resistance estimation: {np.sum(np.abs(resistance_est_error) < 0.15)}")
    print(f"Number of cases with error less than 15% for capacitance estimation: {np.sum(np.abs(capacitance_est_error) < 0.15)}")

    print(f"Number of cases with error less than 10% for resistance estimation: {np.sum(np.abs(resistance_est_error) < 0.10)}")
    print(f"Number of cases with error less than 10% for capacitance estimation: {np.sum(np.abs(capacitance_est_error) < 0.10)}")

    print(f"Number of cases with error less than 5% for resistance estimation: {np.sum(np.abs(resistance_est_error) < 0.05)}")
    print(f"Number of cases with error less than 5% for capacitance estimation: {np.sum(np.abs(capacitance_est_error) < 0.05)}")


def run_validation():
    ############################################################################################################################################################
    # %% Initialization, data loading and data processing

    nNodes = 5
    n_FFT = 4096
    Adj = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 1, 0]])

    freq_uLim = 10e3
    freq_mLim = 1e2
    freq_lLim = 1

    # Load time series data
    time_series_path = "data/time_series_data/bio_pot_sensor/healthy"
    list_of_nodes = ['a','b','c','d','o']

    data_Normal_HF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_100mHz50kHz.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    f_samp_HF = 1/10e-6
    nperseg_HF, noverlap_HF, win_HF = 4096, 0, 'hann'

    data_Normal_LF = np.column_stack([
        pd.read_csv(f'{time_series_path}/V{node}_10mHz5kHz.csv').iloc[:, 1:].values.flatten() 
        for node in list_of_nodes
    ])
    f_samp_LF = 1/100e-6
    nperseg_LF, noverlap_LF, win_LF = 4096*4, 0, 'hann'
        
    # %% 
    #
    Tag_path = "data/time_series_data/bio_pot_sensor/faulty/Faulty_Params_Tag.csv"
    start_row = 2
    end_row = 61
    conf_matrices, resistance_est_error, capacitance_est_error = stastical_validation_of_fault_diag_biopot_sensor_vectorized(Tag_path, start_row, end_row, data_Normal_HF, data_Normal_LF, f_samp_HF, nperseg_HF, noverlap_HF, win_HF, f_samp_LF, nperseg_LF, noverlap_LF, win_LF, n_FFT, Adj, freq_lLim, freq_mLim, freq_uLim)

    return conf_matrices, resistance_est_error, capacitance_est_error


def plot_estimation_error_boxplot(resistance_est_error, capacitance_est_error):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 8

    # Convert lists to a format seaborn loves (DataFrame)
    max_len = max(len(resistance_est_error), len(capacitance_est_error))
    res_err_padded = resistance_est_error + [None] * (max_len - len(resistance_est_error))
    cap_err_padded = capacitance_est_error + [None] * (max_len - len(capacitance_est_error))
    df = pd.DataFrame({
        'Res.': res_err_padded,
        'Cap.': cap_err_padded
    })

    # Melt the dataframe into "long" format
    df_melted = df.melt(var_name='Component', value_name='Estimation Error')

    # Plotting
    plt.figure(figsize=(1.7, 1.7))

    ax = sns.boxplot(x='Component', y='Estimation Error', data=df_melted,
                    width=0.4, palette="muted", linewidth=0.8,
                    flierprops={"marker": "o", "alpha": 0.4, "markersize": 3})

    # Add the zero-error reference line
    ax.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5, zorder=0)

    # Tweak fonts
    plt.ylabel('Est. Error (Rel.)', fontsize=FONTSIZE)
    plt.xlabel('')
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.tight_layout(pad=0)
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, 'bio_pot_sensor_fault_diag_estimation_error_boxplot.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


# Entry point of the script
if __name__ == "__main__":
    main() 