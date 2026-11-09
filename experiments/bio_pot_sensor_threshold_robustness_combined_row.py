'''Produces three separate, equally-sized figures -- the W-PC threshold-sensitivity plot and the
two noise-robustness plots (vs. SNR and vs. noise correlation) -- meant to be placed together as
subfigures in a paper. All three share the same figure size and the same legend font size, so
they look consistent when placed side by side.

This script only reads previously-saved sweep data -- it does not re-run any reconstruction
sweeps itself. Run (or --load) bio_pot_sensor_threshold_sensitivity.py and
bio_pot_sensor_noise_robustness.py first so their JSON caches exist.
'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.utils import load_threshold_sensitivity_data, load_noise_robustness_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Must match the DATA_PATH constants in the source scripts.
THRESH_DATA_PATH = 'data/experimental_results/algo_outputs/bio_pot_sensor_threshold_sensitivity.json'
ROBUSTNESS_DATA_PATH = 'data/experimental_results/algo_outputs/bio_pot_sensor_noise_robustness.json'

# Shared across all three so they look consistent as subfigures.
SUBFIG_SIZE = (2.4, 2.2)  # roughly one-third of the IEEE full text width (~7.16in), each standalone
FONTSIZE = 8
LEGEND_FONTSIZE = FONTSIZE - 1


def plot_threshold_sensitivity_figure(thresh_vec, nSamplesList, f1_results, auto_thresh_results, save_path):
    '''
    Draws the F-1-vs-threshold figure (see bio_pot_sensor_threshold_sensitivity.py): one curve per
    sample size, sparse staggered markers, red stars marking each curve's auto-selected threshold.
    '''

    okabe_ito_colors = ['#0072B2', '#E69F00', '#009E73']
    line_styles = ['-', '--', '-.']
    marker_styles = ['o', 's', '^']
    marker_step = max(1, len(thresh_vec) // 5)
    marker_offsets = [round(idx * marker_step / len(nSamplesList)) for idx in range(len(nSamplesList))]

    plt.figure(figsize=SUBFIG_SIZE)
    ax = plt.gca()
    for idx, nSamples in enumerate(nSamplesList):
        color = okabe_ito_colors[idx % len(okabe_ito_colors)]
        ax.plot(thresh_vec, f1_results[nSamples],
               linestyle=line_styles[idx % len(line_styles)],
               marker=marker_styles[idx % len(marker_styles)],
               markevery=(marker_offsets[idx], marker_step), markersize=4, color=color,
               label=f'N={nSamples}')
        auto_thresh = auto_thresh_results[nSamples]
        if auto_thresh is not None:
            auto_idx = np.argmin(np.abs(np.asarray(thresh_vec) - auto_thresh))
            ax.plot(auto_thresh, f1_results[nSamples][auto_idx], marker='*', markersize=9,
                   color='red', markeredgecolor='black', markeredgewidth=0.5,
                   linestyle='None', zorder=5)

    auto_marker_proxy = mlines.Line2D([], [], marker='*', markersize=9, color='none',
                                      markerfacecolor='red', markeredgecolor='black',
                                      markeredgewidth=0.5, linestyle='None', label='Auto-sel.')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [auto_marker_proxy], fontsize=LEGEND_FONTSIZE, loc='best')

    ax.set_xlabel('Wiener-PC Threshold', fontsize=FONTSIZE)
    ax.set_ylabel('F-1 Score', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_robustness_snr_figure(SNR_dB_list, snr_metrics, save_path):
    '''
    Draws the performance-vs-SNR figure (see bio_pot_sensor_noise_robustness.py): F1/TPR/FPR each
    as their own Okabe-Ito-colored, distinctly-styled curve.
    '''

    OKABE_ITO_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9']
    metric_styles = {'f1': ('-', 'o'), 'tpr': ('--', 'd'), 'fpr': (':', '*')}
    metric_labels = {'f1': 'F-1', 'tpr': 'TPR', 'fpr': 'FPR'}

    plt.figure(figsize=SUBFIG_SIZE)
    ax = plt.gca()
    for idx, key in enumerate(['f1', 'tpr', 'fpr']):
        linestyle, marker = metric_styles[key]
        ax.plot(SNR_dB_list, snr_metrics[key], linestyle=linestyle, marker=marker,
               color=OKABE_ITO_COLORS[idx % len(OKABE_ITO_COLORS)], markersize=4,
               label=metric_labels[key])
    ax.invert_xaxis()
    ax.set_xlabel('SNR (dB)', fontsize=FONTSIZE)
    ax.set_ylabel('Performance Metrics', fontsize=FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_robustness_rho_figure(rho_list, SNR_dB_list_for_rho_sweep, rho_metrics_by_snr, save_path):
    '''
    Draws the performance-vs-correlation-coefficient figure (see bio_pot_sensor_noise_robustness.py):
    color encodes SNR, linestyle/marker encodes metric, with a combined two-section legend.
    '''

    OKABE_ITO_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#56B4E9']
    metric_styles = {'f1': ('-', 'o'), 'tpr': ('--', 'd'), 'fpr': (':', '*')}
    metric_labels = {'f1': 'F-1', 'tpr': 'TPR', 'fpr': 'FPR'}

    plt.figure(figsize=SUBFIG_SIZE)
    ax = plt.gca()
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
    # This legend has the most entries (2 headers + 3 metrics + N(SNR) colors) of the three
    # figures, so it uses 2 columns and tighter spacing to still fit at the shared font size.
    ax.legend(handles=[metric_header] + style_handles + [snr_header] + color_handles,
             fontsize=LEGEND_FONTSIZE, loc='upper left', bbox_to_anchor=(0.02, 0.85), ncol=2,
             handlelength=1.2, labelspacing=0.25, columnspacing=0.6, framealpha=1.0)

    ax.set_xlabel('Noise correlation coefficient', fontsize=FONTSIZE)
    ax.set_ylabel('Performance Metrics', fontsize=FONTSIZE)
    ax.tick_params(axis='both', labelsize=FONTSIZE)
    ax.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def load_required_data():
    '''
    Loads both source scripts' cached sweep data, raising a clear error if either is missing.
    '''

    for path, source_script in [(THRESH_DATA_PATH, 'bio_pot_sensor_threshold_sensitivity.py'),
                                (ROBUSTNESS_DATA_PATH, 'bio_pot_sensor_noise_robustness.py')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing cached data at '{path}'. Run `python experiments/{source_script}` "
                f"first (or with --load, if it has already been run) so this file exists.")

    thresh_data = load_threshold_sensitivity_data(THRESH_DATA_PATH)
    robustness_data = load_noise_robustness_data(ROBUSTNESS_DATA_PATH)
    return thresh_data, robustness_data


def main():
    (thresh_vec, nSamplesList, f1_results, auto_thresh_results), \
        (SNR_dB_list, snr_metrics, rho_list, SNR_dB_list_for_rho_sweep, rho_metrics_by_snr) = load_required_data()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })

    save_dir = 'data/experimental_results/plots/Combined_Figures'
    os.makedirs(save_dir, exist_ok=True)

    plot_threshold_sensitivity_figure(thresh_vec, nSamplesList, f1_results, auto_thresh_results,
                                      f'{save_dir}/bio_pot_sensor_threshold_sensitivity.pdf')
    plot_robustness_snr_figure(SNR_dB_list, snr_metrics,
                               f'{save_dir}/bio_pot_sensor_robustness_vs_SNR.pdf')
    plot_robustness_rho_figure(rho_list, SNR_dB_list_for_rho_sweep, rho_metrics_by_snr,
                               f'{save_dir}/bio_pot_sensor_robustness_vs_rho.pdf')

    plt.show()


# Entry point of the script
if __name__ == "__main__":
    main()
