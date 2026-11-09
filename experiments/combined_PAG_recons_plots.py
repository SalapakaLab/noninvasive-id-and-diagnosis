'''Combines saved PAG reconstruction metrics from multiple circuits into comparison plots:
one figure comparing skeleton recovery, and one figure per edge-mark class comparing
edge-orientation recovery across circuits.'''

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from src.utils import load_pag_recons_metrics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CIRCUITS = [
    ('data/experimental_results/algo_outputs/LDO_metrics.json', 'LDO'),
    ('data/experimental_results/algo_outputs/three_MSTA_TIA_metrics.json', 'MSTA'),
]

EDGE_LABEL_TAGS = {
    'no_edge': 'no_edge',
    'o-o': 'circle_circle',
    '-->': 'arrow_fwd',
    '<--': 'arrow_bwd',
    '---': 'tail_tail',
    'o--': 'circle_tail',
    '--o': 'tail_circle',
}

METRIC_STYLES = {
    'f1': ('-o', 'F-1'),
    'tpr': ('-d', 'TPR'),
    'fpr': ('-*', 'FPR'),
}


def main():
    circuit_data = {name: load_pag_recons_metrics(path) for path, name in CIRCUITS}

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    })
    FONTSIZE = 6

    # Skeleton comparison plot: F1/TPR/FPR for every circuit, overlaid
    plt.figure(figsize=(2.35, 1.7))
    for name, data in circuit_data.items():
        nSamplesList = data['nSamplesList']
        skel = data['skeleton']
        for metric, (style, metric_label) in METRIC_STYLES.items():
            plt.semilogx(nSamplesList, skel[metric], style, label=f'{name} {metric_label}')
    plt.ylabel('Performance Metrics', fontsize=FONTSIZE)
    plt.xlabel('Sample Size', fontsize=FONTSIZE)
    plt.legend(fontsize=FONTSIZE, ncol=2)
    plt.tick_params(axis='both', labelsize=FONTSIZE)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10))))
    ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='', sep=''))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    plt.grid(True)
    plt.title('Skeleton Recovery Metrics', fontsize=FONTSIZE)
    plt.savefig('data/experimental_results/plots/reconstruction_results/combined_skel_recons_error.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Edge-mark comparison plots: one figure per edge-mark class present in any circuit,
    # overlaying whichever circuits have that class
    all_edge_labels = []
    for data in circuit_data.values():
        for label in data['edge_marks']:
            if label not in all_edge_labels:
                all_edge_labels.append(label)

    for label in all_edge_labels:
        label_tag = EDGE_LABEL_TAGS.get(label, ''.join(c if c.isalnum() else '_' for c in label))

        plt.figure(figsize=(2.35, 1.7))
        for name, data in circuit_data.items():
            if label not in data['edge_marks']:
                continue
            nSamplesList = data['nSamplesList']
            edge = data['edge_marks'][label]
            for metric, (style, metric_label) in METRIC_STYLES.items():
                plt.semilogx(nSamplesList, edge[metric], style, label=f'{name} {metric_label}')
        plt.ylabel('Performance Metrics', fontsize=FONTSIZE)
        plt.xlabel('Sample Size', fontsize=FONTSIZE)
        if label_tag == 'arrow_fwd' or label_tag == 'arrow_bwd':
            plt.legend(fontsize=FONTSIZE, ncol=1)
        else:
            if label_tag == 'circle_circle':
                plt.legend(fontsize=FONTSIZE, ncol=2, loc='lower left', bbox_to_anchor=(0.02, 0.08))
            else:
                plt.legend(fontsize=FONTSIZE, ncol=2)
        plt.tick_params(axis='both', labelsize=FONTSIZE)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.xaxis.set_major_formatter(ticker.EngFormatter(unit='', sep=''))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        plt.grid(True)
        if label_tag == 'no_edge':
            plt.title('No Edge Class', fontsize=FONTSIZE)
        elif label_tag == 'arrow_fwd':
            plt.title('Metrics for Edge Type $\\rightarrow$', fontsize=FONTSIZE)
        elif label_tag == 'arrow_bwd':
            plt.title('Metrics for Edge Type $\\leftarrow$', fontsize=FONTSIZE)
        elif label_tag == 'circle_circle':
            plt.title('Metrics for Edge Type $\\circ\mbox{--}\\circ$', fontsize=FONTSIZE)
        plt.savefig(f'data/experimental_results/plots/reconstruction_results/combined_edgpt_recons_error_{label_tag}.pdf', dpi=300, bbox_inches='tight')
        plt.show()


# Entry point of the script
if __name__ == "__main__":
    main()
