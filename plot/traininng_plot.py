import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from types import SimpleNamespace as Namespace


def plot_group_training(sub_source_folder, save_name, y_label):
    plot_training_steps = 100000
    interval_num = 50  # we split 1e6 into 30 intervals to calculate the confidence interval

    plt.figure(figsize=(6, 4))
    exp_set_list = os.listdir(sub_source_folder)
    exp_set_data_list = []
    sns.set_style("darkgrid")

    points_per_interval = int(plot_training_steps / interval_num)
    interval_labels = np.arange(interval_num + 1) * points_per_interval  # [0, 2e4, 4e4...6e5]

    plot = None
    legend_list = []
    data_frame_list = []

    for exp_set in exp_set_list:
        #
        # if exp_set == "S Reward":
        #     continue

        legend_list.append(exp_set)
        set_repeats_path = os.path.join(sub_source_folder, exp_set)
        set_repeats_list = os.listdir(set_repeats_path)
        print(set_repeats_list)

        if len(set_repeats_list) == 0:
            continue

        set_labels_list = []
        set_values_list = []

        for data_name in set_repeats_list:
            meta_data_path = os.path.join(set_repeats_path, data_name)
            json_data = open(meta_data_path, "r").read()
            meta_data = np.array(json.loads(json_data, object_hook=lambda d: Namespace(**d)))  # n, 3

            meta_data_steps = meta_data[:, 1]
            meta_data_value = meta_data[:, 2]

            if max(meta_data_steps) > plot_training_steps:
                valid_index = np.where(meta_data_steps < plot_training_steps)
                meta_data_steps = meta_data_steps[valid_index]
                meta_data_value = meta_data_value[valid_index]

            meta_data_interval_label = np.zeros_like(meta_data_value)

            for i in range(len(interval_labels) - 1):
                interval_lower = interval_labels[i]
                interval_upper = interval_labels[i + 1]

                index_mask = (meta_data_steps > interval_lower) * (meta_data_steps <= interval_upper)
                index = np.where(index_mask)
                meta_data_interval_label[index] = interval_upper

            set_labels_list.append(meta_data_interval_label)
            set_values_list.append(meta_data_value)

        set_labels = np.concatenate(set_labels_list, axis=-1)
        set_values = np.concatenate(set_values_list, axis=-1)

        data_frame = pd.DataFrame({"label": set_labels, "Value": set_values})
        data_frame_list.append(data_frame)

    plot, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    l1 = sns.lineplot(data=data_frame_list[0], ax=ax, x="label", y="Value", label=legend_list[0], legend=True)
    l2 = sns.lineplot(data=data_frame_list[1], ax=ax, x="label", y="Value", label=legend_list[1], legend=True)
    l3 = sns.lineplot(data=data_frame_list[2], ax=ax2, x="label", y="Value", color="green", label=legend_list[2],
                      legend=True)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(-15, -0.1)
    plt.subplots_adjust(hspace=0.04)

    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.set_xlabel('.', color=(0, 0, 0, 0))
    ax.set_ylabel('.', color=(0, 0, 0, 0))
    ax2.set_xlabel('.', color=(0, 0, 0, 0))
    ax2.set_ylabel('.', color=(0, 0, 0, 0))

    plot.supxlabel('Training steps')
    plot.supylabel('Reward')

    ax.legend(loc="lower left", fontsize='medium',
              borderaxespad=0.1,
              bbox_to_anchor=(.45, 0.08),
              ncol=1,
              framealpha=0)

    ax2.legend(loc="lower right", fontsize='medium',
               borderaxespad=0.1,
               bbox_to_anchor=(.45, 1.02),
               ncol=1,
               framealpha=0)

    # plot.legend([l1, l2, l3], labels=legend_list,
    #             loc="lower left", fontsize='small',
    #             borderaxespad=0.1,
    #             bbox_to_anchor=(.15, 0.1),
    #             ncol=1,
    #             framealpha=0.0)
    # plt.savefig(f'plot/{save_name}.pdf', format='pdf')
    plt.savefig(f'plot/{save_name}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default="plot/data_source/reward plot", help='')
    parser.add_argument('--y_label', default="Reward", help='Enable to write default config only')
    parser.add_argument('--save_name', default="reward", help='Enable to write default config only')

    args = parser.parse_args()
    source_folder = args.folder
    y_label = args.y_label
    save_name = args.save_name
    plot_group_training(source_folder, save_name, y_label)
