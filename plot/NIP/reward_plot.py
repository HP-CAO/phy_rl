import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from types import SimpleNamespace as Namespace


def plot_group_training(sub_source_folder, save_name, y_label):
    plot_training_steps = 1.0e6
    interval_num = 20  # we split 1e6 into 30 intervals to calculate the confidence interval

    plt.figure()
    exp_set_list = os.listdir(sub_source_folder)
    exp_set_data_list = []
    sns.set_style("darkgrid")

    points_per_interval = int(plot_training_steps / interval_num)
    interval_labels = np.arange(interval_num + 1) * points_per_interval  # [0, 2e4, 4e4...6e5]

    plot = None
    legend_list = []

    # ['FC MLP', 'KN 20', 'KN 25', 'KN 15', 'KN 5', 'KN 10']

    print_list = ['FC MLP', 'KN 15']
    save_name = print_list[0] + print_list[1] + "200k"

    for exp_set in print_list:

        # if exp_set == "S Reward":
        #     continue

        legend_list.append(exp_set)
        set_repeats_path = os.path.join(sub_source_folder, exp_set)
        set_repeats_list = os.listdir(set_repeats_path)

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
            index = np.count_nonzero(meta_data_steps < 200000)
            meta_data_steps = meta_data_steps[index:]
            meta_data_value = meta_data_value[index:]

            # meta_data_value = np.log10(-1 * meta_data_value + 0.005)

            if max(meta_data_steps) > plot_training_steps:
                valid_index = np.where(meta_data_steps < plot_training_steps)
                meta_data_steps = meta_data_steps[valid_index]
                meta_data_value = meta_data_value[valid_index]

            meta_data_interval_label = np.zeros_like(meta_data_value)

            for i in range(len(interval_labels) - 1):

                interval_lower = interval_labels[i]
                interval_upper = interval_labels[i+1]

                index_mask = (meta_data_steps > interval_lower) * (meta_data_steps <= interval_upper)
                index = np.where(index_mask)
                meta_data_interval_label[index] = interval_upper

            set_labels_list.append(meta_data_interval_label)
            set_values_list.append(meta_data_value)

        set_labels = np.concatenate(set_labels_list, axis=-1)
        set_values = np.concatenate(set_values_list, axis=-1)

        data_frame = pd.DataFrame({"label": set_labels, "Value": set_values})
        plot = sns.lineplot(data=data_frame, x="label", y="Value", ci=95)
        plt.xlim(2e5, 1e6)
        # plt.ylim(-0.3, -0.10)
        # plt.xticks([0, 200000, 400000, 600000, 800000, 1000000],
        #            ('0', '200k', '400k', '600k', '800k', '1M')
        #
        plt.xticks([200000, 400000, 600000, 800000, 1000000],
                   ('200k', '400k', '600k', '800k', '1M'))

    plot.set(xlabel="Training Steps", ylabel="Episode Reward")
    # plot.ticklabel_format(useOffset=False, style='plain')
    plot.legend(loc='lower right', fontsize='large', labels=legend_list)
    plt.savefig(f'plot/NIP/{save_name}.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig(f'plot/{save_name}.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default="plot/NIP/source/", help='Activate usage of GPU')
    parser.add_argument('--y_label', default="Reward", help='Enable to write default config only')
    parser.add_argument('--save_name', default="base_15", help='Enable to write default config only')

    args = parser.parse_args()
    source_folder = args.folder
    y_label = args.y_label
    save_name = args.save_name
    plot_group_training(source_folder, save_name, y_label)



