import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from types import SimpleNamespace as Namespace

path_reward_source = "plot/data_source/reward plot"
path_critic_source = "plot/data_source/critic loss plot"
path_list = [path_reward_source, path_critic_source]

plot_training_steps = 200000
interval_num = 50  # we split 1e6 into 30 intervals to calculate the confidence interval

exp_set_list = os.listdir(path_reward_source)
exp_set_data_list = []
sns.set_style("darkgrid")

points_per_interval = int(plot_training_steps / interval_num)
interval_labels = np.arange(interval_num + 1) * points_per_interval  # [0, 2e4, 4e4...6e5]

plot = None
legend_list = []
data_frame_list = []

for id, sour_path in enumerate(path_list):
    for exp_set in exp_set_list:
        legend_list.append(exp_set)
        set_repeats_path = os.path.join(sour_path, exp_set)
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

            if id == 0:
                meta_data_value = np.log10(-1 * meta_data_value + 0.005)
            else:
                meta_data_value = np.log10(meta_data_value)

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9.5))

l1 = sns.lineplot(data=data_frame_list[0], ax=ax1, x="label", y="Value", label=legend_list[3], legend=False)
l2 = sns.lineplot(data=data_frame_list[1], ax=ax1, x="label", y="Value", label=legend_list[4], legend=False)
l3 = sns.lineplot(data=data_frame_list[2], ax=ax1, x="label", y="Value", label=legend_list[5], legend=False)

l4 = sns.lineplot(data=data_frame_list[3], ax=ax2, x="label", y="Value", label=legend_list[3], legend=False)
l5 = sns.lineplot(data=data_frame_list[4], ax=ax2, x="label", y="Value", label=legend_list[4], legend=False)
l6 = sns.lineplot(data=data_frame_list[5], ax=ax2, x="label", y="Value", label=legend_list[5], legend=False)

# fig.legend([l1, l2, l3], labels=legend_list,
#            loc="lower left", fontsize='small',
#            borderaxespad=0.1,
#            bbox_to_anchor=(.15, 0.1),
#            ncol=1,
#            framealpha=0.0)

ax1.set(xlabel="Training steps", ylabel="Cost\nlog(-R+0.005)")
ax2.set(xlabel="Training steps", ylabel="log(Critic Loss)")
#
# ax1.set(xlabel="Training steps", ylabel="Log(Reward)")
# ax2.set(xlabel="Training steps", ylabel="Log(Critic Loss)")

ax1.title.set_text('(a)')
ax2.title.set_text('(b)')

ax1.legend(loc="upper right", fontsize='small',
           borderaxespad=0.5,
           ncol=1,
           framealpha=0.25)
plt.subplots_adjust(hspace=0.25)

ax2.legend(loc="upper right", fontsize='small',
           borderaxespad=0.5,
           ncol=1,
           framealpha=0.25)
plt.subplots_adjust(wspace=0.22)


plt.savefig(f'plot/training_all.pdf', format='pdf')
plt.tight_layout()
# plt.savefig(f'plot/training_all.png', dpi=300)
