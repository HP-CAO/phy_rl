import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
from matplotlib.markers import MarkerStyle
import copy
from lib.agent.ddpg import DDPGAgent, DDPGParams
from lib.env.cart_pole import CartpoleParams, Cartpole, states2observations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x_t_list = np.linspace(-0.9, 0.9, 90)
theta_t_list = np.linspace(-0.8, 0.8, 80)

trajectory_length = 150
P_matrix = np.array([[4.6074554, 1.49740096, 5.80266046, 0.99189224],
                     [1.49740096, 0.81703147, 2.61779592, 0.51179642],
                     [5.80266046, 2.61779592, 11.29182733, 1.87117709],
                     [0.99189224, 0.51179642, 1.87117709, 0.37041435]])

cP = P_matrix

tP = np.zeros((2, 2))

tP[0][0] = cP[0][0]
tP[1][1] = cP[2][2]
tP[0][1] = cP[0][2]
tP[1][0] = cP[0][2]

wp, vp = LA.eig(tP)

theta = np.linspace(-np.pi, np.pi, 1000)

ty1 = (np.cos(theta)) / np.sqrt(wp[0])
ty2 = (np.sin(theta)) / np.sqrt(wp[1])

ty = np.stack((ty1, ty2))
tQ = inv(vp.transpose())
tx = np.matmul(tQ, ty)

tx1 = np.array(tx[0]).flatten()
tx2 = np.array(tx[1]).flatten()

# fig1 = plt.figure()
fig, axs = plt.subplots(2)
# plt.figure(num = 3,figsize = (10,5))
axs[0].plot(tx1, tx2, linewidth=2, color='black')
axs[0].vlines(x=-0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
axs[0].vlines(x=0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
axs[0].hlines(y=-0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
axs[0].hlines(y=0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
axs[0].set_xlim(-1.2, 1.2)
axs[0].set_ylim(-1, 1)

axs[1].plot(tx1, tx2, linewidth=2, color='black')
axs[1].vlines(x=-0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
axs[1].vlines(x=0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
axs[1].hlines(y=-0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
axs[1].hlines(y=0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
axs[1].set_xlim(-1.2, 1.2)
axs[1].set_ylim(-1, 1)

# Loading model
params = read_config("./config/eval.json")
model_path = "test_models/nips_unk_our_with_res_1_best"
# model_path_no_unk = "test_models/batch_2_nips_no_unk_our_with_res_1_best"
model_path_ubc = "test_models/nips_unk_ubc_no_res_2M_2_best"

agent_our = DDPGAgent(params.agent_params,
                      params.taylor_params,
                      shape_observations=5,
                      shape_action=1,
                      model_path=model_path,
                      mode="test")

agent_no_unk = DDPGAgent(params.agent_params,
                         params.taylor_params,
                         shape_observations=5,
                         shape_action=1,
                         model_path=model_path_ubc,
                         mode="test")

# Set interaction env

env = Cartpole(params.cartpole_params)

trajectory = []
tx_list = []


def interact_loop(x_t, theta_t, ai_agent):
    init_states = [x_t, 0., theta_t, 0, False]
    # init_states = [0, 0., 0, 0, False]
    env.reset(init_states)
    # trajectory
    trajectory = []
    tx_list = []
    position_list = []
    angle_list = []

    for step in range(trajectory_length):
        current_states = copy.deepcopy(env.states)
        tx = np.matmul(np.array(current_states)[:4], P_matrix) @ np.array(current_states)[:4].transpose()

        position = current_states[0]
        angle = current_states[2]

        position_list.append(position)
        angle_list.append(angle)

        tx_list.append(tx)
        trajectory.append(current_states)
        observations, _ = states2observations(current_states)
        action = ai_agent.get_action(observations, mode="test")
        env.step(action, use_residual=True)

    tx_array = np.array(tx_list)
    position_array = np.abs(position_list)
    angle_array = np.abs(angle_list)

    return tx_array, position_array, angle_array


# interaction loop
for x_t in tqdm(x_t_list):
    for theta_t in theta_t_list:
        tx_array_ubc, position_array_ubc, angle_array_ubc = interact_loop(x_t, theta_t, ai_agent=agent_no_unk)

        if len(tx_array_ubc[tx_array_ubc > 1]) == 0:
            p1 = axs[0].scatter(x_t, theta_t, c='blue', s=8)
        elif len(position_array_ubc[position_array_ubc > 0.9]) == 0 \
                and len(angle_array_ubc[angle_array_ubc > 0.8]) == 0:
            p2 = axs[0].scatter(x_t, theta_t, c='green', s=8)

        tx_array, position_array, angle_array = interact_loop(x_t, theta_t, ai_agent=agent_our)

        if len(tx_array[tx_array > 1]) == 0:
            p3 = axs[1].scatter(x_t, theta_t, c='blue', s=8)
        elif len(position_array[position_array > 0.9]) == 0 \
                and len(angle_array[angle_array > 0.8]) == 0:
            p4 = axs[1].scatter(x_t, theta_t, c='green', s=8)

plt.xlabel(r'$x$', fontsize=16)

axs[0].set_ylabel(r"${\Theta}$", fontsize=16)
axs[1].set_ylabel(r"${\Theta}$", fontsize=16)
axs[0].grid()
axs[1].grid()

plt.show()
# fig.savefig(f'plot/NIP/safety_env.pdf', format='pdf', bbox_inches='tight')
fig.savefig(f'plot/NIP/safety_env_sub.png', dpi=300, bbox_inches='tight')
