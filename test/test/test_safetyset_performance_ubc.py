import math

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
from matplotlib import cm


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x_t_list = np.linspace(-0.9, 0.9, 80)
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

fig = plt.figure()

# plt.figure(num = 3,figsize = (10,5))
# plt.plot(tx1, tx2, linewidth=2, color='black')

# plt.vlines(x=-0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
# plt.vlines(x=0.9, ymin=-0.85, ymax=0.85, color='black', linewidth=2.5)
# plt.hlines(y=-0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)
# plt.hlines(y=0.8, xmin=-0.95, xmax=0.95, color='black', linewidth=2.5)

# ax.plot(x, y, z, color='black')


plt.xlim(-0.95, 0.95)
plt.ylim(-0.85, 0.85)

# Loading model
params = read_config("./config/eval.json")
model_path = "test_models/nips_unk_our_with_res_1_best"
# model_path_ubc = "test_models/nips_unk_ubc_no_res_1_best"
model_path_ubc = "test_models/nips_unk_ubc_no_res_1.5M_1_best"

agent_our = DDPGAgent(params.agent_params,
                      params.taylor_params,
                      shape_observations=5,
                      shape_action=1,
                      model_path=model_path,
                      mode="test")

agent_ubc = DDPGAgent(params.agent_params,
                      params.taylor_params,
                      shape_observations=5,
                      shape_action=1,
                      model_path=model_path_ubc,
                      mode="test")

# Set interaction env

env = Cartpole(params.cartpole_params)

trajectory = []
tx_list = []


def get_distance_score(position, angle):
    distance_score_factor = 5
    cart_position = position
    pendulum_angle_sin = math.sin(angle)
    pendulum_angle_cos = math.cos(angle)

    target_cart_position = 0
    target_pendulum_angle = 0

    pendulum_length = 0.64

    pendulum_tip_position = np.array(
        [cart_position + pendulum_length * pendulum_angle_sin, pendulum_length * pendulum_angle_cos])

    target_tip_position = np.array(
        [target_cart_position + pendulum_length * np.sin(target_pendulum_angle),
         pendulum_length * np.cos(target_pendulum_angle)])

    distance = np.linalg.norm(target_tip_position - pendulum_tip_position)
    distance_score = np.exp(-distance * distance_score_factor)
    return distance_score


def interact_loop(x_t, theta_t, ai_agent):
    init_states = [x_t, 0., theta_t, 0, False]
    # init_states = [0, 0., 0, 0, False]k
    env.reset(init_states)
    # trajectory
    trajectory = []
    tx_list = []
    position_list = []
    angle_list = []
    performance_list = []

    for step in range(trajectory_length):
        current_states = copy.deepcopy(env.states)
        tx = np.matmul(np.array(current_states)[:4], P_matrix) @ np.array(current_states)[:4].transpose()
        position = current_states[0]
        angle = current_states[2]
        distance_score = get_distance_score(position, angle)
        performance_list.append(distance_score)

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
    average_performance = np.mean(performance_list)

    return tx_array, position_array, angle_array, average_performance

x_list_our = []
y_list_our = []
z_list_our = []

x_list_ubc = []
y_list_ubc = []
z_list_ubc = []

# interaction loop
ubc_result = np.zeros(shape=(len(x_t_list), len(theta_t_list)))
# our_result = np.zeros(shape=(len(x_t_list), len(theta_t_list)))

for i, x_t in tqdm(enumerate(x_t_list)):
    for j, theta_t in enumerate(theta_t_list):
        # tx_array, position_array, angle_array, average_performance = interact_loop(x_t, theta_t, ai_agent=agent_our)
        #
        # if len(position_array[position_array > 0.9]) == 0 \
        #         and len(angle_array[angle_array > 0.8]) == 0:
        #     our_result[i][j] = average_performance

        tx_array_ubc, position_array_ubc, angle_array_ubc, average_performance = interact_loop(x_t, theta_t,
                                                                                               ai_agent=agent_ubc)

        if len(position_array_ubc[position_array_ubc > 0.9]) == 0 \
                and len(angle_array_ubc[angle_array_ubc > 0.8]) == 0:
            ubc_result[i][j] = average_performance

X, Y = np.meshgrid(x_t_list, theta_t_list)

# c = plt.pcolormesh(X, Y, ubc_result, cmap='OrRd', vmin=0, vmax=1)
c = plt.pcolormesh(X, Y, ubc_result, vmin=0, vmax=1)
# plt.set_xlabel(r'$x$', fontsize=10)
# plt.set_ylabel(r"${\Theta}$", fontsize=10)
# plt.set_zlabel("$Performance$", fontsize=10)
fig.colorbar(c)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r"${\Theta}$", fontsize=16)
# plt.grid()
# plt.show()
fig.savefig(f'plot/NIP/safety_set_performance_ubc_2.pdf', format='pdf', bbox_inches='tight')

# fig1.savefig(f'plot/NIP/safety_env_our_vs_ubc_1.5.png', dpi=300, bbox_inches='tight')
