# import math
# import time
import math

from lib.env.gym_physics import GymPhysics, GymPhysicsParams
from lib.env.linear_physics import LinearPhysicsParams, LinearPhysics
from lib.env.reward import RewardParams, RewardFcn
import matplotlib.pyplot as plt
from lib.utils import states2observations

reward_function = RewardFcn(RewardParams())

car_pole_params = GymPhysicsParams()
car_pole_params.ini_states = [0.0, 0.0, 0.1, 0.0, False]
cart_pole = GymPhysics(car_pole_params)
cart_pole.reset()

l_car_pole_params = LinearPhysicsParams()
l_car_pole_params.ini_states = [0.0, 0.0, 0.1, 0.0, False]
l_cart_pole = LinearPhysics(l_car_pole_params)
l_cart_pole.reset()
reward_list = []
steps = range(500)

for _ in steps:

    cart_pole.step(action=0.0)  # stable states
    l_cart_pole.step()
    cart_pole.render()
    l_cart_pole.render()

    print("original_system:", cart_pole.states)
    print("linearized_system:", l_cart_pole.states)
    tracking_reward = reward_function.reference_tracking_error(l_cart_pole.states, cart_pole.states)
    print("reward:", tracking_reward)
    reward_list.append(tracking_reward)

# plt.plot(steps, reward_list)
# plt.show()
    # cart_pole.render(mode='human')
