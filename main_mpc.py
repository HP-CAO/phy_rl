from lib.env.locomotion.envs.a1_env import A1Params
from lib.env.locomotion.envs.a1_env import A1Robot

a1_params = A1Params()
# a1_params.if_add_terrain = True
a1 = A1Robot(a1_params)
a1.add_lane()

actions = [0.] * 60
i = 1

for i in range(int(2e8)):
    i += 1
    s = a1.step(actions, action_mode="mpc")
    a1.get_vision_observations()
    print(a1.get_states_vector())