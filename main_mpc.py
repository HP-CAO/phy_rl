from lib.env.locomotion.envs.a1_env import A1Params
from lib.env.locomotion.envs.a1_env import A1Robot

a1_params = A1Params()
a1 = A1Robot(a1_params)

actions = [0.] * 60
i = 1

for i in range(10000):
    i += 1
    a1.step(actions, action_mode="mpc")

    print(a1.get_states_vector())
