# here we test the beta distribution thing
import numpy as np
import matplotlib.pyplot as plt


def get_unk_unk_dis(a=None, b=None):
    rng = np.random.default_rng(seed=1)

    if a is None:
        a1 = 11 * np.random.random(1)[0]  # [0, 11]
        a2 = 11 * np.random.random(1)[0]  # [0, 11]

    if b is None:
        b1 = 11 * np.random.random(1)[0]  # [0, 11]
        b2 = 11 * np.random.random(1)[0]  # [0, 11]

    uu1 = -rng.beta(a1, b1) + rng.beta(a2, b2)
    # uu2 = -rng.beta(a, b) + rng.beta(a, b)

    uu1 *= 2.0
    # uu2 *= 2.0

    return uu1


def get_uniform_dis():
    a = np.random.uniform(low=-1.0, high=1.0) * 2.0
    # b = np.random.uniform() * 2.0
    return a


beta_list = []
uni_List = []
num_points = 1000


for _ in range(num_points):
    dis_beta = get_unk_unk_dis()
    dis_uni = get_uniform_dis()
    beta_list.append(dis_beta)
    uni_List.append(dis_uni)

plt.scatter(x=range(num_points), y=beta_list, label='beta')
plt.scatter(x=range(num_points), y=uni_List,  label='uniform')
plt.legend()
plt.show()








