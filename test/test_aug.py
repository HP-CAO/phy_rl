import numpy as np
from lib.agent.network import TaylorParams, TaylorModel
from lib.agent.network import get_knowledge_matrix_new

a = get_knowledge_matrix_new()

for i in range(3):
    # print(a[i][0].shape)
    # print(a[i][1].shape)
    # print(a[i][2].shape)
    # print(a[i][3].shape)
    print("==========================")
    print(a[i][0])
    print(a[i][1])
    print(a[i][2])
    print(a[i][3])




