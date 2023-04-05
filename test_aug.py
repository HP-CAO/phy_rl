import numpy as np
from lib.agent.network import TaylorParams, TaylorModel
from lib.agent.network import get_knowledge_matrix

a = get_knowledge_matrix()
for i in range(3):
    print(a[i][0].shape)
    print(a[i][1].shape)


