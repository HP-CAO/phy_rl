import numpy as np
from lib.agent.network import TaylorParams, TaylorModel

params = TaylorParams()
model = TaylorModel(params, input_dim=4, output_dim=1, output_activation='tanh')
a = np.random.random(size=(2, 4))
b = model(a)
print(b)
model.summary()
