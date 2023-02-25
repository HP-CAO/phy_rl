import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras import Model
import numpy as np


class AugmentLayer(Layer):
    def __init__(self, augment_order):
        """
        the augment order starts from 1, 1 means not augmenting
        """
        super(AugmentLayer, self).__init__()
        self.order = augment_order - 1

    def call(self, inputs, *args, **kwargs):

        input_exp = inputs
        _, c = inputs.shape
        print(inputs.shape)
        exp_list = [input_exp]
        index_list = np.array(range(c))

        for i in range(self.order):
            augment_list = []
            index_next_order = np.zeros(shape=c, dtype=int)
            n_aug_var = 0

            for j in range(c):
                index_next_order[j] += n_aug_var
                exp_tensor = exp_list[-1]
                ind_start = index_list[j]
                N, len_pre = exp_tensor.shape
                input_variable = tf.reshape(inputs[:, j], shape=(N, -1))  # n, 1
                exp_tensor_pre = tf.reshape(exp_tensor[:, ind_start:], shape=(N, -1))
                exp_tem = input_variable * exp_tensor_pre
                n_aug_var = len_pre - ind_start
                augment_list.append(exp_tem)

            index_list = index_next_order
            augment_tensors = tf.concat(augment_list, axis=-1)
            exp_list.append(augment_tensors)

        exp_all = tf.concat(exp_list, axis=-1)
        return exp_all


class TaylorModel(Model):
    def __init__(self):
        super(TaylorModel, self).__init__()
        self.aug_1 = AugmentLayer(augment_order=2)
        self.dense_1 = Dense(units=32, activation="relu", use_bias=True)
        self.aug_2 = AugmentLayer(augment_order=2)
        self.dense_2 = Dense(units=16, activation="relu", use_bias=True)
        self.aug_3 = AugmentLayer(augment_order=3)
        self.last = Dense(units=1, activation="tanh", use_bias=True)

    def call(self, inputs, training=None, mask=None):
        x = self.aug_1(inputs)
        x = self.dense_1(x)
        x = self.aug_2(x)
        x = self.dense_2(x)
        x = self.aug_3(x)
        x = self.last(x)
        return x


if __name__ == '__main__':
    # a = np.array([[1, 2], [1, 2]])
    a = np.array([[1, 2, 3], [1, 2, 3]])
    print(a.shape)
    b = AugmentLayer(augment_order=2)(a)
    print(b)

    # taylor_model = TaylorModel()
    # b = taylor_model(a)
    # print(b)
    # taylor_model.summary()

