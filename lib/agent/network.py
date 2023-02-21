import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import Model
import numpy as np


def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float16)
    dense1 = Dense(256, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(64, activation='relu', name=name + 'dense3')(dense2)
    output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
    model = Model(inputs=input, outputs=output, name=name)
    return model


def weight_variable(shape, var_name, distribution, scale=0.1, trainable=True):
    if distribution == 'tn':
        initial = tf.random.truncated_normal(shape, stddev=scale, dtype=tf.float32)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random.normal(shape, mean=0, stddev=scale, dtype=tf.float32)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError('Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.'
                             % (var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name, trainable=trainable)


def bias_variable(shape, var_name, distribution, trainable=True):
    """Create a variable for a bias vector.

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')

    Returns:
        a TensorFlow variable for a bias vector
    """

    if distribution == 'uniform':
        initial = tf.random.uniform(shape, minval=-0.2, maxval=0.2, dtype=tf.float32)
    elif distribution == 'normal':
        initial = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    elif distribution == 'none':
        initial = tf.constant(1, shape=shape, dtype=tf.float32)
    else:
        raise NotImplementedError
    return tf.Variable(initial, name=var_name, trainable=trainable)


class TaylorLayer(Layer):
    def __init__(self, weights_shape, var_name, weight_init_dis, bias_init_dis, scale, cp_type, exp_order, act_type,
                 phyweights, inx_phy, inx_unk, pre_weight_A, unkweights):
        super(TaylorLayer, self).__init__()
        self.weights_variables = weight_variable(weights_shape, var_name=var_name, distribution=weight_init_dis,
                                                 scale=scale, trainable=True)
        self.biases_variables = bias_variable([weights_shape[0], 1], var_name=var_name, distribution=bias_init_dis)
        self.aa = 0.001
        self.bb = 100
        self.cp_type = cp_type
        self.exp_order = exp_order
        self.act_type = act_type
        self.phyweights = phyweights
        self.inx_phy = inx_phy
        self.inx_unk = inx_unk
        self.pre_weight_A = pre_weight_A
        self.unkweights = unkweights

    def call(self, inputs, *args, **kwargs):

        prev_layer = inputs
        inx_unk = self.inx_unk
        pre_weight_A = self.pre_weight_A
        unkweights = self.unkweights
        inx_phy = self.inx_phy

        tem_temp = None

        if self.cp_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif self.cp_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif self.cp_type == 'lin':
            prev_layer = tf.multiply(prev_layer, self.aa) + self.bb
        else:
            prev_layer = prev_layer

        input_shape = prev_layer.shape[1:]
        Id = np.arange(input_shape[0])

        input_epd = prev_layer

        # create Taylor expansion here we can use matrix multiplication in the future
        for _ in range(self.exp_order):
            for j in range(input_shape[0]):
                for q in range(input_shape[1]):
                    x_temp = tf.multiply(prev_layer[:, j, q], input_epd[:, Id[j]:(Id[input_shape[0] - 1] + 1), q])
                    x_temp = tf.expand_dims(x_temp, 2)
                    if q == 0:
                        tem_temp = x_temp
                    else:
                        tem_temp = tf.concat((tem_temp, x_temp), 2)
                Id[j] = input_epd.shape[1]
                input_epd = tf.concat((input_epd, tem_temp), 1)

        prev_layer = tf.matmul(self.weights_variables, input_epd) + self.biases_variables

        if self.act_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif self.act_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif self.act_type == 'elu':
            prev_layer = tf.nn.elu(prev_layer)
        elif self.act_type == 'none':
            prev_layer = prev_layer

        # pre_bias = tf.multiply(self.biases_variables, inx_unk)
        # pre_weight_B = tf.multiply(self.weights_variables, unkweights)
        # prev_layer_A = tf.matmul(pre_weight_A, input_epd) + inx_phy
        # prev_layer_B = tf.matmul(pre_weight_B, input_epd) + pre_bias

        # if self.act_type == 'sigmoid':
        #     prev_layer_B = pysigmoid(prev_layer_B, inx_unk)
        # elif self.act_type == 'relu':
        #     prev_layer_B = pyrelu(prev_layer_B, inx_unk)
        # elif self.act_type == 'elu':
        #     prev_layer_B = pyelu(prev_layer_B, inx_unk)
        # else:
        #     prev_layer_B = prev_layer_B
        # return prev_layer_A + prev_layer_B

        return prev_layer


def pyelu(x, inx_unk):
    ex_unk = tf.nn.elu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex


def pyrelu(x, inx_unk):
    ex_unk = tf.nn.relu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex


def pysigmoid(x, inx_unk):
    ex_unk = tf.nn.sigmoid(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex


def pytanh(x, inx_unk):
    ex_unk = tf.nn.tanh(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex


def pysoftsign(x, inx_unk):
    ex_unk = tf.nn.softsign(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex


def pyselu(x, inx_unk):
    ex_unk = tf.nn.selu(x)
    ex = tf.multiply(ex_unk, inx_unk)
    return ex
