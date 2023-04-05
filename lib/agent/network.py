import copy

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras import Model
import numpy as np


class TaylorParams:
    def __init__(self):
        self.dense_dims = [10, 8]  # dim for hidden layers, not including input dim and output dim
        self.aug_order = [1, 1, 0]  # augmentation order for all hidden layers. 2 will lead to 3rd order
        self.initializer_w = 'tn'
        self.initializer_b = 'uniform'
        self.activations = ['relu', 'relu']  # activations for hidden layers, not including output


class TaylorModel(Model):
    def __init__(self, params: TaylorParams, input_dim, output_dim, output_activation, taylor_editing=False):
        super(TaylorModel, self).__init__()
        dim_list = [input_dim, params.dense_dims[0], params.dense_dims[1], output_dim]
        aug_order = params.aug_order
        activation_list = params.activations
        activation_list.append(output_activation)
        weights_shape = exp_length(dim_list, aug_order).astype(np.int64)  # w = [dim_neurons, input_dim]
        num_layers = len(weights_shape)

        self.layer_list = []

        if taylor_editing:
            editing_matrix = get_knowledge_matrix()
        else:
            editing_matrix = [None] * num_layers

        for i in range(num_layers):
            if aug_order[i] != 0:
                aug_layer = TaylorAugmentLayer(augment_order=aug_order[i])
                self.layer_list.append(aug_layer)

            taylor_dense_layer = TaylorDenseLayer(input_dim=weights_shape[i][1],
                                                  units=weights_shape[i][0],
                                                  name='taylor_dense',
                                                  init_w=params.initializer_w,
                                                  init_b=params.initializer_b,
                                                  scale=0.1,
                                                  trainable=True,
                                                  activation=activation_list[i],
                                                  editing_matrix=editing_matrix[i])

            self.layer_list.append(taylor_dense_layer)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        return x


class TaylorAugmentLayer(Layer):
    def __init__(self, augment_order):
        """
        the augment order starts from 1, 1 means not augmenting
        """
        super(TaylorAugmentLayer, self).__init__()
        self.order = augment_order

    def call(self, inputs, *args, **kwargs):

        input_exp = inputs
        _, c = inputs.shape
        exp_list = [input_exp]
        index_list = np.array(range(c))
        index_next_order = np.zeros(shape=c + 1, dtype=int)

        for i in range(self.order):
            augment_list = []
            num_aug_term = 0

            for j in range(c):
                exp_tensor = exp_list[-1]
                ind_start = index_list[j]
                N, len_pre = exp_tensor.shape
                input_variable = tf.reshape(inputs[:, j], shape=(N, -1))  # n, 1
                exp_tensor_pre_slice = tf.reshape(exp_tensor[:, ind_start:], shape=(N, -1))
                exp_tem = input_variable * exp_tensor_pre_slice
                num_aug_term += exp_tensor_pre_slice.shape[-1]
                index_next_order[j + 1] = num_aug_term
                augment_list.append(exp_tem)

            index_list = copy.deepcopy(index_next_order)
            augment_tensors = tf.concat(augment_list, axis=-1)
            exp_list.append(augment_tensors)

        exp_all = tf.concat(exp_list, axis=-1)
        return exp_all


class TaylorDenseLayer(Layer):
    def __init__(self, input_dim, units, name, init_w, init_b, activation, scale=0.1, trainable=True,
                 editing_matrix=None):
        """
        input_dim: the dimension of the input vector
        units: the dimension of the output, corresponding to the number of neurons
        Usage example:
        if the input X is a batch of 1D vector of m dimensions X: [bs, m] and the units is n
        this layer will create a weights matrix W with the shape of [n, m] and bias vector with the shape of n
        the output will be Y = WX + b, leading to an output tensor with [bs, n]
        """
        super(TaylorDenseLayer, self).__init__()
        weights_shape = [units, input_dim]
        bias_shape = [units]
        self.weights_variables = weight_variable(weights_shape, name=name, distribution=init_w, scale=scale,
                                                 trainable=trainable)
        self.biases_variables = bias_variable(bias_shape, name=name, distribution=init_b, trainable=trainable)
        self.activation = activation

        if editing_matrix is not None:
            self.weights_editing_matrix = editing_matrix[0]
            self.bias_editing_matrix = editing_matrix[1]
            self.nn_editing = True
        else:
            self.nn_editing = False

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)  # [bs, dim, 1]

        # Here to do network editing
        # self.weights_variables ---> # [n, dim] self.biased_variables -----> m
        # tf.linalg.matmul(self.weights_variables, inputs) -----> [bs, n, 1]
        # tf.squeeze ----> [bs, n]
        # logits -----> [bs, n]

        if self.nn_editing:
            logits = tf.squeeze(tf.linalg.matmul(self.weights_variables * self.weights_editing_matrix, inputs), axis=-1) \
                     + self.biases_variables * tf.squeeze(self.bias_editing_matrix, axis=-1)
        else:
            logits = tf.squeeze(tf.linalg.matmul(self.weights_variables, inputs), axis=-1) + self.biases_variables

        if self.activation == 'sigmoid':
            y = tf.sigmoid(logits)
        elif self.activation == 'relu':
            y = tf.nn.relu(logits)
        elif self.activation == 'lin':
            aa = 0.001
            bb = 100
            y = tf.multiply(logits, aa) + bb
        elif self.activation == 'tanh':
            y = tf.nn.tanh(logits)
        else:
            y = logits
        return y


def exp_length(output_size, epd):
    """Generate shape list of expanded layer.

    Arguments:
        output_size -- [input dimension, layer output size list]
        epd         -- layer expansion order list
    Returns:
        shape list of expanded layer
    """

    layer_shape = np.zeros((len(epd), 2))  # layer shape width
    for layer_index in range(len(output_size) - 1):
        expansion_index = np.ones([output_size[layer_index], 1])  # expansion index
        EP_length = np.sum(expansion_index)  # expansion length
        if epd[layer_index] >= 1:
            for ed in range(epd[layer_index]):
                for g in range(output_size[layer_index]):
                    expansion_index[g] = np.sum(expansion_index[g:(output_size[layer_index])])
                EP_length = np.sum(expansion_index) + EP_length
        layer_shape[layer_index, 0] = output_size[layer_index + 1]
        layer_shape[layer_index, 1] = EP_length
    return layer_shape


def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float16)
    dense1 = Dense(256, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(64, activation='relu', name=name + 'dense3')(dense2)
    output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
    model = Model(inputs=input, outputs=output, name=name)
    return model


def weight_variable(shape, name, distribution, scale=0.1, trainable=True):
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
                             % (name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=name, trainable=trainable)


def bias_variable(shape, name, distribution, trainable=True):
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
    return tf.Variable(initial, name=name, trainable=trainable)


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


def get_knowledge_matrix():
    out1 = 10
    out2 = 8
    a_input_dim = 27
    params = {}

    CPhy_lay1_A = np.zeros((out1, a_input_dim), dtype=np.float32)
    CPhy_lay1_B = np.ones((out1, a_input_dim), dtype=np.float32)
    CphyBias_lay1_A = np.zeros((out1, 1), dtype=np.float32)
    CphyBias_lay1_B = np.ones((out1, 1), dtype=np.float32)

    CPhy_lay1_B[0][0] = 0
    CPhy_lay1_B[0][1] = 0
    CPhy_lay1_B[0][2] = 0
    CPhy_lay1_B[0][3] = 0
    CPhy_lay1_B[0][4] = 0
    CPhy_lay1_B[0][5] = 0

    CPhy_lay1_B[1][0] = 0
    CPhy_lay1_B[1][1] = 0
    CPhy_lay1_B[1][2] = 0
    CPhy_lay1_B[1][3] = 0
    CPhy_lay1_B[1][4] = 0
    CPhy_lay1_B[1][5] = 0

    CPhy_lay1_B[2][0] = 0
    CPhy_lay1_B[2][1] = 0
    CPhy_lay1_B[2][2] = 0
    CPhy_lay1_B[2][3] = 0
    CPhy_lay1_B[2][4] = 0
    CPhy_lay1_B[2][5] = 0

    CPhy_lay1_B[3][0] = 0
    CPhy_lay1_B[3][1] = 0
    CPhy_lay1_B[3][2] = 0
    CPhy_lay1_B[3][3] = 0
    CPhy_lay1_B[3][4] = 0
    CPhy_lay1_B[3][5] = 0

    CPhy_lay1_B[4][0] = 0
    CPhy_lay1_B[4][1] = 0
    CPhy_lay1_B[4][2] = 0
    CPhy_lay1_B[4][3] = 0
    CPhy_lay1_B[4][4] = 0
    CPhy_lay1_B[4][5] = 0

    CPhy_lay1_B[5][0] = 0
    CPhy_lay1_B[5][1] = 0
    CPhy_lay1_B[5][2] = 0
    CPhy_lay1_B[5][3] = 0
    CPhy_lay1_B[5][4] = 0
    CPhy_lay1_B[5][5] = 0

    CPhy_lay1_B[6][0] = 0
    CPhy_lay1_B[6][1] = 0
    CPhy_lay1_B[6][2] = 0
    CPhy_lay1_B[6][3] = 0
    CPhy_lay1_B[6][4] = 0
    CPhy_lay1_B[6][5] = 0

    CPhy_lay1_B[7][0] = 0
    CPhy_lay1_B[7][1] = 0
    CPhy_lay1_B[7][2] = 0
    CPhy_lay1_B[7][3] = 0
    CPhy_lay1_B[7][4] = 0
    CPhy_lay1_B[7][5] = 0

    CPhy_lay1_B[8][0] = 0
    CPhy_lay1_B[8][1] = 0
    CPhy_lay1_B[8][2] = 0
    CPhy_lay1_B[8][3] = 0
    CPhy_lay1_B[8][4] = 0
    CPhy_lay1_B[8][5] = 0

    CPhy_lay1_B[9][0] = 0
    CPhy_lay1_B[9][1] = 0
    CPhy_lay1_B[9][2] = 0
    CPhy_lay1_B[9][3] = 0
    CPhy_lay1_B[9][4] = 0
    CPhy_lay1_B[9][5] = 0
    #######################

    ######second layer######
    out1_a = 65

    CPhy_lay2_A = np.zeros((out2, out1_a), dtype=np.float32)
    CPhy_lay2_B = np.ones((out2, out1_a), dtype=np.float32)
    CphyBias_lay2_A = np.zeros((out2, 1), dtype=np.float32)
    CphyBias_lay2_B = np.ones((out2, 1), dtype=np.float32)
    #######################

    ######third layer######
    CPhy_lay3_A = np.zeros((1, out2), dtype=np.float32)
    CPhy_lay3_B = np.ones((1, out2), dtype=np.float32)
    CphyBias_lay3_A = np.zeros((1, 1), dtype=np.float32)
    CphyBias_lay3_B = np.ones((1, 1), dtype=np.float32)

    params['phyweightsA'] = [CPhy_lay1_A, CPhy_lay2_A, CPhy_lay3_A]
    params['phyweightsB'] = [CPhy_lay1_B, CPhy_lay2_B, CPhy_lay3_B]
    params['phybiasesA'] = [CphyBias_lay1_A, CphyBias_lay2_A, CphyBias_lay3_A]
    params['phybiasesB'] = [CphyBias_lay1_B, CphyBias_lay2_B, CphyBias_lay3_B]

    editing_matrix = []

    for k in range(len(params['phyweightsA'])):
        weights_matrix = params['phyweightsB'][k]
        bias_matrix = params['phybiasesB'][k]
        editing_matrix.append([weights_matrix, bias_matrix])

    return editing_matrix
