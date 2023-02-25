import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
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
    def __init__(self, aug_order, output_activation=None):
        super(TaylorModel, self).__init__()
        # more initializer to check here https://www.tensorflow.org/api_docs/python/tf/keras/initializers
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=None)
        initializer = None
        self.aug_1 = AugmentLayer(augment_order=aug_order)

        self.dense_1 = Dense(units=32, activation="relu", use_bias=True, kernel_initializer=initializer,
                             bias_initializer=initializer)

        self.aug_2 = AugmentLayer(augment_order=aug_order)
        self.dense_2 = Dense(units=16, activation="relu", use_bias=True, kernel_initializer=initializer,
                             bias_initializer=initializer)

        self.aug_3 = AugmentLayer(augment_order=aug_order)
        self.last = Dense(units=1, activation=output_activation, use_bias=True, kernel_initializer=initializer,
                          bias_initializer=initializer)

    def call(self, inputs, training=None, mask=None):
        x = self.aug_1(inputs)
        x = self.dense_1(x)
        x = self.aug_2(x)
        x = self.dense_2(x)
        x = self.aug_3(x)
        x = self.last(x)
        return x


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

        if self.cp_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif self.cp_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif self.cp_type == 'lin':
            prev_layer = tf.multiply(prev_layer, self.aa) + self.bb
        elif self.cp_type == 'tanh':
            prev_layer = tf.nn.tanh(prev_layer)
        else:
            prev_layer = prev_layer

        if self.cp_type == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif self.cp_type == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif self.cp_type == 'lin':
            prev_layer = tf.multiply(prev_layer, self.aa) + self.bb
        else:
            prev_layer = prev_layer

        print(prev_layer)
        print(prev_layer.shape)
        print(tf.reshape(prev_layer, shape=(prev_layer.shape[1], prev_layer.shape[0])))
        # [none, 5]
        input_shape = prev_layer.shape[1:]
        Id = np.arange(input_shape[0])

        input_epd = prev_layer
        tem_temp = None
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

        # prev_layer = tf.matmul(self.weights_variables, input_epd) + self.biases_variables

        # Id = np.arange(prev_layer.shape[1])
        #
        # print("before:", prev_layer.shape)
        # N, c = prev_layer.shape
        # prev_layer = tf.reshape(prev_layer, shape=(c, -1))
        # input_epd = prev_layer
        # input_shape = prev_layer.shape
        # tem_temp = None
        #
        # # create Taylor expansion here we can use matrix multiplication in the future
        # for _ in range(self.exp_order):
        #     for j in range(input_shape[0]):
        #         for q in range(input_shape[1]):
        #             x_temp = tf.multiply(prev_layer[:, j, q], input_epd[:, Id[j]:(Id[input_shape[0] - 1] + 1), q])
        #             x_temp = tf.expand_dims(x_temp, 1)
        #             if q == 0:
        #                 tem_temp = x_temp
        #             else:
        #                 tem_temp = tf.concat((tem_temp, x_temp), 1)
        #         Id[j] = input_epd.shape[0]
        #
        #         print("input", input_epd.shape)
        #         print("tem", tem_temp.shape)
        #         input_epd = tf.concat((input_epd, tem_temp), 0)

        # for _ in range(self.exp_order):
        #     augment_list = []
        #     x_temp = None
        #     for j in range(prev_layer.shape[1]):
        #         x_temp = tf.multiply(prev_layer[:, j], input_epd[:, Id[j]:input_epd.shape[1]])
        #         Id[j] = x_temp.shape[1]
        #         augment_list.append(x_temp)
        #     Id += len(augment_list)
        #     input_epd = tf.concat((input_epd, x_temp), axis=1)

        # prev_layer_reshape_1 = tf.reshape(prev_layer, shape=(-1, prev_layer.shape[1], 1, 1))
        # prev_layer_reshape_2 = tf.reshape(prev_layer, shape=(-1, 1, prev_layer.shape[1], 1))
        # prev_layer_reshape_3 = tf.reshape(prev_layer, shape=(-1, 1, 1, prev_layer.shape[1]))
        #
        # input_epd = prev_layer_reshape_1 * prev_layer_reshape_2 * prev_layer_reshape_3
        # input_epd = tf.linalg.band_part(input_epd, 0, -1)
        # print(input_epd.shape)
        #
        # input_epd = tf.keras.layers.Flatten()(input_epd)
        print("input_epd_shape", input_epd.shape)
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
        print("after:", prev_layer.shape)
        prev_layer = tf.reshape(prev_layer, shape=(-1, c))
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


def build_critic_model(input_dim, output_dim, output_activation):
    params = {}
    out1 = 10
    out2 = 8

    params['data_name'] = 'Car_test'
    params['seed'] = 10
    params['uncheckable_dist_weights'] = ['tn', 'tn', 'tn']
    params['uncheckable_output_size'] = [input_dim, out1, out2, output_dim]
    params['uncheckable_epd'] = np.array([1, 1, 0])
    params['uncheckable_act'] = ['elu', 'elu', output_activation]
    params['uncheckable_com_type1'] = ['none', 'none', 'none']
    params['uncheckable_com_type2'] = ['none', 'none', 'none']
    params['uncheckable_dist_biases'] = ['normal', 'normal', 'normal']
    params['uncheckable_num_of_layers'] = len(np.array([0, 0, 0]))

    T = 0.005

    a_input_dim = 27

    Phy_lay1_A = np.zeros((out1, a_input_dim), dtype=np.float32)
    Phy_lay1_B = np.zeros((out1, a_input_dim), dtype=np.float32)
    phyBias_lay1_A = np.zeros((out1, 1), dtype=np.float32)
    phyBias_lay1_B = np.ones((out1, 1), dtype=np.float32)

    Phy_lay1_A[0][0] = 1
    Phy_lay1_A[0][3] = T
    Phy_lay1_A[1][1] = 1
    Phy_lay1_A[1][4] = T
    Phy_lay1_A[2][2] = 1
    Phy_lay1_A[2][5] = T
    Phy_lay1_A[3][3] = 1
    Phy_lay1_A[4][4] = 1
    Phy_lay1_A[5][5] = 1

    Phy_lay1_B[3][0] = 1
    Phy_lay1_B[3][3] = 1
    Phy_lay1_B[3][6] = 1
    Phy_lay1_B[3][9] = 1
    Phy_lay1_B[3][21] = 1
    Phy_lay1_B[4] = 1
    Phy_lay1_B[5] = 1
    Phy_lay1_B[6] = 1
    Phy_lay1_B[7] = 1
    Phy_lay1_B[8] = 1
    Phy_lay1_B[9] = 1

    phyBias_lay1_B[0:3] = 0

    out1_a = 65

    Phy_lay2_A = np.zeros((out2, out1_a), dtype=np.float32)
    Phy_lay2_B = np.zeros((out2, out1_a), dtype=np.float32)
    phyBias_lay2_A = np.zeros((out2, 1), dtype=np.float32)
    phyBias_lay2_B = np.ones((out2, 1), dtype=np.float32)

    Phy_lay2_A[0][0] = 1
    Phy_lay2_A[1][1] = 1
    Phy_lay2_A[2][2] = 1
    Phy_lay2_A[3][3] = 1

    Phy_lay2_B[3][0] = 1
    Phy_lay2_B[3][3] = 1
    Phy_lay2_B[3][10] = 1
    Phy_lay2_B[3][13] = 1
    Phy_lay2_B[3][37] = 1
    Phy_lay2_B[4] = 1
    Phy_lay2_B[5] = 1
    Phy_lay2_B[6] = 1
    Phy_lay2_B[7] = 1

    phyBias_lay2_B[0:3] = 0

    Phy_lay3_A = np.zeros((6, out2), dtype=np.float32)
    Phy_lay3_B = np.zeros((6, out2), dtype=np.float32)
    phyBias_lay3_A = np.zeros((6, 1), dtype=np.float32)
    phyBias_lay3_B = np.ones((6, 1), dtype=np.float32)

    Phy_lay3_A[0][0] = 1
    Phy_lay3_A[1][1] = 1
    Phy_lay3_A[2][2] = 1
    Phy_lay3_A[3][3] = 1

    Phy_lay3_B[3][0] = 1
    Phy_lay3_B[3][3] = 1
    Phy_lay3_B[4] = 1
    Phy_lay3_B[5] = 1

    phyBias_lay3_B[0:3] = 0

    params['uncheckable_phyweightsA'] = [Phy_lay1_A, Phy_lay2_A, Phy_lay3_A]
    params['uncheckable_phyweightsB'] = [Phy_lay1_B, Phy_lay2_B, Phy_lay3_B]
    params['uncheckable_phybiasesA'] = [phyBias_lay1_A, phyBias_lay2_A, phyBias_lay3_A]
    params['uncheckable_phybiasesB'] = [phyBias_lay1_B, phyBias_lay2_B, phyBias_lay3_B]

    params['traj_len'] = 100  # must equal to batch size
    params['Xwidth'] = 6
    params['Ywidth'] = 6
    params['lYwidth'] = 6
    params['dynamics_lam'] = 1

    model = create_DeepTaylor_net(params)

    return model


def build_actor_model(input_dim, output_dim, output_activation):
    params = {}
    out1 = 10
    out2 = 8

    params['data_name'] = 'Car_test'
    params['seed'] = 10
    params['uncheckable_dist_weights'] = ['tn', 'tn', 'tn']
    params['uncheckable_output_size'] = [input_dim, out1, out2, output_dim]
    params['uncheckable_epd'] = np.array([1, 1, 0])
    params['uncheckable_act'] = ['elu', 'elu', output_activation]
    params['uncheckable_com_type1'] = ['none', 'none', 'none']
    params['uncheckable_com_type2'] = ['none', 'none', 'none']
    params['uncheckable_dist_biases'] = ['normal', 'normal', 'normal']
    params['uncheckable_num_of_layers'] = len(np.array([0, 0, 0]))

    T = 0.005

    a_input_dim = 27

    Phy_lay1_A = np.zeros((out1, a_input_dim), dtype=np.float32)
    Phy_lay1_B = np.zeros((out1, a_input_dim), dtype=np.float32)
    phyBias_lay1_A = np.zeros((out1, 1), dtype=np.float32)
    phyBias_lay1_B = np.ones((out1, 1), dtype=np.float32)

    Phy_lay1_A[0][0] = 1
    Phy_lay1_A[0][3] = T
    Phy_lay1_A[1][1] = 1
    Phy_lay1_A[1][4] = T
    Phy_lay1_A[2][2] = 1
    Phy_lay1_A[2][5] = T
    Phy_lay1_A[3][3] = 1
    Phy_lay1_A[4][4] = 1
    Phy_lay1_A[5][5] = 1

    Phy_lay1_B[3][0] = 1
    Phy_lay1_B[3][3] = 1
    Phy_lay1_B[3][6] = 1
    Phy_lay1_B[3][9] = 1
    Phy_lay1_B[3][21] = 1
    Phy_lay1_B[4] = 1
    Phy_lay1_B[5] = 1
    Phy_lay1_B[6] = 1
    Phy_lay1_B[7] = 1
    Phy_lay1_B[8] = 1
    Phy_lay1_B[9] = 1

    phyBias_lay1_B[0:3] = 0

    out1_a = 65

    Phy_lay2_A = np.zeros((out2, out1_a), dtype=np.float32)
    Phy_lay2_B = np.zeros((out2, out1_a), dtype=np.float32)
    phyBias_lay2_A = np.zeros((out2, 1), dtype=np.float32)
    phyBias_lay2_B = np.ones((out2, 1), dtype=np.float32)

    Phy_lay2_A[0][0] = 1
    Phy_lay2_A[1][1] = 1
    Phy_lay2_A[2][2] = 1
    Phy_lay2_A[3][3] = 1

    Phy_lay2_B[3][0] = 1
    Phy_lay2_B[3][3] = 1
    Phy_lay2_B[3][10] = 1
    Phy_lay2_B[3][13] = 1
    Phy_lay2_B[3][37] = 1
    Phy_lay2_B[4] = 1
    Phy_lay2_B[5] = 1
    Phy_lay2_B[6] = 1
    Phy_lay2_B[7] = 1

    phyBias_lay2_B[0:3] = 0

    Phy_lay3_A = np.zeros((6, out2), dtype=np.float32)
    Phy_lay3_B = np.zeros((6, out2), dtype=np.float32)
    phyBias_lay3_A = np.zeros((6, 1), dtype=np.float32)
    phyBias_lay3_B = np.ones((6, 1), dtype=np.float32)

    Phy_lay3_A[0][0] = 1
    Phy_lay3_A[1][1] = 1
    Phy_lay3_A[2][2] = 1
    Phy_lay3_A[3][3] = 1

    Phy_lay3_B[3][0] = 1
    Phy_lay3_B[3][3] = 1
    Phy_lay3_B[4] = 1
    Phy_lay3_B[5] = 1

    phyBias_lay3_B[0:3] = 0

    params['uncheckable_phyweightsA'] = [Phy_lay1_A, Phy_lay2_A, Phy_lay3_A]
    params['uncheckable_phyweightsB'] = [Phy_lay1_B, Phy_lay2_B, Phy_lay3_B]
    params['uncheckable_phybiasesA'] = [phyBias_lay1_A, phyBias_lay2_A, phyBias_lay3_A]
    params['uncheckable_phybiasesB'] = [phyBias_lay1_B, phyBias_lay2_B, phyBias_lay3_B]

    params['traj_len'] = 100  # must equal to batch size
    params['Xwidth'] = 6
    params['Ywidth'] = 6
    params['lYwidth'] = 6
    params['dynamics_lam'] = 1

    model = create_DeepTaylor_net(params)

    return model


def create_DeepTaylor_net(params):
    # UN_widths = exp_length(output_size=params['uncheckable_output_size'], epd=params['uncheckable_epd'])
    # UN_widths = UN_widths.astype(np.int64)  # [[10. 27.] [ 8. 65.] [ 6.  8.]]
    #
    # layer_list = []
    #
    # x = tf.keras.Input(shape=(params['uncheckable_output_size'][0]), dtype=tf.float32)
    #
    # phyweights = [params['uncheckable_phyweightsA'][i] for i in range(len(UN_widths))]
    #
    # for i in range(len(UN_widths)):
    #     layer = TaylorLayer(weights_shape=UN_widths[i],
    #                         var_name="Taylorlayer",
    #                         weight_init_dis=params['uncheckable_dist_weights'][i],
    #                         bias_init_dis=params['uncheckable_dist_biases'][i],
    #                         scale=0.1,
    #                         cp_type=params['uncheckable_com_type1'][i],
    #                         exp_order=params['uncheckable_epd'][i],
    #                         act_type=params['uncheckable_act'][i],
    #                         phyweights=params['uncheckable_phyweightsA'][i],
    #                         inx_unk=params['uncheckable_phybiasesB'][i],
    #                         inx_phy=params['uncheckable_phybiasesA'][i],
    #                         pre_weight_A=phyweights[i],
    #                         unkweights=params['uncheckable_phyweightsB'][i])
    #     layer_list.append(layer)
    #
    # output_1 = layer_list[0](x)
    # output_2 = layer_list[1](output_1)
    # output = layer_list[2](output_2)
    #
    # model = tf.keras.Model(inputs=x, outputs=output)

    model = TaylorModel(params)
    return model


# class TaylorModel(Model):
#     def __init__(self, params):
#         super(TaylorModel, self).__init__()
#         self.layer_list = []
#
#         UN_widths = exp_length(output_size=params['uncheckable_output_size'], epd=params['uncheckable_epd'])
#         UN_widths = UN_widths.astype(np.int64)  # [[10. 27.] [ 8. 65.] [ 6.  8.]]
#
#         phyweights = [params['uncheckable_phyweightsA'][i] for i in range(len(UN_widths))]
#
#         for i in range(len(UN_widths)):
#             layer = TaylorLayer(weights_shape=UN_widths[i],
#                                 var_name="Taylorlayer",
#                                 weight_init_dis=params['uncheckable_dist_weights'][i],
#                                 bias_init_dis=params['uncheckable_dist_biases'][i],
#                                 scale=0.1,
#                                 cp_type=params['uncheckable_com_type1'][i],
#                                 exp_order=params['uncheckable_epd'][i],
#                                 act_type=params['uncheckable_act'][i],
#                                 phyweights=params['uncheckable_phyweightsA'][i],
#                                 inx_unk=params['uncheckable_phybiasesB'][i],
#                                 inx_phy=params['uncheckable_phybiasesA'][i],
#                                 pre_weight_A=phyweights[i],
#                                 unkweights=params['uncheckable_phyweightsB'][i])
#             self.layer_list.append(layer)
#
#     def call(self, inputs, training=None, mask=None):
#         output_1 = self.layer_list[0](inputs)
#         output_2 = self.layer_list[1](output_1)
#         output = self.layer_list[2](output_2)
#         return output


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
