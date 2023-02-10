import copy
import time
import pickle

import tensorflow_constrained_optimization as tfco

import tensorflow as tf
import numpy as np


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


def taylor_nn(prev_layer, weights, phyweights, unkweights, biases, inx_phy, inx_unk, com_type1, com_type2, act_type,
              num_of_layers, expansion_order, name='U'):
    """Apply a NN to input from previous later

    Arguments:
        prev_layer      -- input from previous NN
        weights         -- dictionary of weights
        biases          -- dictionary of biases (uniform(-1,1) distribution, normal(0,1) distrubution, none--zeros)
        act_type        -- dictionary of activation functions (sigmoid, relu, elu, or none): user option
        num_of_layers   -- number of weight matrices or layers: user option
        expansion_order -- dictionary of Taylor expansion order: user option

    Returns:
        output of network for input from previous layer
    """

    for i in np.arange(num_of_layers):

        aa = 0.001
        bb = 100
        ##

        # Compressor One###
        if com_type1['com1%s%d' % (name, i + 1)] == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif com_type1['com1%s%d' % (name, i + 1)] == 'relu':
            prev_layer = tf.nn.relu(prev_layer)
        elif com_type1['com1%s%d' % (name, i + 1)] == 'lin':
            prev_layer = tf.multiply(prev_layer, aa) + bb
        elif com_type1['com1%s%d' % (name, i + 1)] == 'none':
            prev_layer = prev_layer

        # Compressor Two###
        if com_type2['com2%s%d' % (name, i + 1)] == 'sigmoid':
            prev_layer = tf.sigmoid(prev_layer)
        elif com_type2['com2%s%d' % (name, i + 1)] == 'none':
            prev_layer = prev_layer

        # save raw input###
        input_raw = prev_layer
        raw_input_shape = input_raw.shape

        # The expaned input via Taylor expansion is denoted by input_epd###
        input_epd = input_raw

        # Anxiliary index###
        Id = np.arange(raw_input_shape[0].value)


        # Nolinear mapping through Taylor expansion###
        for _ in range(expansion_order['E%s%d' % (name, i + 1)]):
            for j in range(raw_input_shape[0]):
                for q in range(raw_input_shape[1]):

                    x_temp = tf.multiply(input_raw[j, q], input_epd[Id[j]:(Id[raw_input_shape[0] - 1] + 1), q])
                    x_temp = tf.expand_dims(x_temp, 1)
                    if q == 0:
                        tem_temp = x_temp
                    else:
                        tem_temp = tf.concat((tem_temp, x_temp), 1)
                Id[j] = input_epd.shape[0]
                input_epd = tf.concat((input_epd, tem_temp), 0)

        pre_bias = tf.multiply(biases['b%s%d' % (name, i + 1)], inx_unk['b%s%d' % (name, i + 1)])

        pre_weight_A = phyweights['W%s%d' % (name, i + 1)]
        pre_weight_B = tf.multiply(weights['W%s%d' % (name, i + 1)], unkweights['W%s%d' % (name, i + 1)])
        prev_layer_A = tf.matmul(pre_weight_A, input_epd) + inx_phy['b%s%d' % (name, i + 1)]
        prev_layer_B = tf.matmul(pre_weight_B, input_epd) + pre_bias

        if act_type['act%s%d' % (name, i + 1)] == 'sigmoid':
            prev_layer_B = pysigmoid(prev_layer_B, inx_unk['b%s%d' % (name, i + 1)])
        elif act_type['act%s%d' % (name, i + 1)] == 'relu':
            prev_layer_B = pyrelu(prev_layer_B, inx_unk['bk%s%d' % (name, i + 1)])
        elif act_type['act%s%d' % (name, i + 1)] == 'elu':
            prev_layer_B = pyelu(prev_layer_B, inx_unk['b%s%d' % (name, i + 1)])
        elif act_type['act%s%d' % (name, i + 1)] == 'none':
            prev_layer_B = prev_layer_B

        prev_layer = prev_layer_A + prev_layer_B
    return prev_layer



def initilization(widths, comT1, comT2, act, epd, dist_weights, dist_biases, phyweightsA, phyweightsB, phybiasesA,
                  phybiasesB,
                  scale, name='U'):
    """Create a decoder network: a dictionaries of weights, biases, activation function and expansion_order

    Arguments:
        widths       -- array or list of widths for layers of network
        comT1        -- list of compressor function 1
        comT2        -- list of compressor function 2
        act          -- list of string for activation functions
        epd          -- array of expansion order
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases  -- array or list of strings for distributions of bias vectors
        scale        -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        name         -- string for prefix on weight matrices (default 'D' for decoder)

    Returns:
        weights         -- dictionary of weights
        com_type1       -- dictionary of compressor function 1
        com_type2       -- dictionary of compressor function 2
        biases          -- dictionary of biases
        act_type        -- dictionary of activation functions
        expansion_order -- dictionary of expansion order
    """

    weights = dict()
    biases = dict()
    act_type = dict()
    expansion_order = dict()
    com_type1 = dict()
    com_type2 = dict()
    phyweights = dict()
    unkweights = dict()
    inx_phy = dict()
    inx_unk = dict()

    for i in np.arange(len(widths)):
        ind = i + 1
        weights['W%s%d' % (name, ind)] = weight_variable(widths[i], var_name='W%s%d' % (name, ind),
                                                         distribution=dist_weights[ind - 1], scale=scale)
        biases['b%s%d' % (name, ind)] = bias_variable([widths[i][0], 1], var_name='b%s%d' % (name, ind),
                                                      distribution=dist_biases[ind - 1])
        act_type['act%s%d' % (name, ind)] = act[i]
        expansion_order['E%s%d' % (name, ind)] = epd[i]

        com_type1['com1%s%d' % (name, ind)] = comT1[i]
        com_type2['com2%s%d' % (name, ind)] = comT2[i]

        phyweights['W%s%d' % (name, ind)] = phyweightsA[i]
        unkweights['W%s%d' % (name, ind)] = phyweightsB[i]
        inx_phy['b%s%d' % (name, ind)] = phybiasesA[i]
        inx_unk['b%s%d' % (name, ind)] = phybiasesB[i]

    return weights, com_type1, com_type2, biases, act_type, expansion_order, phyweights, unkweights, inx_phy, inx_unk


def weight_variable(shape, var_name, distribution, scale=0.1):
    """Create a variable for a weight matrix.

    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization (default 'tn')
        scale -- (for tn distribution): standard deviation of normal distribution before truncation (default 0.1)

    Returns:
        a TensorFlow variable for a weight matrix

    Raises ValueError if distribution is filename but shape of data in file does not match input shape
    """

    if distribution == 'tn':
        initial = tf.random.truncated_normal(shape, stddev=scale, dtype=tf.float32)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float32)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name, distribution):
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
    return tf.Variable(initial, name=var_name)


def exp_length(output_size, epd):
    """Generate shape list of expanded layer.

    Arguments:
        output_size -- [input dimention, layer output size list]
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


def create_DeepTaylor_net(params):
    """Create a DeepTaylor that consists of uncheckable and check models in order

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        x       --  placeholder for input
        y       --  output of uncheckable model, e.g., y = [x(k+1); u(k+1)]
        z       --  output of checkable model, e.g., z = f(y(k+m))
        ly      --  labels of uncheckable model
        lz      --  labels of checkable model
        weights --  dictionary of weights
        biases  --  dictionary of biases
    """
    x = tf.keras.Input(shape=(params['Xwidth'], params['traj_len']), dtype=tf.float32)
    ly = tf.keras.Input(shape=(params['lYwidth'], params['traj_len']), dtype=tf.float32)

    UN_widths = exp_length(output_size=params['uncheckable_output_size'], epd=params['uncheckable_epd'])
    UN_widths = UN_widths.astype(np.int64)

    weights, com_type1, com_type2, biases, act_type, expansion_order, phyweights, unkweights, inx_phy, inx_unk = initilization(
        widths=UN_widths, comT1=params['uncheckable_com_type1'], comT2=params['uncheckable_com_type2'],
        act=params['uncheckable_act'], epd=params['uncheckable_epd'], dist_weights=params['uncheckable_dist_weights'],
        dist_biases=params['uncheckable_dist_biases'], phyweightsA=params['uncheckable_phyweightsA'],
        phyweightsB=params['uncheckable_phyweightsB'], phybiasesA=params['uncheckable_phybiasesA'],
        phybiasesB=params['uncheckable_phybiasesB'], scale=0.1, name='U')

    # y = taylor_nn(prev_layer=x, weights=weights, phyweights=phyweights, unkweights=unkweights, biases=biases,
    #               inx_phy=inx_phy, inx_unk=inx_unk, com_type1=com_type1, com_type2=com_type2, act_type=act_type,
    #               num_of_layers=params['uncheckable_num_of_layers'], expansion_order=expansion_order, name='U')

    y = taylor_nn(prev_layer=x, expansion_order=1, num_of_layers=1)

    return x, y, ly, weights, biases


class ExampleProblem(tfco.ConstrainedMinimizationProblem):

    def __init__(self, y, ly, params, upper_bound):
        #         self._x = x
        self._y = y
        self._ly = ly
        self._params = params
        self._upper_bound = upper_bound

    # Note: the number of constaint must be element-wise, cannot be vector- or matrix-wise!!!!!!!!!!!!!
    @property
    def num_constraints(self):
        return 1

    def objective(self):
        return define_loss(self._y, self._ly, self._params, self._trainable_var)

    def constraints(self):
        return self._z[0][0] - self._upper_bound


# Define loss funcions for training
def define_loss(y, ly, params, trainable_var):
    """Define the (unregularized) loss functions for the training.

    Arguments:
       import os
import time

import numpy as np
import tensorflow as tf

import tensorflow_constrained_optimization as tfco

import TaylorNN as tnn
print(tf.__version__) x  -- placeholder for input: x(k)
        y  -- output of uncheckable model, e.g., y = [x(k+1); u(k+1)]:
        z  -- output of checkable model for prediction: e.g., z = f(y(k+m))

        ly -- label of y
        lz -- lable of z

    Returns:
        loss1 -- supervised dynamics loss
        loss2 -- supervised future loss
        loss --  sum of above two losses

    Side effects:
        None
    """

    # Embedding dynamics into uncheck model, learning via prediction
    loss1 = params['dynamics_lam'] * tf.reduce_mean(tf.square(y - ly))
    loss = loss1
    return loss


# ====================================================================================
def save_files(sess, csv_path, params, weights, biases):
    """Save error files, weights, biases, and parameters.

    Arguments:
        sess -- TensorFlow session
        csv_path -- string for path to save error file as csv
        train_val_error -- table of training and validation errors
        params -- dictionary of parameters for experiment
        weights -- dictionary of weights for all networks
        biases -- dictionary of biases for all networks

    Returns:
        None (but side effect of saving files and updating params dict.)

    Side effects:
        Save train_val_error, each weight W, each bias b, and params dict to file.
        Update params dict: minTrain, minTest, minRegTrain, minRegTest
    """

    for key, value in weights.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    for key, value in biases.items():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')

    save_params(params)


def save_params(params):
    """Save parameter dictionary to file.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None

    Side effects:
        Saves params dict to pkl file
    """
    with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


# =======================================================================================================
def try_exp(params):
    """Initilize Taylor NN"""
    x, y, ly, weights, biases = create_DeepTaylor_net(params)

    '''return a list of all the trainable variables'''

    trainable_var = tf.trainable_variables()

    loss = define_loss(y, ly, params, trainable_var)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)

    train_op = optimizer.minimize(loss, var_list=trainable_var)

    sess = tf.Session()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')

    count = 0
    best_error = 10000

    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])

    valx = np.loadtxt(('./trainingdata1/%s_valX.csv' % (params['data_name'])), delimiter=',',
                      dtype=np.float32)
    valy = np.loadtxt(('./trainingdata1/%s_valY.csv' % (params['data_name'])), delimiter=',',
                      dtype=np.float32)

    for f in range(params['number_of_data files_for_training'] * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % params['number_of_data files_for_training']) + 1  # 1...data_train_len

        if (params['number_of_data files_for_training'] > 1) or (f == 0):  # don't keep reloading data if always same;

            data_train_x = np.loadtxt(('./trainingdata1/%s_X%d.csv' % (params['data_name'], file_num)), delimiter=',', dtype=np.float32)
            data_train_ly = np.loadtxt(('./trainingdata1/%s_Y%d.csv' % (params['data_name'], file_num)), delimiter=',', dtype=np.float32)
            # Total len in 1 training file
            total_length = data_train_x.shape[0]
            num_batches = int(np.floor(total_length / params['batch_size']))

        ind = np.arange(total_length)
        np.random.shuffle(ind)

        data_train_x = data_train_x[ind, :]
        data_train_ly = data_train_ly[ind, :]
        valx = valx[ind, :]
        valy = valy[ind, :]

        for step in range(params['num_steps_per_batch'] * num_batches):
            if params['batch_size'] < data_train_x.shape[0]:
                offset = (step * params['batch_size']) % (total_length - params['batch_size'])
            else:
                offset = 0
            batch_data_train_x = data_train_x[offset:(offset + params['batch_size']), :]
            batch_data_train_ly = data_train_ly[offset:(offset + params['batch_size']), :]
            batch_data_valx = valx[offset:(offset + params['batch_size']), :]
            batch_data_valy = valy[offset:(offset + params['batch_size']), :]

            feed_dict_train = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
            feed_dict_train_loss = {x: np.transpose(batch_data_train_x), ly: np.transpose(batch_data_train_ly)}
            feed_dict_val = {x: np.transpose(batch_data_valx), ly: np.transpose(batch_data_valy)}

            sess.run(train_op, feed_dict=feed_dict_train)

            if step % params['loops for val'] == 0:
                train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                val_error = sess.run(loss, feed_dict=feed_dict_val)

                Error = [train_error, val_error]

                # print(train_error)
                print(Error)

                count = count + 1

                save_files(sess, csv_path, params, weights, biases)

    saver.restore(sess, params['model_path'])
    save_files(sess, csv_path, params, weights, biases)
    tf.reset_default_graph()


def main_exp(params):
    tf.random.set_seed(params['seed'])
    np.random.seed(params['seed'])
    try_exp(params)


if __name__ == '__main__':

    params = {}
    input_dim = 6
    out1 = 10
    out2 = 8

    params['data_name'] = 'Car_test'
    params['seed'] = 10
    params['uncheckable_dist_weights'] = ['tn', 'tn', 'tn']
    params['uncheckable_output_size'] = [input_dim, out1, out2, 6]
    params['uncheckable_epd'] = np.array([1, 1, 0])
    params['uncheckable_act'] = ['elu', 'elu', 'none']
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

    params['traj_len'] = 200  # must equate to batch size
    params['Xwidth'] = 6
    params['Ywidth'] = 6
    params['lYwidth'] = 6
    params['dynamics_lam'] = 1

    params['exp_name'] = 'exp'
    params['folder_name'] = 'EmPhyCas10'
    params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])
    params['number_of_data files_for_training'] = 2  # 2
    params['num_passes_per_file'] = 1000 * 8 * 1000
    params['batch_size'] = 200  # todo 50
    params['num_steps_per_batch'] = 2
    params['loops for val'] = 1

    for count in range(1):
        main_exp(copy.deepcopy(params))
    print('Done Done Done')
