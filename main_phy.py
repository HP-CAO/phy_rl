import tensorflow as tf
import copy
import pickle
from lib.agent.network import TaylorLayer
import numpy as np

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


def create_DeepTaylor_net(params):
    UN_widths = exp_length(output_size=params['uncheckable_output_size'], epd=params['uncheckable_epd'])
    UN_widths = UN_widths.astype(np.int64)  # [[10. 27.] [ 8. 65.] [ 6.  8.]]

    layer_list = []

    x = tf.keras.Input(shape=(params['Xwidth'], params['traj_len']), dtype=tf.float32)

    phyweights = [params['uncheckable_phyweightsA'][i] for i in range(len(UN_widths))]

    for i in range(len(UN_widths)):
        layer = TaylorLayer(weights_shape=UN_widths[i],
                            var_name="Taylorlayer",
                            weight_init_dis=params['uncheckable_dist_weights'][i],
                            bias_init_dis=params['uncheckable_dist_biases'][i],
                            scale=0.1,
                            cp_type=params['uncheckable_com_type1'][i],
                            exp_order=params['uncheckable_epd'][i],
                            act_type=params['uncheckable_act'][i],
                            phyweights=params['uncheckable_phyweightsA'][i],
                            inx_unk=params['uncheckable_phybiasesB'][i],
                            inx_phy=params['uncheckable_phybiasesA'][i],
                            pre_weight_A=phyweights[i],
                            unkweights=params['uncheckable_phyweightsB'][i])
        layer_list.append(layer)

    output_1 = layer_list[0](x)
    output_2 = layer_list[1](output_1)
    output = layer_list[2](output_2)

    model = tf.keras.Model(inputs=x, outputs=output)
    return model


def compute_loss(y, ly, params):
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

    model = create_DeepTaylor_net(params)
    model.summary()

    '''return a list of all the trainable variables'''

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)

    valx = np.loadtxt(('./data/%s_valX.csv' % (params['data_name'])), delimiter=',', dtype=np.float32)
    valy = np.loadtxt(('./data/%s_valY.csv' % (params['data_name'])), delimiter=',', dtype=np.float32)

    training_log_writer = tf.summary.create_file_writer('logs/phy/training')
    val_log_writer = tf.summary.create_file_writer('logs/phy/val')

    total_length = None
    data_train_x = None
    data_train_ly = None
    num_batches = None
    counter = 0

    for f in range(params['number_of_data files_for_training'] * params['num_passes_per_file']):

        file_num = (f % params['number_of_data files_for_training']) + 1  # 1...data_train_len

        if (params['number_of_data files_for_training'] > 1) or (f == 0):  # don't keep reloading data if always same;

            data_train_x = np.loadtxt(('./data/%s_X%d.csv' % (params['data_name'], file_num)), delimiter=',',
                                      dtype=np.float32)
            data_train_ly = np.loadtxt(('./data/%s_Y%d.csv' % (params['data_name'], file_num)), delimiter=',',
                                       dtype=np.float32)

            total_length = data_train_x.shape[0]
            num_batches = int(np.floor(total_length / params['batch_size']))

        ind = np.arange(total_length)
        np.random.shuffle(ind)

        data_train_x = data_train_x[ind, :]
        data_train_ly = data_train_ly[ind, :]
        valx = valx[ind, :]
        valy = valy[ind, :]

        for step in range(params['num_steps_per_batch'] * num_batches):
            counter += 1
            if params['batch_size'] < data_train_x.shape[0]:
                offset = (step * params['batch_size']) % (total_length - params['batch_size'])
            else:
                offset = 0
            batch_data_train_x = data_train_x[offset:(offset + params['batch_size']), :]
            batch_data_train_ly = data_train_ly[offset:(offset + params['batch_size']), :]

            batch_data_valx = valx[offset:(offset + params['batch_size']), :]
            batch_data_valy = valy[offset:(offset + params['batch_size']), :]

            with tf.GradientTape() as tape:
                y_predict = model(np.expand_dims(np.transpose(batch_data_train_x), axis=0), training=True)
                loss_train = compute_loss(y_predict, np.expand_dims(np.transpose(batch_data_train_ly), axis=0), params)
                print(loss_train)
            gradients = tape.gradient(loss_train, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            with training_log_writer.as_default():
                tf.summary.scalar('train_eval/loss', loss_train, counter)

            if counter % 20 == 0:
                y_predict_val = model(np.expand_dims(np.transpose(batch_data_valx), axis=0), training=False)
                loss_val = compute_loss(y_predict_val, np.expand_dims(np.transpose(batch_data_valy), axis=0), params)
                with val_log_writer.as_default():
                    tf.summary.scalar('train_eval/loss', loss_val, counter)

    # todo add save models


def main_exp(params):
    tf.random.set_seed(params['seed'])
    np.random.seed(params['seed'])
    try_exp(params)


if __name__ == '__main__':

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        exit("GPU allocated failed")

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

    params['traj_len'] = 100  # must equal to batch size
    params['Xwidth'] = 6
    params['Ywidth'] = 6
    params['lYwidth'] = 6
    params['dynamics_lam'] = 1

    params['exp_name'] = 'exp'
    params['folder_name'] = 'EmPhyCas10'
    params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], params['exp_name'])
    params['number_of_data files_for_training'] = 2
    params['num_passes_per_file'] = 1000 * 8 * 1000
    params['batch_size'] = 100
    params['num_steps_per_batch'] = 2
    params['loops for val'] = 1

    for count in range(1):
        main_exp(copy.deepcopy(params))
    print('Done Done Done')
