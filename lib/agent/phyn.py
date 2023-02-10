import tensorflow as tf
import numpy as np


def taylor_nn(prev_layer, expansion_order=1, num_of_layers=1, name='U'):
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

        # save raw input###
        input_raw = prev_layer
        raw_input_shape = input_raw.shape

        input_epd = input_raw

        Id = np.arange(raw_input_shape[0])

        for _ in range(expansion_order):
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

    return input_epd


if __name__ == '__main__':
    prev_layer = np.array([[1], [2]])
    input_epd = taylor_nn(prev_layer, expansion_order=1)
    print(input_epd)
