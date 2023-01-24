import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model


def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float16)
    dense1 = Dense(256, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(64, activation='relu', name=name + 'dense3')(dense2)
    output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
    model = Model(inputs=input, outputs=output, name=name)
    return model

