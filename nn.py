import tensorflow as tf


def create_mlp(dims, activation='relu', final_activation='linear'):
    assert len(dims) >= 2
    if isinstance(activation, str):
        activation = [activation] * (len(dims) - 2)
    if isinstance(activation, list):
        assert len(activation) == len(dims) - 2

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(dims[0],)))
    for dim, active in zip(dims[1:-1], activation):
        model.add(tf.keras.layers.Dense(dim, activation=active))
    model.add(tf.keras.layers.Dense(dims[-1], activation=final_activation))

    model.summary()

    return model
