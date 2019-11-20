import logging
from typing import List, Dict

import click
import numpy as np
import pandas as pd

from lyner._main import rnax
from lyner.click_extras import DICT, pass_pipe, arggist, Pipe

LOGGER = logging.getLogger("lyner")
logging.basicConfig(level=logging.NOTSET)


@rnax.command()
@click.option("--layer-config", "-l", type=DICT, multiple=True, default=None)
@click.option("--from-file", "-f", type=click.Path(exists=True, dir_okay=False))
@click.option("--store-model", "-s", type=click.Path())
@click.option("--loss",
              type=click.Choice(['kld', 'mae', 'mape', 'mse', 'msle', 'binary_crossentropy', 'categorical_crossentropy',
                                 'categorical_hinge', 'cosine', 'cosine_proximity', 'hinge', 'logcosh', 'poisson',
                                 'sparse_categorical_crossentropy', 'squared_hinge']), default='mse')
@click.option("--optimiser", "-o",
              type=click.Choice(['adadelta', 'adagrad', 'adam', 'adamax', 'nadam', 'rmsprop', 'sgd']),
              default='adam')
@click.option("--epochs", "-e", type=click.INT, default=500)
@click.option("--batch-size", "-b", type=click.INT, default=5)
@click.option("--shuffle", "-s", type=click.BOOL, default=True)
@click.option("--validation-split", "-v", type=click.FloatRange(0, 1), default=0.1)
@click.option("--adjust-weights", "-w", type=click.FLOAT, default=None)
@click.option("--mode", "-m", type=click.Choice(['discard', 'nodes', 'weights']), default='nodes')
@pass_pipe
@arggist
def autoencode(pipe: Pipe,
               layer_config: List[Dict],
               from_file: str,
               store_model: str,
               loss: str,
               optimiser: str,
               epochs: int,
               batch_size: int,
               shuffle: bool,
               validation_split: float,
               adjust_weights: float,
               mode: str):
    """Build and train an autoencoder."""
    import keras
    from keras import regularizers, Sequential, Input, Model
    from keras.callbacks import EarlyStopping, TensorBoard
    from keras.engine import InputLayer
    from keras.engine.saving import model_from_yaml, model_from_json
    from keras.layers import Dense
    from numpy.random.mtrand import seed
    from tensorflow import set_random_seed
    from lyner.keras_extras import SignalHandler
    seed(1)
    set_random_seed(2)
    matrix = pipe.matrix.copy()
    if matrix.isnull().values.any():
        LOGGER.warning("Dropping rows containing nan values")
        matrix.dropna(how='any', inplace=True)

    def parse_layout(layer_conf):
        get_layer_type = lambda t: getattr(keras.layers, t, None)
        regdict = {'l1_l2': regularizers.l1_l2, 'l1': regularizers.l1, 'l2': regularizers.l2}
        lc = layer_conf.copy()
        layer_type = lc.get('type', None)
        if layer_type:
            lc['type'] = get_layer_type(layer_type)

        # TODO parse regularizers
        kernel_reg_type = lc.get('kernel_regularizer', None)
        if kernel_reg_type:
            if '(' in kernel_reg_type and ')' in kernel_reg_type:
                params = kernel_reg_type[kernel_reg_type.index('(') + 1:kernel_reg_type.index(')')]
                if '+' in params:
                    params = params.split('+')
                else:
                    params = [params]
                params = [float(p) for p in params]
                kernel_reg_type = kernel_reg_type[:kernel_reg_type.index('(')]
            lc['kernel_regularizer'] = regdict[kernel_reg_type](*params)
        return lc.pop('type'), int(lc.pop('n')), lc

    layout = [parse_layout(layer_conf) for layer_conf in layer_config]
    labels = matrix.columns.values.tolist()
    data = matrix.values
    shape = (data.shape[0],)
    data = data.transpose()
    if layout:
        encoding_dim = layout[-1][1]
        encoder = Sequential(name="encoder")
        encoder.add(InputLayer(shape, name="encoder_input"))
        for layer_num, (Layer, n_nodes, extra_args) in enumerate(layout):
            encoder.add(Layer(n_nodes, name=f"encoder_{layer_num}_{n_nodes}", **extra_args))
            # kernel_regularizer=regularizers.l1_l2(0.001, 0.001),
            # kernel_regularizer=regularizers.l1(0.0001),

        decoder = Sequential(name="decoder")
        decoder.add(InputLayer((encoding_dim,), name="decoder_input"))
        for layer_num, (Layer, n_nodes, _) in enumerate(layout[::-1][1:]):
            decoder.add(Layer(n_nodes, name=f"decoder_{layer_num}_{n_nodes}"))
        decoder.add(Dense(shape[0], activation='linear', name="decoder_output"))

        input_layer = Input(shape=shape, name="autoencoder_input")
        encode_layer = encoder(input_layer)
        decode_layer = decoder(encode_layer)

        autoencoder = Model(input_layer, decode_layer)
        if store_model:
            if store_model.endswith('.yaml'):
                model_string = autoencoder.to_yaml()
            elif store_model.endswith('.json'):
                model_string = autoencoder.to_json()
            else:
                model_string = autoencoder.to_yaml()
            with open(store_model, 'wt') as writer:
                writer.write(model_string)
    elif from_file:
        with open(from_file, 'rt') as reader:
            model_string = '\n'.join(reader.readlines())
        if from_file.endswith('.yaml'):
            autoencoder = model_from_yaml(model_string)
        elif from_file.endswith('.json'):
            autoencoder = model_from_json(model_string)
        # TODO set encoder and decoder correctly
    else:
        raise ValueError("No model specified. Use either of --layer-config or --from-file.")
    # from pprint import pprint
    # pprint(autoencoder.get_config())
    autoencoder.compile(optimizer=optimiser, loss=loss, metrics=['mse'], )

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=50)

    sh = SignalHandler()
    autoencoder.fit(np.vsplit(data, 1), np.vsplit(data, 1),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder'), sh, early_stopping],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    shuffle=shuffle
                    )
    sh.uninit()

    class Autoencoder:
        def __init__(self, encoder=None, decoder=None):
            self._encoder = encoder
            self._decoder = decoder

        def inverse_transform(self, data):
            return self._decoder.predict(data).transpose()

        def transform(self, data):
            return self._encoder.predict(data).transpose()

    pipe.decomposition = Autoencoder(encoder, decoder)

    encoded_data = pipe.decomposition.transform(data)
    decoded_data = pipe.decomposition.inverse_transform(encoded_data.T)
    pre_error = ((data.T - decoded_data) ** 2).mean(axis=None)
    print(f"MSE: {pre_error}")

    pipe._index = pipe.matrix.index
    pipe._columns = pipe.matrix.columns
    if adjust_weights:
        quant = float(adjust_weights)
        for i, layer in enumerate(encoder.layers):
            W, b = layer.get_weights()
            low, median, high = np.quantile(W.flatten(), [quant, 0.5, 1 - quant])
            W_low = W * (W < low)
            W_high = W * (W > high)
            selected_weights = W_low + W_high
            # oplot([Histogram(x=W.flatten()), Histogram(x=W[W < low].flatten()), Histogram(x=W[W > high].flatten())])
            layer.set_weights([selected_weights, b])
            break
        encoded_data = pipe.decomposition.transform(data)
        decoded_data = pipe.decomposition.inverse_transform(encoded_data.T)
        post_error = ((data.T - decoded_data) ** 2).mean(axis=None)
        print(f"MSE: {post_error}")
    if 'weights' == mode:
        layer = 0
        layer_weights = encoder.layers[layer].get_weights()
        layer = encoder.layers[layer]
        if len(layer_weights) == 0:
            layer_weights = encoder.layers[0].get_weights()
        if len(layer_weights) >= 2:
            layer_weights = layer_weights[:-1]  # last one is bias
        new_data = layer_weights[0]
        index = [f'Weight_{i}' for i in range(new_data.shape[0])]
        num_nodes = new_data.shape[1]
        columns = [f"{layer.name}_{i}" for i in range(num_nodes)]
    elif 'nodes' == mode:
        new_data = encoder.predict(np.vsplit(data, 1)).transpose()
        columns = labels
        index = [f"{mode}_{i}" for i in range(encoding_dim)]
    elif 'discard' == mode:
        W, b = encoder.layers[0].get_weights()
        W = np.sum(np.abs(W), axis=1)
        W[W != 0] = 1
        print(f"Kept {np.sum(W)} weights")
        v: np.array = pipe.matrix.values
        new_data = (v.T * W).T
        columns = pipe.matrix.columns
        index = pipe.matrix.index
    else:
        raise ValueError(f"Unknown mode {mode}")
    pipe.matrix = pd.DataFrame(data=new_data,
                               columns=columns,
                               index=index,
                               )
    return
