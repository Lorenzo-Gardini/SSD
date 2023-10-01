# keras
from keras.layers import (Dense, Conv1D, Input, Flatten, MaxPool1D, LSTM, SimpleRNN, GRU)
from keras import Sequential
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt


# -------------------- UTILITY --------------------

def compile_model(model):
    model.compile(loss='mse', metrics=[RootMeanSquaredError(name='rmse')], optimizer='adam')
    return model


def keras_fit(model, train_x, train_y, validation_x, validation_y, epoch_count=100, patience=5, batch_size=32,
              history_plot=True):
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(x=train_x,
                        y=train_y,
                        validation_data=(validation_x, validation_y),
                        epochs=epoch_count,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=True)
    # plot history
    if history_plot:
        plot_history(history)
    return model, history


def plot_keras_model(model):
    print(model.summary())


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'])
    plt.title('Loss')
    plt.show()


# -------------------- BUILDER FUNCTIONS --------------------

def build_deep_mlp(features, hidden_layers=[128]):
    sequential = Sequential()

    sequential.add(Input(shape=features, name='input'))

    for hidden in hidden_layers:
        sequential.add(Dense(hidden, activation='relu'))

    sequential.add(Dense(1, name='ouput'))

    return compile_model(sequential)


def build_deep_cnn(features):
    sequential = Sequential(
        [
            Input(shape=(features, 1), name='input'),
            Conv1D(filters=32, kernel_size=3, strides=4, activation='relu', padding='same', name='c1'),
            MaxPool1D(pool_size=2, padding='same'),
            Conv1D(filters=64, kernel_size=2, strides=2, activation='relu', padding='same', name='c2'),
            MaxPool1D(pool_size=2, padding='same'),
            Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same', name='c3'),
            Flatten(),
            Dense(256, activation='relu', name='fc'),
            Dense(1, activation='linear', name='output')
        ])
    return compile_model(sequential)


def build_deep_rnn(features, units_per_rnn_layer=[128, 128]):
    sequential = Sequential()
    sequential.add(Input(shape=(features, 1)))

    for unit in units_per_rnn_layer[:-1]:
        sequential.add(SimpleRNN(unit, return_sequences=True))

    sequential.add(SimpleRNN(units_per_rnn_layer[-1], return_sequences=False))

    sequential.add(Dense(1))

    return compile_model(sequential)


def build_deep_gru(features, units_per_rnn_layer=[128, 128]):
    sequential = Sequential()
    sequential.add(Input(shape=(features, 1)))

    for unit in units_per_rnn_layer[:-1]:
        sequential.add(GRU(unit, return_sequences=True))

    sequential.add(GRU(units_per_rnn_layer[-1], return_sequences=False))

    sequential.add(Dense(1))

    return compile_model(sequential)


def build_lstm(features, units_per_rnn_layer=[128, 128]):
    sequential = Sequential()
    sequential.add(Input(shape=(features, 1)))

    for unit in units_per_rnn_layer[:-1]:
        sequential.add(LSTM(unit, return_sequences=True))

    sequential.add(LSTM(units_per_rnn_layer[-1], return_sequences=False))

    sequential.add(Dense(1))

    return compile_model(sequential)
