# model_nn.py

import tensorflow as tf
from tensorflow.keras import layers
from kerastuner import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(layers.Dense(hp.Int('input_units', 32, 512, step=32), activation='relu', input_shape=(X_train.shape[1],)))
        for i in range(hp.Int('n_layers', 1, 3)):
            model.add(layers.Dense(hp.Int(f'dense_{i}_units', 32, 512, step=32), activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    tuner = RandomSearch(
        lambda hp: build_model(hp),
        objective='val_loss', max_trials=5, executions_per_trial=1,
        directory='keras_tuning', project_name='bond_nn'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, verbose=0)
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
    model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0, shuffle=False)
    y_pred = model.predict(X_test).flatten()
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), y_test, y_pred, model
