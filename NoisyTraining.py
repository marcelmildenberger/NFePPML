import json
import os
import time

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout, Input
from keras.optimizers import Adam ,RMSprop
from keras.utils import to_categorical, plot_model
from keras import  backend as K
from NoisyDense import NoisyDense
import tensorflow as tf
import scipy.optimize
from keras.datasets import mnist
# import dataset


RESULTS_DIR = "experiment_results"
NUM_ATTACK_SAMPLES = 100
NUM_IMAGES_TO_SAVE = 10


def _prepare_run_directory(epsilon):
    epsilon_str = format(epsilon, "g")
    run_dir = os.path.join(RESULTS_DIR, f"experiment_epsilon_{epsilon_str}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_png(image_array, output_path):
    tensor = tf.convert_to_tensor(image_array.reshape(28, 28, 1), dtype=tf.uint8)
    png_bytes = tf.image.encode_png(tensor)
    tf.io.write_file(output_path, png_bytes)


def experiment(epsilon=1.0):
    start_time = time.time()
    # load dataset
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # compute the number of labels
    num_labels = len(np.unique(y_train))
    # convert to one-hot vector
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    # image dimensions (assumed square)
    image_size = x_train.shape[1]
    input_size = image_size * image_size


    # resize (flatten) and normalize
    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train.astype('float32') / 255
    x_test = np.reshape(x_test, [-1, input_size])
    x_test = x_test.astype('float32') / 255


    # network parameters
    batch_size = 128
    hidden_units = 256
    dropout = 0.45


    # model is a 3-layer MLP with ReLU and dropout after each layer
    model = Sequential()
    model.add(Input(shape=(input_size,)))
    model.add(NoisyDense(units=hidden_units, epsilon=epsilon, delta=1e-5, clip_norm=1.0, spectral_norm_cap=1.0, name="noisy_dense"))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])


    model.fit(x_train, y_train, epochs=20, batch_size=batch_size)


    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    run_dir = _prepare_run_directory(epsilon)
    mses = []
    saved_image_paths = []

    # ### Attack
    # Build a submodel that ends at  NoisyDense layer => "attacking" the layer
    noisy_layer = model.get_layer("noisy_dense")
    w, bias_np = noisy_layer.get_weights() 
    layer_model = tf.keras.Model(inputs=model.inputs, outputs=noisy_layer.output)
    x = x_train[:NUM_ATTACK_SAMPLES]
    z = layer_model.predict(x)
    n_vars = w.shape[0]
    bounds = [(0.0, 1.0)] * n_vars 
    A_eq = w.T
    c = np.zeros(n_vars, dtype=np.float64)
    
    image_dir = os.path.join(run_dir, "reconstructed_images")

    for i in range(NUM_ATTACK_SAMPLES):
        z_i = z[i].astype(np.float64)
        b_i = bias_np[i].astype(np.float64)
        b_eq = z_i - b_i
        report = scipy.optimize.linprog(c, A_ub = None, b_ub = None, A_eq = A_eq, b_eq = b_eq, bounds=bounds, method = "interior-point")
        mse = ((x[i] - report['x']) ** 2).mean()
        mses.append(float(mse))
        img = report['x'].reshape(28, 28) * 255.0
        img_uint8 = img.astype(np.uint8)
        if len(saved_image_paths) < NUM_IMAGES_TO_SAVE:
            image_name = f"reconstruction_{len(saved_image_paths) + 1}.png"
            image_path = os.path.join(image_dir, image_name)
            _save_png(img_uint8, image_path)
            saved_image_paths.append(image_name)

    runtime_seconds = time.time() - start_time
    metrics = {
        "epsilon": float(epsilon),
        "accuracy": float(acc),
        "loss": float(loss),
        "runtime_seconds": runtime_seconds,
        "num_attack_samples": NUM_ATTACK_SAMPLES,
        "average_mse_reconstruction": float(np.mean(mses)),
        "mse_reconstructions": mses,
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"Saved experiment results to {run_dir}")

def start_experiments():
    for epsilon in [1000, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8, 409.6 , 819.2, 1638.4, None]:
        print(f"Running experiment with epsilon={epsilon}")
        experiment(epsilon=epsilon)

if __name__ == "__main__":
    start_experiments()



