import numpy as np
import tensorflow as tf
from tensorflow import keras

class Generator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=8):
        self.x, self.y = X, y
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x)/float(self.batch_size)))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [self.x[i] for i in inds]
        batch_y = [self.y[j] for j in inds]

        return np.asarray(batch_x), np.asarray(batch_y)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
