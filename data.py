import cv2
import numpy as np
import h5py

def load(X, y, image_size=128):
  image = X; _mask_image = y

  image = cv2.resize(image, (image_size, image_size))
  mask = np.zeros((image_size, image_size, 1))

  _mask_image = cv2.resize(_mask_image, (image_size, image_size)) #128x128
  _mask_image = np.expand_dims(_mask_image, axis=-1)
  mask = np.maximum(mask, _mask_image)

  ## Normalizaing
  image = image/255.0
  mask = mask/255.0

  return image, mask

def read_data(DATA_PATH, image_size):
    X_train = h5py.File(DATA_PATH+'X_train.hdf5', 'r')
    X_tr = X_train.get('X_train')[:]
    y_train = h5py.File(DATA_PATH+'y_train.hdf5', 'r')
    y_tr = y_train.get('y_train')[:]

    X_val = h5py.File(DATA_PATH+'X_test.hdf5', 'r')
    X_v = X_val.get('X_test')[:]
    y_val = h5py.File(DATA_PATH+'y_test.hdf5', 'r')
    y_v = y_val.get('y_test')[:]

    features, features2 = [], []
    labels, labels2 = [], []

    for i in range(len(X_tr)):
        feature, label = load(X_tr[i], y_tr[i],image_size)
        features.append(feature.reshape(feature.shape[0], feature.shape[1], 1))
        labels.append(label.reshape(label.shape[0], label.shape[1], 1))

    for j in range(len(X_v)):
        feature, label = load(X_v[j], y_v[j],image_size)
        features2.append(feature.reshape(feature.shape[0], feature.shape[1], 1))
        labels2.append(label.reshape(label.shape[0], label.shape[1], 1))

    return features, labels, features2, labels2
