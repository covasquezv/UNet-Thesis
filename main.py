import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#import keras
#sys.modules['keras'] = keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data import *
from model import *
import my_callback
import generator

#===============================================================================
## Custom metric
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    #print(type((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
#def dice_coef_loss(y_true, y_pred):
#    return -dice_coef(y_true, y_pred)
#===============================================================================


DATA_PATH = 'data/'
image_size = 128
batch_size = 128
epochs = 10000

#===============================================================================
##  Data
X_train, y_train, X_val, y_val = read_data(DATA_PATH, image_size)

train_gen = generator.Generator(X_train, y_train, batch_size)
val_gen = generator.Generator(X_train, y_val, batch_size)
#===============================================================================

#===============================================================================
##  Model
model = UNet(image_size)
adam = keras.optimizers.Adam(lr = 1e-3)
loss_binary_crossentropy = keras.losses.BinaryCrossentropy()
model.compile( loss = loss_binary_crossentropy,
               optimizer = adam,
               metrics=[dice_coef]
             )
model.summary()
#===============================================================================

#===============================================================================
##  Training
train_steps = len(X_train)//batch_size
valid_steps = len(X_val)//batch_size

## Callbacks
history = my_callback.Histories(np.asarray(X_val), batch_size)

checkpoint = ModelCheckpoint(DATA_PATH+'modelcheckpoint_full-{epoch:02d}.h5',
                verbose=1,
                save_best_only=True,
                monitor='val_loss')

earlystop = EarlyStopping(monitor='val_loss',
            min_delta=0.00000001,
            patience=2000,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True)

callbacks = [history, checkpoint, earlystop]

model.fit_generator(train_gen,
                    validation_data=val_gen,
                    steps_per_epoch=train_steps, validation_steps=valid_steps,
                    epochs=epochs, verbose=1, #validation_freq = 200,
                    callbacks = callbacks)
                    #callbacks = [checkpointer])

##  Save weights and model

model.save_weights(DATA_PATH+"UNet_weights_full.h5")
#model.save(DATA_PATH+"model_full.h5")

tf.keras.models.save_model(
    model,
    DATA_PATH+"model_full.h5",
    overwrite=True,
    include_optimizer=True
)

#del model

##  Save loss and validation_loss
history_dictionary_loss = history.loss
np.save(DATA_PATH+'history_loss_full.npy', history_dictionary_loss)
history_dictionary_val_loss = history.val_loss
np.save(DATA_PATH+'history_val_loss_full.npy', history_dictionary_val_loss)

##  Plot
plt.figure(1)
plt.yscale("log")
plt.plot(history.loss)
plt.plot(history.val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig(DATA_PATH+'loss_full.png')
