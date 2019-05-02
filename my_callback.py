import tensorflow as tf
from tensorflow import keras

class Histories(keras.callbacks.Callback):
  def __init__(self, x, batch_size):
    self.batch_size = batch_size
    self.x = x

  def on_train_begin(self, logs={}):
    self.loss = []
    self.val_loss = []
    self.acc = []
    self.val_acc = []
    self.dc = []
    self.dc2 = []
    self.out = []
    self.LR = []

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    self.loss.append(logs.get('loss'))
    self.val_loss.append(logs.get('val_loss'))
    self.acc.append(logs.get('acc'))
    self.val_acc.append(logs.get('val_acc'))

    #decay = self.model.optimizer.decay
    #print("decay",decay)
    #lr = self.model.optimizer.lr
    #print("lr",lr)
    #iters = self.model.optimizer.iterations # only this should not be const
    #print("iters", K.eval(iters))
    #beta_1 = self.model.optimizer.beta_1
    #beta_2 = self.model.optimizer.beta_2

    # calculate

    #lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
    #t = K.cast(iters, K.floatx()) + 1
    #lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
    #ev_lr = np.float32(K.eval(lr_t))

    #ev_lr = K.eval(model.optimizer.lr)
    #print("LR", ev_lr)
    #self.LR.append(ev_lr)


    #self.x = self.x.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], 1)
    #self.out.append(self.model.predict(self.x, batch_size=self.batch_size))

    #tmp = []
    #tmp2 = []
    #for i in range(5):
      #print(type(self.batch[i][0]))
    #  y_pred = self.model.predict(self.batch[i][0])
    #  y_pred2 = self.model.predict(self.batch2[i][0])
    #  dice = tf.keras.backend.eval(dice_coef(tf.cast(self.batch[i][1], tf.float32), tf.cast(y_pred, tf.float32)))
    #  dice2 = tf.keras.backend.eval(dice_coef(tf.cast(self.batch2[i][1], tf.float32), tf.cast(y_pred2, tf.float32)))
    #  tmp.append(dice)
    #  tmp2.append(dice2)

    #self.dc.append((sum(tmp) / len(tmp)))
    #self.dc2.append((sum(tmp2) / len(tmp2)))

    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return
