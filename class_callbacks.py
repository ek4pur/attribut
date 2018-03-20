from keras.callbacks import Callback
import os


class CekPoint(Callback):
    def __init__(self, path,generator,discriminator):
        self.path = path
        self.generator = generator
        self.discriminator = discriminator

    def on_train_begin(self, logs={}):
        self.losses = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.generator.save(os.path.join(self.path, "generator-epoch-{:03d}.h5".format(epoch)))
        self.discriminator.save(os.path.join(self.path, "discriminator-epoch-{:03d}.h5".format(epoch)))

