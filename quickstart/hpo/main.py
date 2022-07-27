import datetime
import os
import sys

import keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import datetime
import socket

log_dir = "/nfs/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + socket.gethostname()
checkpoints_file = log_dir + "/weights.best.hdf5"

os.makedirs(log_dir + "/ckpt/", exist_ok=True)

# import Run:AI HPO assistance library
import runai.hpo

def usage():
    print("usage: python %s <HPO directory> [strategy: random | grid; default: grid]" % sys.argv[0])
    exit(1)

if len(sys.argv) != 2 and len(sys.argv) != 3:
    usage()

hpo_dir = sys.argv[1]
print("Using HPO directory %s" % hpo_dir)

if len(sys.argv) == 3:
    if sys.argv[2] == 'grid':
        strategy = runai.hpo.Strategy.GridSearch
    elif sys.argv[2] == 'random':
        strategy = runai.hpo.Strategy.RandomSearch
    else:
        usage()
else:
    strategy = runai.hpo.Strategy.GridSearch

# initialize the Run:AI HPO assistance library
subdir = '%s_%s' % (os.getenv('jobName', 'hpo'), os.getenv('jobUUID', datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')))
runai.hpo.init(hpo_dir, subdir=subdir)

# pick a configuration for this HPO experiment
# we pass the options of all hyperparameters we want to test
# `config` will hold a single value for each parameter
config = runai.hpo.pick(
    grid=dict(
        batch_size=[4, 8, 16, 32, 64, 128],
        lr=[10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]),
    strategy=strategy)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = keras.applications.vgg16.preprocess_input(x_train)
x_test = keras.applications.vgg16.preprocess_input(x_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

optimizer = keras.optimizers.SGD(lr=config['lr'])

model = keras.applications.vgg16.VGG16(
    include_top=True,
    weights=None,
    input_shape=(32,32,3),
    classes=10,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

class ReportCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        runai.hpo.report(epoch, dict(
            loss=float(logs['loss']),
            acc=float(logs['acc']),
            val_loss=float(logs['val_loss']),
            val_acc=float(logs['val_acc']),
        ))


# register a 'save checkpoints' callback. Default is every epoch
checkpoint_callback = ModelCheckpoint(
    checkpoints_file, monitor='val_acc', 
    verbose=1, save_best_only=True, mode='max')

# Allow logs to be read from TensorBoard
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['batch_size'],
    epochs=5,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[ReportCallback(), checkpoint_callback, tensorboard_callback],
)
