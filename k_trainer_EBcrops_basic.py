# Train a CNN model to discriminate electron vs photon showers
# Image size: ieta:32 x iphi:32 x qty:2 (0: energy, 1:timing)

import numpy as np
np.random.seed(1337)  # for reproducibility

import argparse
parser = argparse.ArgumentParser(description='Electron v Photon VGG classifier.')
parser.add_argument('--lr_init', default=1.e-3, type=float, help='Initial learning rate.')
parser.add_argument('--train_size', default=160000, type=int, help='Training size per decay.')
parser.add_argument('--valid_size', default=44800, type=int, help='Validation size per decay.')
parser.add_argument('--test_size', default=44200, type=int, help='Test size per decay.')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for gradient update during training.')
parser.add_argument('--epochs', default=25, type=int, help='Number of training epochs.')
parser.add_argument('--doGPU', default=False, type=bool, help='Enable GPU memory optimizations.')
args = parser.parse_args()

# If running GPU, optimize memory allocation over multiple GPUs
if args.doGPU:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))

### Load Image Data ###

import h5py
img_rows, img_cols, nb_channels = 32, 32, 2
input_dir = 'IMG'
decays = ['SinglePhotonPt50_IMGCROPS_n249k_RHv1', 'SingleElectronPt50_IMGCROPS_n249k_RHv1']

def load_data(decays, start, stop):
    global input_dir
    dsets = [h5py.File('%s/%s.hdf5'%(input_dir,decay)) for decay in decays]
    X = np.concatenate([dset['/X'][start:stop] for dset in dsets])
    y = np.concatenate([dset['/y'][start:stop] for dset in dsets])
    assert len(X) == len(y)
    return X, y

# Set range of training set
train_start, train_stop = 0, args.train_size
assert train_stop > train_start
assert (len(decays)*args.train_size) % args.batch_size == 0
X_train, y_train = load_data(decays,train_start,train_stop)

# Set range of validation set
valid_start, valid_stop = 160000, 160000+args.valid_size
assert valid_stop  >  valid_start
assert valid_start >= train_stop
X_valid, y_valid = load_data(decays,valid_start,valid_stop)

# Set range of test set
test_start, test_stop = 204800, 204800+args.test_size
assert test_stop  >  test_start
assert test_start >= valid_stop
X_test, y_test = load_data(decays,test_start,test_stop)

samples_requested = len(decays) * (args.train_size + args.valid_size + args.test_size)
samples_available = len(y_train) + len(y_valid) + len(y_test)
assert samples_requested == samples_available

### Define CNN Model ###

from keras.models import Sequential
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(16, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal', input_shape=(img_rows, img_cols, nb_channels)))
model.add(Conv2D(16, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(Conv2D(32, activation='relu', kernel_size=3, padding='same', kernel_initializer='TruncatedNormal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='TruncatedNormal'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_initializer='TruncatedNormal'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid', kernel_initializer='TruncatedNormal'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.lr_init), metrics=['accuracy'])
model.summary()

### Train Model ###

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1.e-6)
model.fit(X_train, y_train,\
        batch_size=args.batch_size,\
        epochs=args.epochs,\
        validation_data=(X_valid, y_valid),\
        callbacks=[reduce_lr],\
        verbose=1, shuffle=True)

### Evaluate Model ###

# Evaluate on validation set
from sklearn.metrics import roc_curve, auc
score = model.evaluate(X_valid, y_valid, verbose=1)
print('\nValidation loss / accuracy: %0.4f / %0.4f'%(score[0], score[1]))
y_pred = model.predict(X_valid)
fpr, tpr, _ = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr)
print('Validation ROC AUC:', roc_auc)

# Evaluate on test set
score = model.evaluate(X_test, y_test, verbose=1)
print('\nTest loss / accuracy: %0.4f / %0.4f'%(score[0], score[1]))
y_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print('Test ROC AUC:', roc_auc)
