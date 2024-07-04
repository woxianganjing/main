import numpy as np
import os
import re
import scipy.io as scio

import numpy as np
import pandas as pd
import scipy.signal
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from keras.layers import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras import backend as k
from keras.callbacks import ModelCheckpoint

raw_num = 30
col_num = 2000


class Data(object):

    def __init__(self):
        self.data = self.get_data()
        self.label = self.get_label()

    def file_list(self):
        return os.listdir('data/')

    def get_data(self):
        file_list = self.file_list()
        for i in range(len(file_list)):
            file = scio.loadmat('data/{}'.format(file_list[i]))
            for k in file.keys():
                file_matched = re.match('X\d{3}_DE_time', k)
                if file_matched:
                    key = file_matched.group()
            if i == 0:
                data = np.array(file[key][0:60000].reshape(raw_num, col_num))
            else:
                data = np.vstack((data, file[key][0:60000].reshape((raw_num, col_num))))
        return data

    def get_label(self):
        file_list = self.file_list()
        title = np.array([i.replace('.mat', '') for i in file_list])
        label = title[:, np.newaxis]
        label_copy = np.copy(label)
        for _ in range(raw_num - 1):
            label = np.hstack((label, label_copy))
        return label.flatten()
Data = Data()
data = Data.data
label = Data.label
lb = LabelBinarizer()
y = lb.fit_transform(label)

# Wiener filtering
data_wiener = scipy.signal.wiener(data, mysize=3, noise=None)

# downsampling
index = np.arange(0,2000, 8)
data_samp = data_wiener[:, index]
print(data_samp.shape)

X_train, X_test, y_train, y_test = train_test_split(data_samp, y, test_size=0.3)


def built_model():
    input_seq = Input(shape=(250,))
    X = Reshape((1, 250))(input_seq)

    # encoder1
    ec1_layer1 = Conv1D(filters=50, kernel_size=20, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec1_layer2 = Conv1D(filters=30, kernel_size=10, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec1_layer1)
    ec1_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec1_layer2)

    # encoder2
    ec2_layer1 = Conv1D(filters=50, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(X)
    ec2_layer2 = Conv1D(filters=40, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer1)
    ec2_layer3 = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                              data_format='channels_first')(ec2_layer2)
    ec2_layer4 = Conv1D(filters=30, kernel_size=6, strides=1,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer3)
    ec2_layer5 = Conv1D(filters=30, kernel_size=6, strides=2,
                        padding='valid', activation='tanh',
                        data_format='channels_first')(ec2_layer4)
    ec2_outputs = MaxPooling1D(pool_size=2, strides=None, padding='valid',
                               data_format='channels_first')(ec2_layer5)

    encoder = multiply([ec1_outputs, ec2_outputs])

    pooled_seq = GlobalAveragePooling1D()(encoder)

    dc_layer4 = Dense(6, activation='softmax')(pooled_seq)

    model = Model(input_seq, dc_layer4)

    return model


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
    plt.imshow(cm , cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_mark = np.arange(len(classes))
    plt.xticks(tick_mark, classes, rotation=40)
    plt.yticks(tick_mark, classes)
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        cm = '%.2f'%cm
    thresh = cm.max()/2.0
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j], horizontalalignment='center',color='black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')

import time
begain_time = time.time()

model = built_model()
opt = Adam(lr=0.0006)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model.summary()

history = model.fit(x=X_train, y=y_train, batch_size = 100, epochs=400,
                    verbose=2, validation_data=(X_test, y_test),
                    shuffle=True, initial_epoch=0)

y_pre = model.predict(X_test)
label_pre = np.argmax(y_pre, axis=1)
label_true = np.argmax(y_test, axis=1)
confusion_mat = confusion_matrix(label_true, label_pre)
plot_confusion_matrix(confusion_mat, classes=range(6))
plt.show()