import numpy as np
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import cv2 as cv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

matTr = loadmat('train_32x32.mat')
matTe = loadmat('test_32x32.mat')

# labels are originally in [1,10] and now will be in [0,9]
Xtr, Ytr = matTr['X'], matTr['y']-1
Xte, Yte = matTe['X'], matTe['y']-1
# Xext, Yext = matExt['X'], matExt['y']-1

# changing the dimensions so that the number of the input image is the first
Xtr = np.transpose(Xtr, (3, 0, 1, 2))
Xte = np.transpose(Xte, (3, 0, 1, 2))
# Xext = np.transpose(Xext, (3, 0, 1, 2))

# Xtr_ext = np.concatenate((Xtr,Xext))
# Ytr_ext = np.concatenate((Ytr,Yext))

# Xtr_ext = Xtr_ext / 255.0
Ytr = np.squeeze(Ytr)
Yte = np.squeeze(Yte)
# Yext = np.squeeze(Yext)

batch_size = 128
epochs = 50
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CHANNEL = 3

Xtr_gray = tf.image.rgb_to_grayscale(Xtr)
Xtr_gray = K.eval(Xtr_gray)

Xtr_bin = np.zeros((len(Xtr_gray),32,32,1))
for i in range(len(Xtr_gray)):
    Xtr_bin[i,:,:,0] = cv.adaptiveThreshold(Xtr_gray[i,:,:,0],255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

# ind = 700
# titles = ['RGB Image', 'GRAY_SCALE', 'BINARY']
# images = [Xtr[ind], Xtr_gray[ind,:,:,0], Xtr_bin[ind,:,:,0]]
# for i in range(3):
#     plt.subplot(1,3,i+1)
#     if i!=0:
#         plt.imshow(images[i],'gray')
#     else:
#         plt.imshow(images[i])
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.savefig('images.png',bbox_inches = 'tight')
# plt.show()

# plt.hist(Ytr, ec='k',bins=10)
# plt.title('Histogram of Train Data')
# plt.savefig('histogram.png',bbox_inches = 'tight')

# model = Sequential()
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(32,32,3)))
# model.add(BatchNormalization())
# model.add(Conv2D(32, 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(64, kernel_size = 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size = 3, activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))

# model.add(Conv2D(128, kernel_size = 4, activation='relu'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dropout(0.4))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

model = Sequential()
model.add(Conv2D(32, kernel_initializer='he_normal', kernel_size=5, activation='relu', padding="same", input_shape=(32,32,3)))
# model.add(BatchNormalization())
# model.add(Conv2D(64, kernel_initializer='he_normal', kernel_size=3, activation='relu', padding="same"))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(64, kernel_initializer='he_normal', kernel_size = 3, activation='relu', padding="same"))
# model.add(Conv2D(64, kernel_initializer='he_normal', kernel_size = 3, activation='relu', padding="same"))
# # model.add(Dropout(0.3))
# model.add(Conv2D(128, kernel_initializer='he_normal', kernel_size = 3, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.summary()
history = model.fit(Xtr,Ytr,epochs=50, validation_split=0.2, batch_size=128, verbose=2)

print("CNN: Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(
    epochs,history['acc'],history['val_acc']))
