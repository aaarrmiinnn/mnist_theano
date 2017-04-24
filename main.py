
# coding: utf-8

# # Importing Libraries

# In[19]:

import os
import numpy as np
np.random.seed(1337)  # for reproducibility
import csv
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
#from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import LeakyReLU



# In[2]:

cwd=os.getcwd()
print("Working Dir Is:\n",cwd)
print(os.listdir(cwd+"/Data/"))


# In[3]:

train_data = csv.reader(open(cwd+"/Data/train.csv","r"),delimiter=',')
x = list(train_data)
train_XY= np.array(x[1:]).astype('float')
test_data = csv.reader(open(cwd+"/Data/test.csv","r"),delimiter=',')
x = list(test_data)
test_XY= np.array(x[1:]).astype('float')


# In[21]:

batch_size = 128
nb_classes = 10
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
nb_filters_2 = 64 
nb_filters_3 = 128 
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


# In[5]:

train_percentage = 0.95
train_size = int(np.round(train_XY.shape[0]*train_percentage))


# In[6]:

X_train = train_XY[0:train_size,1:]
print ("X_train size is:",X_train.shape)
y_train = train_XY[0:train_size,0]
print ("y_train size is:",y_train.shape)
X_test = train_XY[train_size:,1:   ]
print ("X_test size is:",X_test.shape)
y_test = train_XY[train_size:,0]
print ("y_test size is:",y_test.shape)

X_submission = test_XY
print ("X_submission size is:",X_submission.shape)


# In[7]:

X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,1)
X_submission = X_submission.reshape(X_submission.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_submission = X_submission.astype('float32')
X_train /= 255
X_test /= 255
X_submission /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[8]:

y_train = np.asarray(y_train).astype('int8')
y_test = np.asarray(y_test).astype('int8')


# In[9]:

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[22]:

model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_rows, img_cols,1)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation(LeakyReLU(alpha=0.1)))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(nb_filters_3, (nb_conv, nb_conv), activation="relu"))
model.add(MaxPooling2D(strides=(2,2)))


#model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.1,training=True))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.2,training=True))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[23]:

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

prediction_results = model.predict_classes(X_submission, batch_size=32, verbose=1)


# In[ ]:

prediction_results = prediction_results.astype('uint8')
print (prediction_results)


# In[ ]:

save_name = 'prediction_results___'+time.strftime("H_%H_%M_%S___D_%d_%m_%Y")+'.csv'


# In[36]:

np.savetxt(save_name, np.c_[range(1,len(prediction_results)+1),prediction_results], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')





