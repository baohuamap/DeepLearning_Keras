from sklearn import metrics
from tensorboard import summary
import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

#load data mnist
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(x_train1, y_train1), (x_test1, y_test1) = keras.datasets.mnist.load_data()


# 60000 anh train va 10000 anh test, kich thuoc anh 28x28
print("Train data shape: ", x_train.shape,"\nTest data shape: ", x_test.shape)

# show anh de test
# plt.imshow(x_train[10])
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_test)

# one-hot encode target column
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

#create model
# model = Sequential()

# model.add(Conv2D(64, kernel_size= (3,3), activation='relu', input_shape=(28,28,1), padding='same' ))
# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# model.add(Conv2D(32, kernel_size= (3,3), activation='relu', padding='same' ))
# model.add(MaxPooling2D(pool_size=(2,2), strides=2))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# model.summary()
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=3)
# model.save('Model_NumRegconition.h5')

model = keras.models.load_model('Model_NumRegconition.h5')

y_hat = model.predict(x_test[19:20])
# print(y_hat)

y_label = np.argmax(y_hat,axis=1)
print(y_label)
plt.imshow(x_test[19])
plt.show()