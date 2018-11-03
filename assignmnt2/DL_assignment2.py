
# coding: utf-8

# In[4]:


from keras import layers
from keras import models
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
import keras.utils.np_utils as kutils
import numpy as np
from keras import backend as K


# In[7]:


img_rows, img_cols = 32, 32
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX / 255
testX = testX / 255
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)


# In[10]:


# initial model define
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='random_normal'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='random_normal'))
model.add(layers.MaxPooling2D((2, 2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
history = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY))
print("Finished compiling 1")


# In[11]:


layer_outputs = [layer.output for layer in model.layers[:4]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(trainX)
first_layer_activation = activations[0]
for i in range(6):
    plt.matshow(first_layer_activation[0, :, :, i], cmap='viridis')


# In[23]:


# Converts tensors to valid images
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Functions that generate filter visualizations
def generate_pattern(filter_index, size=32):
    layer_output = model.layers[0].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)
    print(len(grads))
    grads = grads[0]
    print(grads.shape)
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]
    return deprocess_image(img)


# Generates a grid of all filter response patterns in a layer
size = 28
plt.figure(figsize=(15, 10))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    filter_img = generate_pattern(i, size=size)
    plt.imshow(filter_img)
    plt.grid([])
    plt.title('Filter' + str(i + 1))
plt.show()


# In[12]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[13]:


# model define: with dropout in FC layer
model2 = models.Sequential()
model2.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='random_normal'))
model2.add(layers.MaxPooling2D((2, 2), strides=2))
model2.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='random_normal'))
model2.add(layers.MaxPooling2D((2, 2), strides=2))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(120, activation='relu'))
model2.add(layers.Dense(84, activation='relu'))
model2.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model2.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
history2 = model2.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY))
print("Finished compiling 2")


# In[16]:


acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy with dropout')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss with dropout')
plt.legend()
plt.show()


# In[15]:


# model define with different initializer
model3 = models.Sequential()
model3.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
model3.add(layers.MaxPooling2D((2, 2), strides=2))
model3.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal'))
model3.add(layers.MaxPooling2D((2, 2), strides=2))
model3.add(layers.Flatten())
model3.add(layers.Dropout(0.5))
model3.add(layers.Dense(120, activation='relu'))
model3.add(layers.Dense(84, activation='relu'))
model3.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model3.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
history3 = model3.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY))
print("Finished compiling 3")


# In[17]:


acc = history3.history['acc']
val_acc = history3.history['val_acc']
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy with he_normal')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss with he_normal')
plt.legend()
plt.show()


# In[21]:


# model define with batch normalization
model4 = models.Sequential()
model4.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
model4.add(layers.BatchNormalization())
model4.add(layers.MaxPooling2D((2, 2), strides=2))
model4.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal'))
model4.add(layers.BatchNormalization())
model4.add(layers.MaxPooling2D((2, 2), strides=2))
model4.add(layers.Flatten())
model4.add(layers.Dropout(0.5))
model4.add(layers.Dense(120, activation='relu'))
model4.add(layers.BatchNormalization())
model4.add(layers.Dense(84, activation='relu'))
model4.add(layers.BatchNormalization())
model4.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model4.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])
history4 = model4.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY))
print("Finished compiling 4")


# In[24]:


acc = history4.history['acc']
val_acc = history4.history['val_acc']
loss = history4.history['loss']
val_loss = history4.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy with BN')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss with BN')
plt.legend()
plt.show()


# In[37]:


# self-defined model
model5 = models.Sequential()
model5.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
model5.add(layers.BatchNormalization())
model5.add(layers.MaxPooling2D((2, 2), strides=2))
model5.add(layers.Conv2D(16, (5, 5), activation='relu', kernel_initializer='he_normal'))
model5.add(layers.BatchNormalization())
model5.add(layers.MaxPooling2D((2, 2), strides=2))
model5.add(layers.Flatten())
model5.add(layers.Dropout(0.5))
model5.add(layers.Dense(120, activation='relu'))
model5.add(layers.Dense(84, activation='relu'))
model5.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model5.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])


# In[38]:


train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen.fit(trainX, seed=0, augment=True)
test_generator = ImageDataGenerator(zca_epsilon=0, horizontal_flip=True, fill_mode='reflect', )
test_generator.fit(testX, seed=0, augment=True)
history5 = model5.fit_generator(train_datagen.flow(trainX, trainY, batch_size=256), steps_per_epoch=100, epochs=50,
                                validation_data=test_generator.flow(testX, testY, batch_size=256),validation_steps=50)


# In[33]:


acc = history5.history['acc']
val_acc = history5.history['val_acc']
loss = history5.history['loss']
val_loss = history5.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy of DIY model')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of DIY model')
plt.legend()
plt.show()


# In[15]:


# model5.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3), kernel_initializer='he_normal'))
# self-defined model based on VGG-16
model6 = models.Sequential()
model6.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(48, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(48, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.MaxPooling2D(pool_size=(2, 2)))
model6.add(layers.Dropout(0.25))
model6.add(layers.Conv2D(80, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(80, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(80, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(80, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(80, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.MaxPooling2D(pool_size=(2, 2)))
model6.add(layers.Dropout(0.25))
model6.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.Conv2D(128, (3, 3), activation='relu',padding='same', input_shape=trainX.shape[1:]))
model6.add(layers.GlobalMaxPooling2D())
model6.add(layers.Dropout(0.25))
model6.add(layers.Dense(500, activation='relu'))
model6.add(layers.Dropout(0.25))
model6.add(layers.Dense(10, activation='softmax'))
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
model6.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['acc'])

train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_datagen.fit(trainX, seed=0, augment=True)
test_generator = ImageDataGenerator(zca_epsilon=0, horizontal_flip=True, fill_mode='reflect', )
test_generator.fit(testX, seed=0, augment=True)
history6 = model6.fit_generator(train_datagen.flow(trainX, trainY, batch_size=64), steps_per_epoch=100, epochs=800,
                                validation_data=test_generator.flow(testX, testY, batch_size=64),validation_steps=50)
print("Finished compiling 6")


# In[19]:


acc = history6.history['acc']
val_acc = history6.history['val_acc']
loss = history6.history['loss']
val_loss = history6.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'go', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy of DIY model')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss of DIY model')
plt.legend()
plt.show()

