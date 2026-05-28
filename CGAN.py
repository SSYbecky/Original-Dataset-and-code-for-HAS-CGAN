from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization,  Embedding
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.losses import BinaryCrossentropy
import numpy as np
import os
import keras.backend as K
import gc
import tensorflow as tf
from keras.layers import Layer

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
batch_size = 32
signal_dimensionality = 1024
channels = 1
input_signal_shape = (signal_dimensionality, channels)
num_classes = 42
latent_dim = 100
epoch = 10000

def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(input_signal_shape),activation='softplus'))
    model.add(Reshape(input_signal_shape))
    model.summary()
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes,latent_dim)(label)) #Embedding是将所有索引标号映射到稀疏的高维向量中,也就是江64个类用100维的向量表示。
    model_input = multiply([noise, label_embedding])  #将label的meaning嵌入noise
    generated_signal = model(model_input)
    return Model([noise, label], generated_signal)

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=np.prod(input_signal_shape)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    signal = Input(shape=(signal_dimensionality,1))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(input_signal_shape))(label))
    flat_signal = Flatten()(signal)
    model_input = multiply([flat_signal, label_embedding])
    validity = model(model_input)
    return Model([signal, label], validity)

d_optimizer = Adam(0.0001, 0.5)
g_optimizer = Adam(0.0001, 0.5)
# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss= 'binary_crossentropy',optimizer=d_optimizer,metrics=['accuracy'])

# Build the generator
generator = build_generator()
# The generator takes noise and the target label as input
# and generates the corresponding new signals of that label
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
gen_signal = generator([noise, label])

# For the combined model we will only train the generator
discriminator.trainable = False

combined_output = discriminator([gen_signal, label])
combined = Model([noise, label], combined_output)

combined.summary()

combined.compile(loss='binary_crossentropy',optimizer=g_optimizer)

# Adversarial ground truths
valid = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

# # Load the dataset
x_train = np.load('Training_and_testing_data_for_CGAN_models/X_except_former8.npy')
y_train = np.load('Training_and_testing_data_for_CGAN_models/Y_except_former8.npy')

def generate_noise():
    a = np.arange(0.1, 1.5, 0.02)
    x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
    r = np.arange(0.1, 6.1, 0.1)
    fs = len(x)
    y = np.zeros((1,len(x)))
    for n in range(batch_size):
        amp = 10*a[np.random.randint(0, len(a)-1)]
        rad= r[np.random.randint(0, len(r)-1)]
        phase = np.random.uniform(-1,1)*np.pi

        y = np.append(y, amp * np.sin(((2* np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise = np.delete(y, 0, axis=0)
    return noise

##training the CGAN
for epoch in range(epoch):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    signals, labels = x_train[idx], y_train[idx]

    # Sample noise as generator input
    noise = generate_noise()
    # Generate a half batch of new images
    gen_signals = generator.predict([noise, labels])

    # Train the discriminator
    discriminator.trainable=True
    d_loss_real = discriminator.train_on_batch([signals, labels], valid)
    d_loss_fake = discriminator.train_on_batch([gen_signals, labels], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # ---------------------
    #  Train Generator

    # Condition on labels
    discriminator.trainable=False
    sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
    # Train the generator
    g_loss = combined.train_on_batch([noise, sampled_labels], valid)

    # Plot the progress
    print ("%d [D loss: %f, acc.: %.2f%%] [ G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    # save the progress

    K.clear_session()
    gc.collect()

#------------------可视化生成的信号-------------------------# #
a = np.arange(0.1, 1.5, 0.02)
x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
r = np.arange(0.1, 6.1, 0.1)
count = 0
fs = len(x)
y = np.zeros((1,len(x)))

amp = 10*a[np.random.randint(0, len(a)-1)]
rad= r[np.random.randint(0, len(r)-1)]
phase = np.random.uniform(-1,1)*np.pi

y = np.append(y, amp * np.sin(((2* np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
noise = np.delete(y, 0, axis=0)

# labels = np.random.randint(0, 42, 1).reshape(-1, 1)
labels = np.array((38)).reshape(-1,1)
generate_signal = generator.predict([noise, labels]).flatten()
plt.plot(generate_signal[500:700])
plt.show()
