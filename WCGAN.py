from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization,  Embedding
from keras.layers import LeakyReLU
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
import os
import json
import gc
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

batch_size = 32
signal_dimensionality = 1024
channels = 1
input_signal_shape = (signal_dimensionality, channels)
num_classes = 51
latent_dim = 100
epoch = 10000

def build_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(input_signal_shape),activation='softplus'))
    model.add(Reshape(input_signal_shape))
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes,latent_dim)(label)) #Embedding是将所有索引标号映射到稀疏的高维向量中,也就是江64个类用100维的向量表示。
    model_input = multiply([noise, label_embedding])  #将label的meaning嵌入noise
    generated_signal = model(model_input)
    return Model([noise, label], generated_signal)

def build_discriminator():
    model = Sequential()
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    signal = Input(shape=(signal_dimensionality,1))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, np.prod(input_signal_shape))(label))
    flat_signal = Flatten()(signal)
    model_input = multiply([flat_signal, label_embedding])
    validity = model(model_input)
    return Model([signal, label], validity)

# Following parameter and optimizer set as recommended in paper
n_critic = 5
clip_value = 0.01
def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# d_optimizer = Adam(0.0001, 0.5)
# g_optimizer = Adam(0.0001, 0.5)
# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(0.00005), metrics=['accuracy'])

# Build the generator
generator = build_generator()
# The generator takes noise and the target label as input
# and generates the corresponding new signals of that label
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
signal = generator([noise, label])

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated image as input and determines validity
# and the label of that image
combined_output = discriminator([signal, label])

# The combined model  (stacked generator and discriminator)
# Trains generator to fool discriminator
combined = Model([noise, label], combined_output)
combined.compile(loss=d_loss, optimizer=RMSprop(0.00005))

# # Load the dataset
x_train = np.load('Data_now/X_TrainData.npy')
y_train = np.arange(0,num_classes,1).repeat(10)

# Adversarial ground truths
valid = np.repeat(0.1, batch_size)
fake = np.ones((batch_size, 1))

# D_loss = []
# D_acc = []
# G_loss = []

##training the CGAN
for epoch in range(epoch):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Select a random half batch of images

    for _ in range(n_critic):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        signals, labels = x_train[idx], y_train[idx]

        a = np.arange(0.1, 0.9, 0.02)
        x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
        r = np.arange(2, 6.1, 0.1)
        count = 0
        fs = len(x)
        y = np.zeros((1, len(x)))
        for n in range(batch_size):
            amp = 10 * a[np.random.randint(0, len(a) - 1)]
            rad = r[np.random.randint(0, len(r) - 1)]
            phase = np.random.uniform(-1, 1) * np.pi

            y = np.append(y, amp * np.sin(((2 * np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
        noise = np.delete(y, 0, axis=0)

        # Generate a half batch of new images
        gen_signals = generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch([signals, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_signals, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Clip critic weights
        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)
    # ---------------------
    #  Train Generator
    # ---------------------
    # D_loss.append(d_loss[0])
    # D_acc.append(100 * d_loss[1])

    # Condition on labels
    sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

    # Train the generator
    g_loss = combined.train_on_batch([noise, sampled_labels], valid)
    # Plot the progress

    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    # 手动清理未使用的变量和内存
    K.clear_session()
    gc.collect()
    # save the progress
    # G_loss.append(g_loss)
    # 保存当前 epoch 的模型权重和架构
    # generator_weight_file = 'generator_latest_weights'
    # generator_config_file = 'generator_latest_config.json'
    # generator.save_weights(generator_weight_file)
    # config = generator.get_config()
    # for layer in config['layers']:
    #     for key, value in layer['config'].items():
    #         if isinstance(value, np.int32):
    #             layer['config'][key] = int(value)
    # with open(generator_config_file, 'w') as f:
    #     json.dump(config, f)
    #
    # discriminator_weight_file = 'discriminator_latest_weights'
    # discriminator.save_weights(discriminator_weight_file)
    # config = discriminator.get_config()
    # for layer in config['layers']:
    #     for key, value in layer['config'].items():
    #         if isinstance(value, np.int32):
    #             layer['config'][key] = int(value)
    # discriminator_config_file = 'discriminator_latest_config.json'
    # with open(discriminator_config_file, 'w') as f:
    #     json.dump(config, f)
    #

# # -----------------绘制loss图-----------------------------# #
# D_loss = np.array(D_loss)
# D_acc = np.array(D_acc)
# G_loss = np.array(G_loss)
# plt.plot(D_loss)
# plt.plot(G_loss)
# plt.show()
# np.save('G-loss',G_loss)
# np.save('D-loss',D_loss)
# np.save('D-acc', D_acc)
#------------------可视化生成的信号-------------------------# #
generator = load_model('WCGAN_v2_60000_generator')
a = np.arange(0.1, 0.9, 0.02)
x = np.arange(0, latent_dim/2, 0.5)  # 128/2=64
r = np.arange(0.1, 6.1, 0.1)
count = 0
fs = len(x)
y = np.zeros((1, len(x)))
for n in range(1):
    amp = 10 * a[np.random.randint(0, len(a) - 1)]
    rad = r[np.random.randint(0, len(r) - 1)]
    phase = np.random.uniform(-1, 1) * np.pi

    y = np.append(y, amp * np.sin(((2 * np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
noise = np.delete(y, 0, axis=0)

# labels = np.random.randint(0, 51, 1).reshape(-1, 1)
labels = np.array((21)).reshape(-1,1)
generate_signal = generator.predict([noise, labels]).flatten()
plt.plot(generate_signal[500:700])
plt.show()
# # #  #---------------产生对应的真实的数据-------------------# #
x = np.load('Training_and_testing_data_for_CGAN_models/X_TrainData.npy')
x_real = x[21*10+4,:]
plt.plot(x_real[500:700])
plt.show()

#---------------观察网络结构---------------------# #

# generator.save('WCGAN_v2_60000_generator')
# discriminator.save('WCGAN_V2_60000_discriminator')
# import pickle

# # 保存生成器优化器状态
# with open('WCGAN_20000_generator_optimizer.pkl', 'wb') as f:
#     pickle.dump(generator.get_weights(), f)
#
# # 保存判别器优化器状态
# with open('WCGAN_20000_discriminator_optimizer.pkl', 'wb') as f:
#     pickle.dump(discriminator.get_weights(), f)
# print(generator.summary())
# print(combined.summary())
# print(discriminator.summary())

# #--------------加载保存的模型---------------------# #
# generator.save('generator1(432best)')
# discriminator.save('discriminator1(432best)')
# combined.save('combined1(432best')


