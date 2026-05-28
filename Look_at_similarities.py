from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

index = 20

# #---------------产生对应的真实的数据-------------------# #
x = np.load('Data_now/X_TrainData.npy')
x_real = x[index*10+3,:]
plt.plot(x_real[500:700])
plt.show()

# #---------------产生噪音数据-------------------------# #

latent_dim = 100

def generate_noise0():
    a = np.arange(0.1, 1.5, 0.02)
    x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
    r = np.arange(0.1, 6.1, 0.1)
    fs = len(x)
    y = np.zeros((1,len(x)))

    amp = 10*a[np.random.randint(0, len(a)-1)]
    rad= r[np.random.randint(0, len(r)-1)]
    phase = np.random.uniform(-1,1)*np.pi

    y = np.append(y, amp * np.sin(((2* np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise0 = np.delete(y, 0, axis=0)
    return noise0

def generate_noise():
    a = np.arange(0.1, 1.5, 0.02)
    x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
    r = np.arange(0.1, 6.1, 0.1)
    fs = len(x)
    y = np.zeros((1,len(x)))
    amp = 10*a[np.random.randint(0, len(a)-1)]
    rad= r[np.random.randint(0, len(r)-1)]
    phase = np.random.uniform(-1,1)*np.pi
    y = np.append(y, amp * np.sin(((2* np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise1 = np.delete(y, 0, axis=0)

    r1 = np.arange(6.1, 12.1, 0.1)
    y1 = np.zeros((1, len(x)))
    rad1 = r[np.random.randint(0, len(r1) - 1)]
    y1 = np.append(y1, amp * np.sin(((2 * np.pi * rad1 * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise2 = np.delete(y1, 0, axis=0)
    noise = noise1+noise2
    return noise

labels = np.array((index)).reshape(-1,1)

# #---------------产生CGAN_1D_customized_loss的信号(1)
noise = generate_noise()
generator_1 = load_model('Well_trained_CGANS_models/HAS_CGAN_60000_generator')
generate_signal_1 = generator_1.predict([noise, labels]).flatten()
plt.plot(generate_signal_1[500:700])
plt.show()

# #---------------产生CGAN_1D的信号(2)
noise0 = generate_noise0()
generator_2 = load_model('Well_trained_CGANS_models/CGAN_60000_generator')
generate_signal_2 = generator_2.predict([noise0, labels]).flatten()
plt.plot(generate_signal_2[500:700])
plt.show()

# #---------------产生ACGAN的信号(4)
generator_4 = load_model('Well_trained_CGANS_models/ACGAN_60000_generator')
generate_signal_4 = generator_4.predict([noise0, labels]).flatten()
plt.plot(generate_signal_4[500:700])
plt.show()

# #---------------产生WCGAN的信号(5）
generator_5 = load_model('Well_trained_CGANS_models/WCGAN_60000_generator')
generate_signal_5 = generator_5.predict([noise0, labels]).flatten()
plt.plot(generate_signal_5[500:700])
plt.show()

# np.save('generator_signal_1(0)', generate_signal_1)
# np.save('generator_signal_2(0)', generate_signal_2)
# np.save('generator_signal_3(0)', generate_signal_3)
# np.save('generator_signal_4(0)', generate_signal_4)
# np.save('generator_signal_5(0)', generate_signal_5)

x = np.load('Data_now/X_TrainData.npy')

x_0 = x[0*10+3,:]
plt.subplot(6, 6, 1)
plt.plot(x_0[500:700])
plt.title("plot 1")

signal1_0 = np.load('generator_signal_1(0).npy')
plt.subplot(6, 6, 2)
plt.plot(signal1_0[500:700])
plt.title("plot 2")

signal2_0 = np.load('generator_signal_2(0).npy')
plt.subplot(6, 6, 3)
plt.plot(signal2_0[500:700])
plt.title("plot 3")

signal3_0 = np.load('generator_signal_3(0).npy')
plt.subplot(6, 6, 4)
plt.plot(signal3_0[500:700])
plt.title("plot 4")

signal4_0 = np.load('generator_signal_4(0).npy')
plt.subplot(6, 6, 5)
plt.plot(signal4_0[500:700])
plt.title("plot 5")

signal5_0 = np.load('generator_signal_5(0).npy')
plt.subplot(6, 6, 6)
plt.plot(signal5_0[500:700])
plt.title("plot 6")

x_5 = x[5*10+3,:]
plt.subplot(6, 6, 7)
plt.plot(x_5[500:700])
plt.title("plot 7")

signal1_5 = np.load('generator_signal_1(5).npy')
plt.subplot(6, 6, 8)
plt.plot(signal1_5[500:700])
plt.title("plot 8")

signal2_5 = np.load('generator_signal_2(5).npy')
plt.subplot(6, 6, 9)
plt.plot(signal2_5[500:700])
plt.title("plot 9")

signal3_5 = np.load('generator_signal_3(5).npy')
plt.subplot(6, 6, 10)
plt.plot(signal3_5[500:700])
plt.title("plot 10")

signal4_5 = np.load('generator_signal_4(5).npy')
plt.subplot(6, 6, 11)
plt.plot(signal4_5[500:700])
plt.title("plot 11")

signal5_5 = np.load('generator_signal_5(5).npy')
plt.subplot(6, 6, 12)
plt.plot(signal5_5[500:700])
plt.title("plot 12")

x_20 = x[20*10+3,:]
plt.subplot(6, 6, 13)
plt.plot(x_20[500:700])
plt.title("plot 13")

signal1_20 = np.load('generator_signal_1(20).npy')
plt.subplot(6, 6, 14)
plt.plot(signal1_20[500:700])
plt.title("plot 14")

signal2_20 = np.load('generator_signal_2(20).npy')
plt.subplot(6, 6, 15)
plt.plot(signal2_20[500:700])
plt.title("plot 15")

signal3_20 = np.load('generator_signal_3(20).npy')
plt.subplot(6, 6, 16)
plt.plot(signal3_20[500:700])
plt.title("plot 16")

signal4_20 = np.load('generator_signal_4(20).npy')
plt.subplot(6, 6, 17)
plt.plot(signal4_20[500:700])
plt.title("plot 17")

signal5_20 = np.load('generator_signal_5(20).npy')
plt.subplot(6, 6, 18)
plt.plot(signal5_20[500:700])
plt.title("plot 18")

x_23 = x[23*10+3,:]
plt.subplot(6, 6, 19)
plt.plot(x_23[500:700])
plt.title("plot 19")

signal1_23 = np.load('generator_signal_1(23).npy')
plt.subplot(6, 6, 20)
plt.plot(signal1_23[500:700])
plt.title("plot 20")

signal2_23 = np.load('generator_signal_2(23).npy')
plt.subplot(6, 6, 21)
plt.plot(signal2_23[500:700])
plt.title("plot 21")

signal3_23 = np.load('generator_signal_3(23).npy')
plt.subplot(6, 6, 22)
plt.plot(signal3_23[500:700])
plt.title("plot 22")

signal4_23 = np.load('generator_signal_4(23).npy')
plt.subplot(6, 6, 23)
plt.plot(signal4_23[500:700])
plt.title("plot 23")

signal5_23 = np.load('generator_signal_5(23).npy')
plt.subplot(6, 6, 24)
plt.plot(signal5_23[500:700])
plt.title("plot 24")

x_31 = x[31*10+3,:]
plt.subplot(6, 6, 25)
plt.plot(x_31[500:700])
plt.title("plot 25")

signal1_31 = np.load('generator_signal_1(31).npy')
plt.subplot(6, 6, 26)
plt.plot(signal1_31[500:700])
plt.title("plot 26")

signal2_31 = np.load('generator_signal_2(31).npy')
plt.subplot(6, 6, 27)
plt.plot(signal2_31[500:700])
plt.title("plot 27")

signal3_31 = np.load('generator_signal_3(31).npy')
plt.subplot(6, 6, 28)
plt.plot(signal3_31[500:700])
plt.title("plot 28")

signal4_31 = np.load('generator_signal_4(31).npy')
plt.subplot(6, 6, 29)
plt.plot(signal4_31[500:700])
plt.title("plot 29")

signal5_31 = np.load('generator_signal_5(31).npy')
plt.subplot(6, 6, 30)
plt.plot(signal5_31[500:700])
plt.title("plot 30")

x_49 = x[49*10+3,:]
plt.subplot(6, 6, 31)
plt.plot(x_49[500:700])
plt.title("plot 31")

signal1_49 = np.load('generator_signal_1(49).npy')
plt.subplot(6, 6, 32)
plt.plot(signal1_49[500:700])
plt.title("plot 32")

signal2_49 = np.load('generator_signal_2(49).npy')
plt.subplot(6, 6, 33)
plt.plot(signal2_49[500:700])
plt.title("plot 33")

signal3_49 = np.load('generator_signal_3(49).npy')
plt.subplot(6, 6, 34)
plt.plot(signal3_49[500:700])
plt.title("plot 34")

signal4_49 = np.load('generator_signal_4(49).npy')
plt.subplot(6, 6, 35)
plt.plot(signal4_49[500:700])
plt.title("plot 35")

signal5_49 = np.load('generator_signal_5(49).npy')
plt.subplot(6, 6, 36)
plt.plot(signal5_49[500:700])
plt.title("plot 36")

plt.suptitle("RUNOOB subplot Test")
plt.show()
