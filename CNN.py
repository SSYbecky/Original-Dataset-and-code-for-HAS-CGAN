import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import time
time_start = time.time()

#加载训练集
X_train_10times_std = np.load('Train_and_test_dataset_for_prediction_models/X_train_10times_std.npy')
X_test_10times_std = np.load('Train_and_test_dataset_for_prediction_models/X_test_10_times_std.npy')
Y_10times = np.load('Train_and_test_dataset_for_prediction_models/Y_10times.npy')
Y_train_10times_std = Y_10times[:550,:]
Y_test_10times_std = Y_10times[550:,:]

X_train_original_std = np.load('Train_and_test_dataset_for_prediction_models/X_train_original_std.npy')
X_test_original_std = np.load('Train_and_test_dataset_for_prediction_models/X_test_original_std.npy')
Y_original = np.load('Train_and_test_dataset_for_prediction_models/Y_original.npy')
Y_train_original_std = Y_original[:55,:]
Y_test_original_std = Y_original[55:,:]

#建模
input = Input(shape=(1024,1))
x = Conv1D(3,4, activation='relu',padding='same')(input)
x = MaxPooling1D(30,padding='same')(x)
x = Flatten()(x)
x = Dense(5,activation='sigmoid')(x)
x = Dense(1,activation='linear')(x)
model = Model(inputs=input, outputs=x)
opt = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=opt, metrics=['mean_absolute_percentage_error'])
print(model.summary())

train_history = model.fit(x=X_train_10times_std,
                        y=Y_train_10times_std,
                        batch_size=32, #调大
                        epochs=1000,
                        validation_data=(X_test_10times_std, Y_test_10times_std),
                        shuffle=False)


y_pred = model.predict(X_test_original_std)
# np.save('BPNN_7times_22.2%', y_pred)