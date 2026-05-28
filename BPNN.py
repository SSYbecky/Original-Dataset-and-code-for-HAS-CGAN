import numpy as np
import keras
from keras.layers import Input,Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler

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

input1 = Input(shape=(1024,))
n_first_middle_layer = 64
# n_second_layer = 16
# n_third_layer = 32
first_middle_layer = Dense(n_first_middle_layer,activation = 'sigmoid')(input1)
# second_middle_layer = Dense(n_second_layer, activation = 'sigmoid')(first_middle_layer)
# third_middle_layer = Dense(n_third_layer, activation = 'sigmoid')(second_middle_layer)

last_layer = Dense(1, activation = 'linear')(first_middle_layer)
model = Model(inputs = input1, outputs = last_layer)
opt = keras.optimizers.Adam()
model.compile(loss='mse', optimizer=opt, metrics=['mean_absolute_percentage_error'])
# callbacks = [
#         keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#     ]

# train_history = model.fit(X_train_original_std,
#                         Y_train_original_std,
#                         batch_size=32, #调大
#                         epochs=300,
#                         # callbacks=callbacks,
#                         validation_data=(X_test_original_std, Y_test_original_std),
#                         shuffle=True)

# y_pred = model.predict(X_test_original_std)
# np.save('Trans_original_31.4%', y_pred)

#训练模型
train_history = model.fit(X_train_10times_std,
                        Y_train_10times_std,
                        batch_size=64, #调大
                        epochs=200,
                        validation_data=(X_test_10times_std, Y_test_10times_std),
                        shuffle=True)

y_pred_std = model.predict(X_test_10times_std)
# np.save('CNN_10times_13.0%.npy', y_pred_std)

# def calculate_mape(y_true, y_pred):
#     y_true, y_pred = np.array(y_true), np.array(y_pred)

#     # 避免除以零（如果真实值中有0，可以加一个极小值或过滤掉）
#     non_zero_mask = y_true != 0
#     y_true = y_true[non_zero_mask]
#     y_pred = y_pred[non_zero_mask]

#     if len(y_true) == 0:
#         return np.nan  # 如果所有真实值都是0，返回NaN

#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     return mape

# mape = calculate_mape(Y_test_10times_std, Y_pred)
# print(f"MAPE: {mape:.2f}%")
# np.save('10times_CNN_8.95%',Y_pred)
#
# original = np.load('Trans_original_31.4%.npy')
# times_5 = np.load('Trans_5times_21.1%.npy')
# times_7 = np.load('Trans_7times_15.3%.npy')
# times_10 = np.load('10times_CNN_8.95%.npy')
