import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

X_train_original_features_std = np.load('X_train_original_features_std.npy')
X_test_original_features_std = np.load('X_test_original_features_std.npy')
X_train_10times_features_std = np.load('X_train_10times_features_std.npy')
X_test_10times_features_std = np.load('X_test_10times_features_std.npy')
Y_original = np.load('Y_original.npy')
Y_10times = np.load('Y_10times.npy')
# 数据标准化1
# scaler_y = StandardScaler()
# Y_original_std = scaler_y.fit_transform(Y_original)
Y_train_original = Y_original[:55,:]
Y_test_original = Y_original[55:,:]

# 数据标准化2
# scaler_y1 = StandardScaler()
# Y_10times_std = scaler_y1.fit_transform(Y_10times)
Y_train_10times = Y_10times[:550,:]
Y_test_10times = Y_10times[550:,:]


#计算MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 避免除以零（如果真实值中有0，可以加一个极小值或过滤掉）
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]

    if len(y_true) == 0:
        return np.nan  # 如果所有真实值都是0，返回NaN

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

# # 训练SVR模型-----------------------------------------------------------
svr_model = SVR(kernel='rbf', C=100, epsilon=1)  # 使用RBF核
svr_model.fit(X_train_original_features_std, Y_train_original)

# 预测

y_pred_original = svr_model.predict(X_test_original_features_std).reshape(-1, 1)
# Y_pred_original = scaler_y.inverse_transform(y_pred_scaled_original)  # 反标准化
# Y_test_original = scaler_y.inverse_transform(Y_test_original_std)

MAPE_original_features = calculate_mape(Y_test_original, y_pred_original)


svr_model1 = SVR(kernel='rbf', C=100, epsilon=0.1)  # 使用RBF核
svr_model1.fit(X_train_10times_features_std, Y_train_10times)
#
# # 预测
y_pred_10times = svr_model1.predict(X_test_10times_features_std).reshape(-1, 1)
# Y_pred_10times = scaler_y1.inverse_transform(y_pred_scaled_10times)  # 反标准化
# Y_test_10times = scaler_y1.inverse_transform(Y_test_10times_std)
# #计算MAPE
MAPE_10times_features = calculate_mape(Y_test_10times, y_pred_10times)
#--------------------------------------------------------------------------------

# 训练LSTM模型----------------------------------------------------------------------
# 调整维度（假设time_step=1)
X_train_original_3d = np.expand_dims(X_train_original_features_std, axis=1)
X_test_original_3d = np.expand_dims(X_test_original_features_std, axis=1)
X_train_10times_3d = np.expand_dims(X_train_10times_features_std, axis=1)
X_test_10times_3d = np.expand_dims(X_test_10times_features_std, axis=1)
# 构建 LSTM 模型
from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 17)))  # 第一层需指定input_shape
model.add(Dense(1))  # 输出层

model.compile(optimizer='adam', loss='mse')
model.summary()
# 训练模型
model.fit(X_train_original_3d, Y_train_original, epochs=100, batch_size=8, verbose=1)
y_pred_original = model.predict(X_test_original_3d)

# #计算MAPE
MAPE_original_features = calculate_mape(Y_test_original, y_pred_original)

model1 = Sequential()
model1.add(LSTM(50, activation='relu', input_shape=(1, 17)))  # 第一层需指定input_shape
model1.add(Dense(1))  # 输出层

model1.compile(optimizer='adam', loss='mse')
model1.summary()
# 训练模型
model1.fit(X_train_10times_3d, Y_train_10times, epochs=100, batch_size=8, verbose=1)
y_pred_10times = model1.predict(X_test_10times_3d)
# #计算MAPE
MAPE_10times_features = calculate_mape(Y_test_10times, y_pred_10times)

# 创建 Random Forest 模型
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_original_features_std, Y_train_original)
# 预测
Y_pred_original = rf.predict(X_test_original_features_std).reshape(-1,1)

MAPE_original_features = calculate_mape(Y_test_original, Y_pred_original)


rf1 = RandomForestRegressor(n_estimators=5, random_state=42)
rf1.fit(X_train_10times_features_std, Y_train_10times)
# 预测
Y_pred_10times = rf1.predict(X_test_10times_features_std).reshape(-1,1)

MAPE_10times_features = calculate_mape(Y_test_10times, Y_pred_10times)



