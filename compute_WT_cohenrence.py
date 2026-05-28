from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy import stats


x = np.load('Data_now/X_TrainData.npy')
x_0 = x[5*10+3,:][500:700]
signal1_0 = np.load('generator_signal_1(5).npy')[500:700]
signal2_0 = np.load('generator_signal_2(5).npy')[500:700]
signal3_0 = np.load('generator_signal_3(5).npy')[500:700]
signal4_0 = np.load('generator_signal_4(5).npy')[500:700]
signal5_0 = np.load('generator_signal_5(5).npy')[500:700]

import numpy as np
import pywt

def wavelet_correlation(signal1, signal2, wavelet='db4', level=4):
    # 进行小波分解
    coeffs1 = pywt.wavedec(signal1, wavelet, level=level)
    coeffs2 = pywt.wavedec(signal2, wavelet, level=level)

    # 计算每个层级的相关性
    correlations = [np.corrcoef(c1, c2)[0, 1] for c1, c2 in zip(coeffs1, coeffs2)]

    return correlations

cor = wavelet_correlation(x_0, signal5_0)


