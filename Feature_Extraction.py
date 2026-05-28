import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.fftpack import fft
import pandas as pd

def feature_extraction(data):
    Xp1 = max(max(data), -min(data)) #峰值指标 1
    Xrms1 = np.sqrt(np.dot(data, data) / len(data)) #均方根值 2
    Xpm1 = Xp1/Xrms1 #峰值因子 3
    Xqd1 = stats.kurtosis(data) #峭度因子 4
    Xfd1 = stats.skew(data) #斜度 5
    Xr1 = np.mean(np.sqrt(abs(data))) * np.mean(np.sqrt(abs(data))) #方根幅度 6
    Xy1 = Xp1/Xr1 #裕度系数 7
    Xmean1 = np.mean(data) #平均值 8
    Xmc1 = Xp1/np.mean(abs(data)) #脉冲因子 9
    Xbx1 = Xrms1/np.mean(data) #波形指标 10
    Xvar1 = np.var(data) #方差 11
    Xstd1 = np.std(data) #标准差 12
    feature = np.array((Xp1, Xrms1, Xpm1, Xqd1, Xfd1, Xr1, Xy1, Xmean1, Xmc1, Xbx1, Xvar1, Xstd1))
    # feature = np.array(feature).reshape(1,-1)
    return feature

def frequency_domain_feature(data):
    fs = 10000
    N = data.shape[0] #计算点数
    y = 2*abs(fft(data))/N #fft变换
    # y = fft(data)
    f = np.array([(j + 1) * fs / N for j in range(N)])
    FC = np.dot(f, y) / np.sum(y) #重心频率
    MSF = np.dot(np.multiply(f, f), y) / np.sum(y)#均方频率
    RMSF = np.sqrt(MSF) #均方根频率
    VF = np.dot(np.multiply(f- FC, f - FC), y) / np.sum(y) #频率方差
    RVF = np.sqrt(VF) #频率标准差
    fea = [FC,MSF,RMSF,VF,RVF]
    return fea



















