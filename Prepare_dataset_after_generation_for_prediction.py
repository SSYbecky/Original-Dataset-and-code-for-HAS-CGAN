import numpy as np
from keras.models import load_model

X_train_original_51 = np.load('Original_dataset/Original_train_and_test_data/X_train_original_51.npy')
X_test_original_12 = np.load('Original_dataset/Original_train_and_test_data/X_test_original_12.npy')
Y_train_original_51 = np.load('Original_dataset/Original_train_and_test_data/Y_train_original_51.npy')
Y_test_original_12 = np.load('Original_dataset/Original_train_and_test_data/Y_test_original_12.npy')

## 生成新的数据
latent_dim = 100


def generate_noise():
    a = np.arange(0.1, 1.5, 0.02)
    x = np.arange(0, latent_dim / 2, 0.5)  # 128/2=64
    r = np.arange(0.1, 6.1, 0.1)
    fs = len(x)
    y = np.zeros((1, len(x)))
    amp = 10 * a[np.random.randint(0, len(a) - 1)]
    rad = r[np.random.randint(0, len(r) - 1)]
    phase = np.random.uniform(-1, 1) * np.pi
    y = np.append(y, amp * np.sin(((2 * np.pi * rad * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise1 = np.delete(y, 0, axis=0)

    r1 = np.arange(6.1, 12.1, 0.1)
    y1 = np.zeros((1, len(x)))
    rad1 = r[np.random.randint(0, len(r1) - 1)]
    y1 = np.append(y1, amp * np.sin(((2 * np.pi * rad1 * x) + phase) / fs).reshape((1, len(x))), axis=0)
    noise2 = np.delete(y1, 0, axis=0)
    noise = noise1 + noise2
    return noise


# #-------------- 修改后的高效数据扩增函数 --------------# #

def enrich_your_dataset_for_n_times(n):
    # 【优化 1】将模型加载移到循环外面，只加载一次
    generator = load_model('Well_trained_CGANS_models/HAS_CGAN_60000_generator')

    signal_blocks = []
    label_blocks = []

    for index in range(51):
        # 【优化 2】批量生成当前 label 所需的 n 个噪声，并组合在一起
        # generate_noise() 返回形状为 (1, 100) 的数组，vstack 后形状为 (n, 100)
        noises = [generate_noise() for _ in range(n)]
        noise_batch = np.vstack(noises)

        # 创建对应的批量标签，形状为 (n, 1)，里面的值全为当前 index
        labels_batch = np.full((n, 1), index)

        # 【优化 3】利用 Keras 的矩阵并行能力进行批量预测！
        # 得到生成的信号矩阵，形状为 (n, 信号特征长度)
        generated_signals = generator.predict([noise_batch, labels_batch])

        # 将当前 label 的数据块和标签块保存到列表中
        signal_blocks.append(generated_signals)
        label_blocks.append(labels_batch)

    # 【核心修改】使用 np.vstack 将 51 个数据块按顺序垂直拼接起来
    # 拼接后的最终矩阵形状为 (51 * n, 信号特征长度)
    # 结构完全符合：
    # 0 到 n-1 行（第 1 ~ n 行）    -> label 0 的 n 倍数据
    # n 到 2n-1 行（第 n+1 ~ 2n 行） -> label 1 的 n 倍数据 …… 以此类推
    final_signals_array = np.vstack(signal_blocks)
    final_labels_array = np.vstack(label_blocks)

    return final_signals_array, final_labels_array

# -------------- 如何调用该函数 -------------- # #
if __name__ == "__main__":
    n = 10  # 假设扩增 10 倍
    X_new, Y_new = enrich_your_dataset_for_n_times(n)
    print("扩增后的数据矩阵形状:", X_new.shape)  # 应该是 (255, 信号长度)
    print("扩增后的标签矩阵形状:", Y_new.shape)    # 应该是 (255, 1)
